#!/usr/bin/env python
#
# Author: Thamme Gowda [tg (at) isi (dot) edu] 
# Created: 2/15/19

import torch
import torch.nn as nn

from rtg.module.tfmnmt import (TransformerNMT, EncoderLayer, DecoderLayer, Generator)
from rtg import TranslationExperiment as Experiment, log

class RobertaGenerator(Generator):
    """
    Trying to match with XLM-R / Roberta of Fairseq which has one extra dense layer than
    std Generator
    """

    def __init__(self, d_model: int, vocab: int, activation='gelu'):
        super().__init__(d_model, vocab)
        self.dense = nn.Linear(d_model, d_model)
        self.activation = None
        self.layer_norm = nn.LayerNorm(d_model)
        self.activation = dict(relu=nn.ReLU, gelu=nn.GELU, elu=nn.ELU,
                               leaky_relu=nn.LeakyReLU)[activation]

    def forward(self, x, score=None, **args):
        assert not args, f'Support for {args} are removed. Please use "{score}" argument'
        if score not in ('embedding', 'identity'):
            x = self.dense(x)
            x = self.activation(x)
            x = self.layer_norm(x)
        return super().forward(x)


class XLMMT(TransformerNMT):
    GeneratorFactory = RobertaGenerator

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert isinstance(self.generator, RobertaGenerator)
        self.generator.activation = self.encoder.layers[0].feed_forward.activation

    @property
    def model_type(self):
        return 'xlmmt'

    @classmethod
    def make_model(cls, name='pytorch/fairseq:xlmr.base', inner_args=None, init=True,
                   exp: Experiment = None):
        """
        Helper: Construct a model from hyper parameters."
        :return: model, args
        """
        log.info(f"making mtfmnmt model: {name}")
        group, model_name = name.split(':')
        hub_api = torch.hub.load(group, model_name)
        # hub_api = torch.hub.load('pytorch/fairseq', 'xlmr.base.v0')
        roberta: 'RobertaEncoder' = hub_api.model.decoder
        roberta.eval()
        ma = roberta.args
        inner_args = inner_args or {}
        args = dict(
            src_vocab=len(roberta.dictionary),  # rows in
            tgt_vocab=len(roberta.dictionary),
            enc_layers=ma.encoder_layers,
            dec_layers=ma.encoder_layers,
            hid_size=ma.encoder_embed_dim,
            ff_size=ma.encoder_ffn_embed_dim,
            n_heads=ma.encoder_attention_heads,
            attn_bias=inner_args.get('attn_bias', True),
            attn_dropout=inner_args.get('attn_dropout', ma.attention_dropout),
            dropout=inner_args.get('dropout', ma.dropout),
            activation=inner_args.get('activation', ma.activation_fn),
            tied_emb=inner_args.get('tied_emb', 'three-way'))

        xlmmt, inner_args = super().make_model(**args, exp=exp)
        if init:
            xlmmt.init_from_roberta(roberta)
        return xlmmt, dict(name=name, inner_args=inner_args, init=init)

    def init_from_roberta(self, roberta: 'RobertaEncoder', init):
        # fairseq stuff; imports are not really needed. Placed here just in case for inspection
        """
        from fairseq.modules.transformer_sentence_encoder_layer import \
            TransformerSentenceEncoderLayer
        from fairseq.models.roberta import RobertaModel, RobertaEncoder, TransformerSentenceEncoder, \
            RobertaLMHead
        from fairseq.modules.multihead_attention import MultiheadAttention as YourAttn
        """
        pieces = {'all', 'src_in_emb', 'tgt_in_emb', 'tgt_out_emb', 'enc_layers', 'dec_layers',
                  'generator_dense'}
        for it in init:
            assert it in pieces, f'Valid args are {pieces}'

        log.info("Initialising model's weights with roberta's pretrained weights")
        rob = roberta
        rob_enc = rob.sentence_encoder
        rob_embs = rob_enc.embed_tokens.weight

        log.info("Initialized SrcOutEmb with pretrained")
        init = {'src_in_emb', 'tgt_in_emb', 'tgt_out_emb', ''}
        if 'all' in init or 'src_in_emb' in init:
            log.info("init src_in_emb: YES")
            self.src_embed[0].lut.weight.data.copy_(rob_embs.data)
        else:
            log.info("init src_in_emb: NO")

        if ('all' in init or 'tgt_in_emb' in init):
            log.info("init tgt_in_emb: YES")
            self.tgt_embed[0].lut.weight.data.copy_(rob_embs.data)
        else:
            log.info("init tgt_in_emb: NO")

        if 'all' in init or 'tgt_out_emb' in init:
            log.info("init tgt_out_emb: YES")
            self.generator.proj.weight.data.copy_(rob.lm_head.weight.data)
            self.generator.proj.bias.data.copy_(rob.lm_head.bias.data)
        else:
            log.info("init tgt_out_emb: NO")

        from rtg.module.tfmnmt import MultiHeadedAttention as MyAttn
        def copy_weights(source: nn.Module, target: nn.Module):
            assert type(target) is type(source)
            target.load_state_dict(source.state_dict())

        def init_multi_head_attn(you: 'YourAttn', me: MyAttn):
            for my, ur in zip(me.linears, [you.q_proj, you.k_proj, you.v_proj, you.out_proj]):
                copy_weights(ur, my)

        def init_enc_layer(ur: 'TransformerSentenceEncoderLayer', my: EncoderLayer):
            # self attention
            init_multi_head_attn(ur.self_attn, my.self_attn)
            # Feed forward
            copy_weights(ur.fc1, my.feed_forward.w_1)
            copy_weights(ur.fc2, my.feed_forward.w_2)
            # Layer Norms
            copy_weights(ur.self_attn_layer_norm, my.sublayer[0].norm)
            copy_weights(ur.final_layer_norm, my.sublayer[-1].norm)

        def init_dec_layer(ur: 'TransformerSentenceEncoderLayer', my: DecoderLayer):
            init_enc_layer(ur, my)
            # src attention is missing on "your" module, reusing self attn and its layer norm
            init_multi_head_attn(ur.self_attn, my.src_attn)
            copy_weights(ur.self_attn_layer_norm, my.sublayer[1].norm)

        # encoder
        if 'all' in init or 'enc_layers' in init:
            log.info("init enc_layers: YES")
            assert len(self.encoder.layers) == len(rob_enc.layers)
            for ur_layer, my_layer in zip(rob_enc.layers, self.encoder.layers):
                init_enc_layer(ur_layer, my_layer)
            copy_weights(rob.lm_head.layer_norm, self.encoder.norm)
        else:
            log.info("init enc_layers: NO")

        if 'all' in init or 'dec_layers' in init:
            log.info("init dec_layers: YES")
            for ur_layer, my_layer in zip(rob_enc.layers, self.decoder.layers):
                init_dec_layer(ur_layer, my_layer)
            log.warning("decoder src_attn is initialized from encoder self_attn ")
            copy_weights(rob.lm_head.layer_norm, self.decoder.norm)
        else:
            log.info("init dec_layers: NO")

        if 'all' in init or 'generator_dense' in init:
            log.info("init generator_dense: YES")
            copy_weights(rob.lm_head.layer_norm, self.generator.layer_norm)
            # dense layer weights
            copy_weights(rob.lm_head.dense, self.generator.dense)
            # Bias term of output embedding is stored separately in Roberta
        else:
            log.info("init generator_dense: NO")

        """ Note:
        1. enc.embed_positions is unused
        2. enc.emb_layer_norm may exist and it is unused 
        """
        log.info(f"Success: initialized {type(roberta)} to {type(self)}")


def __test_model__():
    model = XLMMT.make_model()


if __name__ == '__main__':
    __test_model__()
