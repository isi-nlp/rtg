#!/usr/bin/env python
#
# Author: Thamme Gowda [tg (at) isi (dot) edu] 
# Created: 2/15/19

import torch
import torch.nn as nn
from typing import List, Mapping
from rtg.module.tfmnmt import (TransformerNMT, EncoderLayer, DecoderLayer, Generator)
from rtg import TranslationExperiment as Experiment, log
from rtg.data.codec import PretrainMatchField
from rtg.utils import get_my_args

class RobertaGenerator(Generator):
    """
    Trying to match with XLM-R / Roberta of Fairseq which has one extra dense layer than
    std Generator
    """

    def __init__(self, d_model: int, vocab: int, activation='gelu'):
        super().__init__(d_model, vocab)
        self.dense = nn.Linear(d_model, d_model)
        self.layer_norm = nn.LayerNorm(d_model)
        self.activation = dict(relu=nn.ReLU, gelu=nn.GELU, elu=nn.ELU,
                               leaky_relu=nn.LeakyReLU)[activation]

    def forward(self, x, score=None, *args, **kwargs):
        assert not args, f'Support for {args} are removed. Please use "{score}" argument'
        if score not in ('embedding', 'identity'):
            x = self.dense(x)
            x = self.activation(x)
            x = self.layer_norm(x)
        return super().forward(x, score=score, **kwargs)


class RoBERTaMT(TransformerNMT):
    GeneratorFactory = RobertaGenerator
    model_type = 'robertamt'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert isinstance(self.generator, RobertaGenerator)
        self.generator.activation = self.encoder.layers[0].feed_forward.activation

    @classmethod
    def make_model(cls, src_vocab:int, tgt_vocab:int, model_id='pytorch/fairseq:xlmr.base',
                   inner_args=None, init=True, exp: Experiment = None,
                   enc_layer_map=None, dec_layer_map=None):
        """
        Helper: Construct a model from hyper parameters."
        :return: model, args
        """
        save_args = get_my_args(exclusions=['exp'])
        if not inner_args:  # extract from pretrained model the first time
            log.info(f"making {cls.model_type} model from {model_id}")
            group, model_name = model_id.split(':')
            hub_api = torch.hub.load(group, model_name)
            # hub_api = torch.hub.load('pytorch/fairseq', 'xlmr.base.v0')
            roberta: 'RobertaEncoder' = hub_api.model.decoder
            roberta.eval()
            ma = roberta.args
            n_layers_max = ma.encoder_layers
            def validate_layer_idxs(layer_idxs):
                if layer_idxs: # check list of integers and each idx is valid
                    assert isinstance(layer_idxs, list) and isinstance(layer_idxs[0], int)
                    assert all(0 <= idx < n_layers_max for idx in layer_idxs)
                else: # default one to one copy
                    layer_idxs = list(range(n_layers_max))
                return layer_idxs
            save_args['enc_layer_map'] = enc_layer_map = validate_layer_idxs(enc_layer_map)
            save_args['dec_layer_map'] = dec_layer_map = validate_layer_idxs(dec_layer_map)

            inner_args = dict(
                src_vocab=src_vocab,  # rows in
                tgt_vocab=tgt_vocab,
                enc_layers=len(enc_layer_map),
                dec_layers=len(dec_layer_map),
                hid_size=ma.encoder_embed_dim,
                ff_size=ma.encoder_ffn_embed_dim,
                n_heads=ma.encoder_attention_heads,
                attn_bias=True,
                attn_dropout=ma.attention_dropout,
                dropout=ma.dropout,
                activation=ma.activation_fn,
                tied_emb='three-way')

        assert isinstance(exp.tgt_vocab, PretrainMatchField)
        assert isinstance(exp.src_vocab, PretrainMatchField)
        xlmmt, inner_args = super().make_model(exp=exp, **inner_args)
        save_args['inner_args'] = inner_args
        return xlmmt, save_args

    @classmethod
    def map_rows(cls, src, dest, mapping):
        skips = 0
        for dest_idx, src_idx in mapping.items():
            if dest_idx < 0 or src_idx < 0:
                skips += 1
            dest[dest_idx] = src[src_idx]
        log.info(f"Mapped rows. Total skips = {skips}")

    def maybe_init_from_parent(self, exp: Experiment):

        tgt_emb_map = exp.tgt_vocab.new_idx2old_idx
        src_emb_map = exp.src_vocab.new_idx2old_idx
        args = exp.model_args
        assert args['inner_args']['src_vocab'] == len(src_emb_map)
        assert args['inner_args']['tgt_vocab'] == len(tgt_emb_map)
        enc_layer_map = args['enc_layer_map']
        dec_layer_map = args['dec_layer_map']
        log.info(f"Initialising self of type {self.model_type} from {args['model_id']}")
        group, model_name = args['model_id'].split(':')
        hub_api = torch.hub.load(group, model_name)
        # hub_api = torch.hub.load('pytorch/fairseq', 'xlmr.base.v0')
        roberta: 'RobertaEncoder' = hub_api.model.decoder
        self.init_from_roberta(roberta, args['init'],
                               src_emb_map=src_emb_map, tgt_emb_map=tgt_emb_map,
                               enc_layer_map=enc_layer_map, dec_layer_map=dec_layer_map)

    def init_from_roberta(self, roberta: 'RobertaEncoder', init: List[str],
                          src_emb_map: Mapping[int, int],
                          tgt_emb_map: Mapping[int, int],
                          enc_layer_map: List[int], dec_layer_map: List[int]):
        """

        :param roberta:
        :param init: List of component names to be initialized
        :param src_emb_map: map[my_emb_idx <-- your_emb_idx] for source language
        :param tgt_emb_map: map[my_emb_idx <-- your_emb_idx] for target language
        :param enc_layer_map: List of layer indices of pretrained model
                    from which the encoder weights be copied from
        :param dec_layer_map: List of layer indices of pretrained model
                    from which the decoder layer weights be copied from
        :return:
        """
        # fairseq stuff; imports are not really needed. Placed here just in case for inspection

        assert len(self.encoder.layers) == len(enc_layer_map)
        assert len(self.decoder.layers) == len(dec_layer_map)

        from fairseq.modules.transformer_sentence_encoder_layer import \
            TransformerSentenceEncoderLayer
        from fairseq.modules.multihead_attention import MultiheadAttention as YourAttn

        pieces = {'all', 'src_in_emb', 'tgt_in_emb', 'tgt_out_emb', 'enc_layers', 'dec_layers',
                  'generator_dense'}
        for it in init:
            assert it in pieces, f'Valid args are {pieces}'

        log.info("Initialising model's weights with roberta's pretrained weights")
        rob = roberta
        rob_enc = rob.sentence_encoder
        rob_embs = rob_enc.embed_tokens.weight
        if 'all' in init or 'src_in_emb' in init:
            log.info("init src_in_emb: YES")
            self.map_rows(rob_embs.data, self.src_embed[0].lut.weight.data, mapping=src_emb_map)
        else:
            log.info("init src_in_emb: NO")

        if ('all' in init or 'tgt_in_emb' in init):
            log.info("init tgt_in_emb: YES")
            #self.tgt_embed[0].lut.weight.data.copy_(rob_embs.data)
            self.map_rows(rob_embs.data, self.tgt_embed[0].lut.weight.data, mapping=tgt_emb_map)
        else:
            log.info("init tgt_in_emb: NO")

        if 'all' in init or 'tgt_out_emb' in init:
            log.info("init tgt_out_emb: YES")
            self.map_rows(rob.lm_head.weight.data, self.generator.proj.weight.data, mapping=tgt_emb_map)
            self.map_rows(rob.lm_head.bias.data, self.generator.proj.bias.data, mapping=tgt_emb_map)
            #self.generator.proj.weight.data.copy_(rob.lm_head.weight.data)
            #self.generator.proj.bias.data.copy_(rob.lm_head.bias.data)
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
            for my_idx, ur_idx in enumerate(enc_layer_map):
                log.info(f"Initialize self.encoder[{my_idx}] <-- pretrained.encoder[{ur_idx}]")
                ur_layer = rob_enc.layers[ur_idx]
                my_layer = self.encoder.layers[my_idx]
                init_enc_layer(ur_layer, my_layer)
            copy_weights(rob.lm_head.layer_norm, self.encoder.norm)
        else:
            log.info("init enc_layers: NO")

        if 'all' in init or 'dec_layers' in init:
            log.info("init dec_layers: YES")
            for my_idx, ur_idx in enumerate(dec_layer_map):
                log.info(f"Initialize self.decoder[{my_idx}] <-- pretrained.encoder[{ur_idx}]")
                ur_layer = rob_enc.layers[ur_idx]
                my_layer = self.decoder.layers[my_idx]
                init_dec_layer(ur_layer, my_layer)
            log.warning("decoder src_attn is initialized from encoder's self_attn ")
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
    model = RoBERTaMT.make_model()


if __name__ == '__main__':
    __test_model__()
