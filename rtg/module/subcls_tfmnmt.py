#!/usr/bin/env python
#

import torch.nn as nn
import torch.nn.functional as F
from abc import ABC
from typing import List, Optional

from rtg.module.tfmnmt import (EncoderLayer, Decoder, DecoderLayer, PositionwiseFeedForward, MultiHeadedAttention,
                               TransformerNMT, TransformerTrainer, Generator)
from rtg.emb.tfmcls import SentenceCompressor

from rtg import TranslationExperiment as Experiment, log
from rtg.utils import get_my_args
from rtg.registry import register, MODEL



MODEL_NAME = 'subcls_tfmnmt'


class SubClassGenerator(Generator):

    def forward(self, x, score=None, sub_select=None, **kwargs):
        assert not kwargs
        if score == 'embedding':
            return x
        if sub_select is None:
            return super(SubClassGenerator, self).forward(x, score=score)
        # assert sub_select is a list of IDs to pick
        proj_weight = self.proj.weight[sub_select]
        proj_bias = self.proj.bias[sub_select]
        logits = F.linear(x, weight=proj_weight, bias=proj_bias)
        return self.scores[score](logits)


class SubClassDecoder(Decoder):

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


@register(MODEL, name=MODEL_NAME)
class SubClassTfmNMT(TransformerNMT, ABC):
    DecoderFactory = SubClassDecoder
    GeneratorFactory = SubClassGenerator
    model_type = MODEL_NAME

    def __init__(self, *args, **kwargs):
        super(SubClassTfmNMT, self).__init__(*args, **kwargs)
        d_model = self.model_dim
        attn = MultiHeadedAttention(h=8, d_model=d_model)
        # self.compressor = SentenceCompressor(d_model=d_model, attn=attn)

    def forward(self, src, tgt, src_mask, tgt_mask, gen_probs=False, log_probs=True):
        "Take in and process masked src and target sequences."
        enc_outs = self.encode(src, src_mask)
        # sent_repr = self.compressor(enc_outs, src_mask)
        feats = self.decode(enc_outs, src_mask, tgt, tgt_mask)
        # tgt_vocabs = self.generator(sent_repr, score='sigmoid')
        # tgt.copy_(tgt_rev_idx)  # remap Ids
        if not gen_probs:
            return feats
        score = 'log_softmax' if log_probs else 'softmax'
        tgt_sub_vocab, tgt_rev_idx = tgt.unique(return_inverse=True)
        tgt.copy_(tgt_rev_idx)  # remap
        return self.generator(feats, score=score, sub_select=tgt_sub_vocab)



def __test_model__():
    from rtg.data.dummy import DummyExperiment
    from rtg import Batch, my_tensor as tensor

    vocab_size = 24
    args = {
        'src_vocab': vocab_size,
        'tgt_vocab': vocab_size,
        'enc_layers': 2,
        'dec_layers': 4,
        'hid_size': 32,
        'eff_dims': [16, 24],
        'dff_dims': [64, 128, 128, 64],
        'n_heads': 4,
        'activation': 'relu'
    }

    from rtg.module.decoder import Decoder

    config = {
        'model_type': 'wvtfmnmt',
        'trainer': {'init_args': {'chunk_size': 2, 'grad_accum': 1}},
        'optim': {
            'args': {
                # "cross_entropy", "smooth_kld", "binary_cross_entropy", "triplet_loss"
                'criterion': "smooth_kld",
                'lr': 0.01,
                'inv_sqrt': True
            }
        }
    }

    exp = DummyExperiment("work.tmp.wvtfmnmt", config=config, read_only=True,
                          vocab_size=vocab_size)
    exp.model_args = args
    trainer = SubClassTfmNMT.make_trainer(exp=exp, warmup_steps=200, **config['optim']['args'])
    decr = Decoder.new(exp, trainer.model)

    assert 2 == Batch.bos_val
    src = tensor([[4, 5, 6, 7, 8, 9, 10, 11, 12, 13, Batch.eos_val, Batch.pad_value],
                  [13, 12, 11, 10, 9, 8, 7, 6, Batch.eos_val, Batch.pad_value, Batch.pad_value,
                   Batch.pad_value]])
    src_lens = tensor([src.size(1)] * src.size(0))

    def check_pt_callback(**args):
        res = decr.greedy_decode(src, src_lens, max_len=12)
        for score, seq in res:
            log.info(f'{score:.4f} :: {seq}')

    batch_size = 50
    steps = 1000
    check_point = 50
    trainer.train(steps=steps, check_point=check_point, batch_size=batch_size,
                  check_pt_callback=check_pt_callback)


if __name__ == '__main__':
    __test_model__()
