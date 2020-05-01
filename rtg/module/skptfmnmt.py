#!/usr/bin/env python
#
# Author: Lukas J. Ferrer [lferrer (at) isi (dot) edu]
# Created: 2/13/20


import copy
import torch
import torch.nn as nn
from abc import ABC
from typing import List, Optional

from rtg.module import tfmnmt as tfm
from rtg.utils import get_my_args
from rtg import TranslationExperiment as Experiment, log


class SkipEncoder(tfm.Encoder):
    """Core encoder is a stack of N layers"""

    def __init__(self, layer: tfm.EncoderLayer, N: int, depth_probs: List[float] = None):
        super().__init__(layer=layer, N=N)
        if depth_probs:
            assert len(depth_probs) == N
            self.depth_probs = depth_probs
        else:
            self.depth_probs = [1.0 for _ in range(N)]

    def forward(self, x, mask):
        """Pass the input (and mask) through each layer in turn."""
        for layer, run_prob in zip(self.layers, self.depth_probs):
            if not self.training or run_prob >= torch.rand(1).item():
                x = layer(x, mask)
        return self.norm(x)


class SkipDecoder(tfm.Decoder):
    """Generic N layer decoder with masking."""

    def __init__(self, layer: tfm.DecoderLayer, N: int, depth_probs: List[float] = None):
        super().__init__(layer=layer, n_layers=N)
        if depth_probs:
            assert len(depth_probs) == N
            self.depth_probs = depth_probs
        else:
            self.depth_probs = [1.0 for _ in range(N)]

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer, run_prob in zip(self.layers, self.depth_probs):
            if not self.training or run_prob >= torch.rand(1).item():
                x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


class SkipTransformerNMT(tfm.AbstractTransformerNMT, ABC):
    """
    A standard Encoder-Decoder Transformer architecture.
    """

    def __init__(self, encoder: SkipEncoder, decoder: SkipDecoder,
                 src_embed, tgt_embed,
                 generator: Optional[tfm.Generator], tgt_vocab=None):
        super().__init__(encoder=encoder, decoder=decoder,
                         src_embed=src_embed, tgt_embed=tgt_embed,
                         generator=generator, tgt_vocab=tgt_vocab)

    @property
    def model_type(self):
        return 'skptfmnmt'

    @classmethod
    def make_model(cls, src_vocab, tgt_vocab, enc_layers=6, dec_layers=6, hid_size=512, ff_size=2048,
                   n_heads=8, attn_bias=True, attn_dropout=0.1, dropout=0.2, activation='relu',
                   enc_depth_probs: List[float] = (1.0, 0.9, 0.8, 0.7, 0.6, 0.5),
                   dec_depth_probs: List[float] = (1.0, 0.9, 0.8, 0.7, 0.6, 0.5),
                   tied_emb='three-way', exp: Experiment = None):
        """Helper: Construct a model from hyper parameters."""
        assert len(enc_depth_probs) == enc_layers
        assert len(dec_depth_probs) == dec_layers
        # get all args for reconstruction at a later phase
        args = get_my_args(exclusions=['cls', 'exp'])

        assert activation in {'relu', 'elu', 'gelu'}
        log.info(f"Make model, Args={args}")
        c = copy.deepcopy
        attn = tfm.MultiHeadedAttention(n_heads, hid_size, dropout=attn_dropout, bias=attn_bias)
        ff = tfm.PositionwiseFeedForward(hid_size, ff_size, dropout, activation=activation)

        if enc_layers == 0:
            log.warning("Zero encoder layers!")
        encoder = SkipEncoder(tfm.EncoderLayer(hid_size, c(attn), c(ff), dropout), enc_layers,
                              enc_depth_probs)

        assert dec_layers > 0
        decoder = SkipDecoder(tfm.DecoderLayer(hid_size, c(attn), c(attn), c(ff), dropout),
                              dec_layers, dec_depth_probs)

        src_emb = nn.Sequential(tfm.Embeddings(hid_size, src_vocab),
                                tfm.PositionalEncoding(hid_size, dropout))
        tgt_emb = nn.Sequential(tfm.Embeddings(hid_size, tgt_vocab),
                                tfm.PositionalEncoding(hid_size, dropout))
        generator = tfm.Generator(hid_size, tgt_vocab)

        model = cls(encoder, decoder, src_emb, tgt_emb, generator)

        if tied_emb:
            model.tie_embeddings(tied_emb)

        model.init_params()
        return model, args


class SKPTransformerTrainer(tfm.TransformerTrainer):

    def __init__(self, *args, model_factory=SkipTransformerNMT.make_model, **kwargs):
        super().__init__(*args, model_factory=model_factory, **kwargs)
        assert isinstance(self.model, SkipTransformerNMT) or \
            (isinstance(self.model, nn.DataParallel) and isinstance(self.model.module, SkipTransformerNMT))


def __test_model__():
    from rtg.data.dummy import DummyExperiment
    from rtg import Batch, my_tensor as tensor

    vocab_size = 24
    args = {
        'src_vocab': vocab_size,
        'tgt_vocab': vocab_size,
        'enc_layers': 0,
        'dec_layers': 4,
        'hid_size': 32,
        'ff_size': 64,
        'enc_depth_probs': [],
        'dec_depth_probs': [1.0, 0.75, 0.5, 0.75],
        'n_heads': 4,
        'activation': 'relu'
    }

    from rtg.module.decoder import Decoder

    config = {
        'model_type': 'skptfmnmt',
        'trainer': {'init_args': {'chunk_size': 2, 'grad_accum': 2}},
        'optim': {
            'args': {
                # "cross_entropy", "smooth_kld", "binary_cross_entropy", "triplet_loss"
                'criterion': "smooth_kld",
                'lr': 0.01,
                'inv_sqrt': True
            }
        }
    }

    exp = DummyExperiment("work.tmp.skptfmnmt", config=config, read_only=True,
                          vocab_size=vocab_size)
    exp.model_args = args
    trainer = SKPTransformerTrainer(exp=exp, warmup_steps=200, **config['optim']['args'])
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
    steps = 500
    check_point = 25
    trainer.train(steps=steps, check_point=check_point, batch_size=batch_size,
                  check_pt_callback=check_pt_callback)


if __name__ == '__main__':
    __test_model__()
