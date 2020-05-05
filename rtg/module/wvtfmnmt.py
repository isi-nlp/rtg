#!/usr/bin/env python
#
# Author: Lukas J. Ferrer [lferrer (at) isi (dot) edu]
# Created: 2/13/20


import torch.nn as nn
from abc import ABC
from typing import List, Optional

from rtg.module.tfmnmt import (EncoderLayer, DecoderLayer, PositionwiseFeedForward, MultiHeadedAttention,
                               Embeddings, PositionalEncoding, Generator, AbstractTransformerNMT, TransformerTrainer)
from rtg import TranslationExperiment as Experiment, log
from rtg.utils import get_my_args


class WidthVaryingEncoder(nn.Module):
    """Stack of N encoders with heterogeneous feed forward dimensions"""

    def __init__(self, d_model: int, ff_dims: List[int], N: int,
                 n_heads: int, attn_dropout: float, dropout: float, activation: str = 'relu'):
        super().__init__()

        # Make N layers with different pointwise ff_dims
        assert len(ff_dims) == N, f'N:{N} != ff_dims:{len(ff_dims)}'
        layers = list()
        for ff_dim in ff_dims:
            attn = MultiHeadedAttention(n_heads, d_model, attn_dropout)
            ff = PositionwiseFeedForward(d_model, ff_dim, dropout, activation=activation)
            layers.append(EncoderLayer(d_model, attn, ff, dropout))
        self.layers = nn.ModuleList(layers)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, mask):
        """Pass the input (and mask) through each layer in turn."""
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class WidthVaryingDecoder(nn.Module):
    """Stack of N decoders with heterogeneous feed forward dimensions"""

    def __init__(self, d_model: int, ff_dims: List[int], N: int,
                 n_heads: int, attn_dropout: float, dropout: float, activation: str = 'relu'):
        super().__init__()

        # Make N layers with different pointwise ff_dims
        assert len(ff_dims) == N, f'N:{N} != ff_dims:{len(ff_dims)}'
        layers = list()
        for ff_dim in ff_dims:
            self_attn = MultiHeadedAttention(n_heads, d_model, attn_dropout)
            src_attn = MultiHeadedAttention(n_heads, d_model, dropout)
            ff = PositionwiseFeedForward(d_model, ff_dim, dropout, activation=activation)
            layers.append(DecoderLayer(d_model, self_attn, src_attn, ff, dropout))
        self.layers = nn.ModuleList(layers)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


class WidthVaryingTransformerNMT(AbstractTransformerNMT, ABC):
    """Enables heterogeneous feed forward dimensions in the Encoder and Decoder"""

    def __init__(self, encoder: WidthVaryingEncoder, decoder: WidthVaryingDecoder,
                 src_embed, tgt_embed, generator: Optional[Generator], tgt_vocab=None):
        super().__init__(encoder=encoder, decoder=decoder,
                         src_embed=src_embed, tgt_embed=tgt_embed,
                         generator=generator, tgt_vocab=tgt_vocab)

    @property
    def model_type(self):
        return 'wvtfmnmt'

    @classmethod
    def make_model(cls, src_vocab, tgt_vocab, enc_layers=6, dec_layers=6, hid_size=512,
                   n_heads=8, attn_dropout=0.1, dropout=0.2, activation='relu',
                   eff_dims: List[int] = (1024, 1024, 2048, 2048, 1024, 1024),   # Using tuple for immutability
                   dff_dims: List[int] = (1024, 1024, 2048, 2048, 1024, 1024),
                   tied_emb='three-way', exp: Experiment = None):
        """Helper: Construct a model from hyper parameters."""

        assert enc_layers == len(eff_dims)
        assert dec_layers == len(dff_dims)

        # get all args for reconstruction at a later phase
        args = get_my_args(exclusions=['cls', 'exp'])
        assert activation in {'relu', 'elu', 'gelu'}
        log.info(f"Make model, Args={args}")

        if enc_layers == 0:
            log.warning("Zero encoder layers!")
        encoder = WidthVaryingEncoder(
            d_model=hid_size, ff_dims=eff_dims, N=enc_layers, n_heads=n_heads,
            attn_dropout=attn_dropout, dropout=dropout, activation=activation
        )

        assert dec_layers > 0
        decoder = WidthVaryingDecoder(
            d_model=hid_size, ff_dims=dff_dims, N=dec_layers, n_heads=n_heads,
            attn_dropout=attn_dropout, dropout=dropout, activation=activation
        )

        src_emb = nn.Sequential(Embeddings(hid_size, src_vocab),
                                PositionalEncoding(hid_size, dropout))
        tgt_emb = nn.Sequential(Embeddings(hid_size, tgt_vocab),
                                PositionalEncoding(hid_size, dropout))
        generator = Generator(hid_size, tgt_vocab)

        model = cls(encoder, decoder, src_emb, tgt_emb, generator)

        if tied_emb:
            model.tie_embeddings(tied_emb)

        model.init_params()
        return model, args


class WVTransformerTrainer(TransformerTrainer):

    def __init__(self, *args, model_factory=WidthVaryingTransformerNMT.make_model, **kwargs):
        super().__init__(*args, model_factory=model_factory, **kwargs)
        assert isinstance(self.model, WidthVaryingTransformerNMT) or \
            (isinstance(self.model, nn.DataParallel) and isinstance(self.model.module, WidthVaryingTransformerNMT))


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
    trainer = WVTransformerTrainer(exp=exp, warmup_steps=200, **config['optim']['args'])
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
