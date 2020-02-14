#!/usr/bin/env python
#
# Author: Lukas J. Ferrer [lferrer (at) isi (dot) edu]
# Created: 2/13/20


import inspect
import torch
import torch.nn as nn
from abc import ABC
from typing import *

from rtg.module.tfmnmt import (EncoderLayer, DecoderLayer, PositionwiseFeedForward, MultiHeadedAttention,
                               Embeddings, PositionalEncoding, Generator, AbstractTransformerNMT, TransformerTrainer)
from rtg import TranslationExperiment as Experiment, log


class WidthVaryingSkipEncoder(nn.Module):
    """Stack of N encoders with heterogeneous feed forward dimensions"""

    def __init__(self, d_model: int, ff_dims: List[int], N: int,
                 n_heads: int, dropout: float, activation: str = 'relu',
                 depth_probs: List[float] = None):
        super().__init__()

        # Make N layers with different pointwise ff_dims
        assert len(ff_dims) >= N, 'Not enough ff_dims to complete the model'
        layers = list()
        for n in range(N):
            attn = MultiHeadedAttention(n_heads, d_model, dropout)
            ff = PositionwiseFeedForward(d_model, ff_dims[n], dropout, activation=activation)
            layers.append(EncoderLayer(d_model, attn, ff, dropout))
        self.layers = nn.ModuleList(layers)
        self.norm = nn.LayerNorm(d_model)

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


class WidthVaryingSkipDecoder(nn.Module):
    """Stack of N decoders with heterogeneous feed forward dimensions"""

    def __init__(self, d_model: int, ff_dims: List[int], N: int,
                 n_heads: int, dropout: float, activation: str = 'relu',
                 depth_probs: List[float] = None):
        super().__init__()

        # Make N layers with different pointwise ff_dims
        assert len(ff_dims) >= N, 'Not enough ff_dims to complete the model'
        layers = list()
        for n in range(N):
            self_attn = MultiHeadedAttention(n_heads, d_model, dropout)
            src_attn = MultiHeadedAttention(n_heads, d_model, dropout)
            ff = PositionwiseFeedForward(d_model, ff_dims[n], dropout, activation=activation)
            layers.append(DecoderLayer(d_model, self_attn, src_attn, ff, dropout))
        self.layers = nn.ModuleList(layers)
        self.norm = nn.LayerNorm(d_model)

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


class WidthVaryingSkipTransformerNMT(AbstractTransformerNMT, ABC):
    """Enables heterogeneous feed forward dimensions in the Encoder and Decoder"""

    def __init__(self, encoder: WidthVaryingSkipEncoder, decoder: WidthVaryingSkipDecoder,
                 src_embed, tgt_embed,
                 generator: Optional[Generator], tgt_vocab=None):
        super().__init__(encoder=encoder, decoder=decoder,
                         src_embed=src_embed, tgt_embed=tgt_embed,
                         generator=generator, tgt_vocab=tgt_vocab)

    @property
    def model_type(self):
        return 'wvskptfmnmt'

    @classmethod
    def make_model(cls, src_vocab, tgt_vocab, enc_layers=9, dec_layers=9, hid_size=512,
                   eff_dims: List[int] = (2048, 2048, 2048, 1024, 1024, 1024, 2048, 2048, 2048),
                   dff_dims: List[int] = (2048, 2048, 2048, 1024, 1024, 1024, 2048, 2048, 2048),
                   enc_depth_probs: List[float] = (1.0, 0.875, 0.75, 0.625, 0.5, 0.625, 0.75, 0.875, 1.0),
                   dec_depth_probs: List[float] = (1.0, 0.875, 0.75, 0.625, 0.5, 0.625, 0.75, 0.875, 1.0),
                   n_heads=8, dropout=0.1, tied_emb='three-way', activation='relu',
                   exp: Experiment = None):
        """Helper: Construct a model from hyper parameters."""

        # get all args for reconstruction at a later phase
        _, _, _, args = inspect.getargvalues(inspect.currentframe())
        for exclusion in ['cls', 'exp']:
            del args[exclusion]  # exclude some args
        # In case you are wondering, why I didnt use **kwargs here:
        #   these args are read from conf file where user can introduce errors, so the parameter
        #   validation and default value assignment is implicitly done by function call for us :)
        assert activation in {'relu', 'elu', 'gelu'}
        log.info(f"Make model, Args={args}")

        if enc_layers == 0:
            log.warning("Zero encoder layers!")
        encoder = WidthVaryingSkipEncoder(d_model=hid_size, ff_dims=eff_dims, N=enc_layers,
                                          n_heads=n_heads, dropout=dropout, activation=activation,
                                          depth_probs=enc_depth_probs)

        assert dec_layers > 0
        decoder = WidthVaryingSkipDecoder(d_model=hid_size, ff_dims=dff_dims, N=dec_layers,
                                          n_heads=n_heads, dropout=dropout, activation=activation,
                                          depth_probs=dec_depth_probs)

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


class WVSKPTransformerTrainer(TransformerTrainer):

    def __init__(self, *args, model_factory=WidthVaryingSkipTransformerNMT.make_model, **kwargs):
        super().__init__(*args, model_factory=model_factory, **kwargs)
        assert isinstance(self.model, WidthVaryingSkipTransformerNMT)  # type check


def __test_model__():
    from rtg.dummy import DummyExperiment
    from rtg import Batch, my_tensor as tensor

    vocab_size = 24
    args = {
        'src_vocab': vocab_size,
        'tgt_vocab': vocab_size,
        'enc_layers': 0,
        'dec_layers': 4,
        'hid_size': 32,
        'eff_dims': [],
        'dff_dims': [64, 128, 128, 64],
        'enc_depth_probs': [],
        'dec_depth_probs': [1.0, 0.75, 0.5, 0.75],
        'n_heads': 4,
        'activation': 'relu'
    }

    from rtg.module.decoder import Decoder

    config = {
        'model_type': 'wvskptfmnmt',
        'trainer': {'init_args': {'chunk_size': 2}},
        'optim': {
            'args': {
                # "cross_entropy", "smooth_kld", "binary_cross_entropy", "triplet_loss"
                'criterion': "smooth_kld"
            }
        }
    }

    exp = DummyExperiment("work.tmp.wvskptfmnmt", config=config, read_only=True,
                          vocab_size=vocab_size)
    exp.model_args = args
    trainer = WVSKPTransformerTrainer(exp=exp, warmup_steps=200, **config['optim']['args'])
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
