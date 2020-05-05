#!/usr/bin/env python
#
# Author: Lukas J. Ferrer [lferrer (at) isi (dot) edu]
# Created: 2/13/20


import torch
import torch.nn as nn
from typing import List, Optional

from rtg.module import wvtfmnmt as wvtfm
from rtg.module import tfmnmt as tfm
from rtg import TranslationExperiment as Experiment, log
from rtg.utils import get_my_args


class WidthVaryingSkipEncoder(wvtfm.WidthVaryingEncoder):
    """Stack of N encoders with heterogeneous feed forward dimensions"""

    def __init__(self, d_model: int, ff_dims: List[int], N: int,
                 n_heads: int, attn_dropout: float, dropout: float,
                 activation: str = 'relu', depth_probs: List[float] = None):
        super().__init__(d_model=d_model, ff_dims=ff_dims,N=N, n_heads=n_heads,
                         attn_dropout=attn_dropout, dropout=dropout, activation=activation)
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


class WidthVaryingSkipDecoder(wvtfm.WidthVaryingDecoder):
    """Stack of N decoders with heterogeneous feed forward dimensions"""

    def __init__(self, d_model: int, ff_dims: List[int], N: int,
                 n_heads: int, attn_dropout: float, dropout: float,
                 activation: str = 'relu', depth_probs: List[float] = None):
        super().__init__(d_model=d_model, ff_dims=ff_dims, N=N, n_heads=n_heads,
                         attn_dropout=attn_dropout, dropout=dropout, activation=activation)

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


class WidthVaryingSkipTransformerNMT(tfm.AbstractTransformerNMT):
    """Enables heterogeneous feed forward dimensions in the Encoder and Decoder"""

    def __init__(self, encoder: WidthVaryingSkipEncoder, decoder: WidthVaryingSkipDecoder,
                 src_embed, tgt_embed,
                 generator: Optional[tfm.Generator], tgt_vocab=None):
        super().__init__(encoder=encoder, decoder=decoder,
                         src_embed=src_embed, tgt_embed=tgt_embed,
                         generator=generator, tgt_vocab=tgt_vocab)

    @property
    def model_type(self):
        return 'wvskptfmnmt'

    @classmethod
    def make_model(cls, src_vocab, tgt_vocab, enc_layers=9, dec_layers=9, hid_size=512,
                   n_heads=8, attn_dropout=0.1, dropout=0.2, activation='relu',
                   eff_dims: List[int] = (2048, 2048, 2048, 1024, 1024, 1024, 2048, 2048, 2048),
                   dff_dims: List[int] = (2048, 2048, 2048, 1024, 1024, 1024, 2048, 2048, 2048),
                   enc_depth_probs: List[float] = (1.0, 0.875, 0.75, 0.625, 0.5, 0.625, 0.75, 0.875, 1.0),
                   dec_depth_probs: List[float] = (1.0, 0.875, 0.75, 0.625, 0.5, 0.625, 0.75, 0.875, 1.0),
                   tied_emb='three-way', exp: Experiment = None):
        """Helper: Construct a model from hyper parameters."""

        assert len(eff_dims) == len(enc_depth_probs) == enc_layers
        assert len(dff_dims) == len(dec_depth_probs) == dec_layers
        # get all args for reconstruction at a later phase
        args = get_my_args(exclusions=['cls', 'exp'])
        assert activation in {'relu', 'elu', 'gelu'}
        log.info(f"Make model, Args={args}")

        if enc_layers == 0:
            log.warning("Zero encoder layers!")
        encoder = WidthVaryingSkipEncoder(
            d_model=hid_size, ff_dims=eff_dims, N=enc_layers, n_heads=n_heads,
            attn_dropout=attn_dropout, dropout=dropout, activation=activation, depth_probs=enc_depth_probs
        )

        assert dec_layers > 0
        decoder = WidthVaryingSkipDecoder(
            d_model=hid_size, ff_dims=dff_dims, N=dec_layers, n_heads=n_heads,
            attn_dropout=attn_dropout, dropout=dropout, activation=activation, depth_probs=dec_depth_probs
        )

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


class WVSKPTransformerTrainer(tfm.TransformerTrainer):

    def __init__(self, *args, model_factory=WidthVaryingSkipTransformerNMT.make_model, **kwargs):
        super().__init__(*args, model_factory=model_factory, **kwargs)
        assert isinstance(self.model, WidthVaryingSkipTransformerNMT) or \
            (isinstance(self.model, nn.DataParallel) and isinstance(self.model.module, WidthVaryingSkipTransformerNMT))


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
        'trainer': {'init_args': {'chunk_size': 2, 'grad_accum': 5}},
        'optim': {
            'args': {
                # "cross_entropy", "smooth_kld", "binary_cross_entropy", "triplet_loss"
                'criterion': "smooth_kld",
                'lr': 0.01,
                'inv_sqrt': True
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
    steps = 200
    check_point = 10
    trainer.train(steps=steps, check_point=check_point, batch_size=batch_size,
                  check_pt_callback=check_pt_callback)


if __name__ == '__main__':
    __test_model__()
