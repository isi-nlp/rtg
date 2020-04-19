#!/usr/bin/env python
#
# Author: Thamme Gowda [tg (at) isi (dot) edu] 
# Created: 5/15/19

"""
An Hybrid model where th encoder is Transformer and decoder is RNN
"""

import inspect
import copy
import torch
import torch.nn as nn

from rtg.module.tfmnmt import (Encoder, EncoderLayer, PositionwiseFeedForward, PositionalEncoding,
                               Generator, MultiHeadedAttention, Embeddings, TransformerNMT,
                               TransformerTrainer)
from rtg import TranslationExperiment as Experiment, log


class RnnDecoder(nn.Module):
    """RNN Decoder - Teacher Forcing Training
    """

    def __init__(self, hid_size: int, n_layers: int, rnn_type: str = 'LSTM', dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.n_layers = n_layers
        self.hid_size = hid_size
        rnn_type = rnn_type.upper()
        self.rnn_type = rnn_type
        rnn_factory = {'LSTM': nn.LSTM, 'GRU': nn.GRU}[rnn_type]
        self.rnn = rnn_factory(self.hid_size, self.hid_size, num_layers=self.n_layers,
                               bidirectional=False, batch_first=True,
                               dropout=dropout if n_layers > 1 else 0)

    def forward(self, hid_state, tgt):
        # tgt: [B x T x H],
        # hid_state: for lstm ([Layers*Directions, B x H ], --),  for GRU: [Layers*Directions, B x H ]
        tgt = self.dropout(tgt)
        if self.rnn_type == 'LSTM':
            hid_state= self.dropout(hid_state[0]), self.dropout(hid_state[1])
        else:
            hid_state = self.dropout(hid_state)
        output, _ = self.rnn(tgt, hid_state)
        # output: [B x T x H]
        return self.dropout(output)


class HybridMT(TransformerNMT):

    @property
    def model_type(self):
        return 'hybridmt'

    def tie_embeddings(self, tie: str):
        assert tie in ('one-way', 'two-way', 'three-way')
        log.info(f"Tying embeddings: {tie}")
        if tie in ('one-way', 'three-way'):
            log.info(f"Tying embeddings: TgtOut == TgtInp")
            self.generator.proj.weight = self.tgt_embed.lut.weight
        if tie in ('two-way', 'three-way'):
            log.info(f"Tying embeddings: SrcInp == TgtInp")
            self.src_embed[0].lut.weight = self.tgt_embed.lut.weight

    @classmethod
    def make_model(cls, src_vocab, tgt_vocab, enc_layers=6, hid_size=512, ff_size=2048, enc_heads=8,
                   dropout=0.1, tied_emb='three-way', dec_rnn_type: str = 'LSTM',
                   dec_layers: int = 1,
                   exp: Experiment = None):
        """
        Helper: Construct a model from hyper parameters."
        :return: model, args
        """
        # get all args for reconstruction at a later phase
        _, _, _, args = inspect.getargvalues(inspect.currentframe())
        for exclusion in ['cls', 'exp']:
            del args[exclusion]  # exclude some args

        log.info(f"making hybridmt model: {args}")

        c = copy.deepcopy
        attn = MultiHeadedAttention(enc_heads, hid_size)
        ff = PositionwiseFeedForward(hid_size, ff_size, dropout)

        enc_layer = EncoderLayer(hid_size, c(attn), c(ff), dropout)
        encoder = Encoder(enc_layer, enc_layers)  # clones n times
        src_emb = nn.Sequential(Embeddings(hid_size, src_vocab),
                                PositionalEncoding(hid_size, dropout))

        decoder = RnnDecoder(rnn_type=dec_rnn_type, hid_size=hid_size, n_layers=dec_layers,
                             dropout=dropout)
        tgt_emb = Embeddings(hid_size, tgt_vocab)
        generator = Generator(hid_size, tgt_vocab)

        model = cls(encoder, decoder, src_emb, tgt_emb, generator)
        if tied_emb:
            model.tie_embeddings(tied_emb)

        model.init_params()
        return model, args

    def forward(self, src, tgt, src_mask, tgt_mask, gen_probs=False, log_probs=True):
        "Take in and process masked src and target sequences."
        assert src.shape[0] == tgt.shape[0]
        sent_repr = self.encode(src, src_mask)
        feats = self.decode(sent_repr, tgt, tgt_mask)
        return self.generator(feats, log_probs=log_probs) if gen_probs else feats

    def encode(self, src, src_mask):
        batch_size = src.shape[0]
        # ADD CLS token
        cls_col = torch.full((batch_size, 1), fill_value=cls_idx, device=src.device,
                             dtype=torch.long)
        src = torch.cat([cls_col, src], dim=1)
        # assuming first col of mask is proper
        src_mask = torch.cat([src_mask[:, :, :1], src_mask], dim=-1)

        embs = self.src_embed(src)
        enc_feats = self.encoder(embs, src_mask)
        sent_repr = enc_feats[:, 0, :]  # CLS token features
        return sent_repr

    def decode(self, sent_repr, tgt, tgt_mask, **extra):
        assert not extra  # no extra stuff;  added to keep the lint happy
        # sentence representation goes to decoder layer
        embs = self.tgt_embed(tgt)
        # sent-repr

        dec_layers, rnn_type = self.decoder.n_layers, self.decoder.rnn_type
        sent_repr = sent_repr.unsqueeze(0).repeat(dec_layers, 1, 1)
        hidden_state = (sent_repr, sent_repr) if rnn_type == 'LSTM' else sent_repr
        return self.decoder(hidden_state, embs)

    @classmethod
    def make_trainer(cls, *args, **kwargs):
        return HybridMTTrainer(*args, **kwargs)

class HybridMTTrainer(TransformerTrainer):

    def __init__(self, *args, model_factory=HybridMT.make_model, **kwargs):
        super().__init__(*args, model_factory=model_factory, **kwargs)
        assert isinstance(self.model, HybridMT)  # type check


def __test_model__():
    from rtg.data.dummy import DummyExperiment
    from rtg import Batch, my_tensor as tensor

    vocab_size = 24
    args = {
        'src_vocab': vocab_size,
        'tgt_vocab': vocab_size,
        'enc_layers': 4,
        'dec_layers': 3,
        'hid_size': 128,
        'ff_size': 256,
        'dec_rnn_type': 'GRU',
        'enc_heads': 4
    }

    from rtg.module.decoder import Decoder

    exp = DummyExperiment("work.tmp.hybridmt", config={'model_type': 'hybridmt'}, read_only=True,
                          vocab_size=vocab_size)
    exp.model_args = args
    trainer = HybridMTTrainer(exp=exp, warmup_steps=200)
    decr = Decoder.new(exp, trainer.model)

    assert 2 == Batch.bos_val
    src = tensor([[4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
                  [13, 12, 11, 10, 9, 8, 7, 6, 5, 4]])
    src_lens = tensor([src.size(1)] * src.size(0))

    def check_pt_callback(**args):
        res = decr.greedy_decode(src, src_lens, max_len=12)
        for score, seq in res:
            log.info(f'{score:.4f} :: {seq}')

    batch_size = 50
    steps = 2000
    check_point = 50
    trainer.train(steps=steps, check_point=check_point, batch_size=batch_size,
                  check_pt_callback=check_pt_callback)


if __name__ == '__main__':
    __test_model__()
