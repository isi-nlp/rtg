#!/usr/bin/env python
#
# Author: Thamme Gowda [tg (at) isi (dot) edu] 
# Created: 3/9/19

import abc
import torch
from rtg.binmt.bicycle import BiNMT
from rtg.module.rnnmt import RNNMT
from rtg.lm.rnnlm import RnnLm
from rtg.lm.tfmlm import TfmLm
from rtg.module.tfmnmt import TransformerNMT
from rtg.dataprep import subsequent_mask, PAD_TOK_IDX as PAD_IDX, \
    BOS_TOK_IDX as BOS_IDX, EOS_TOK_IDX as EOS_IDX

INTERACTIVE = False


# TODO: simplify the generators
class GeneratorFactory(abc.ABC):

    def __init__(self, model, **kwargs):
        self.model = model

    @abc.abstractmethod
    def generate_next(self, past_ys):
        pass


class Seq2SeqGenerator(GeneratorFactory):

    def __init__(self, model: RNNMT, x_seqs, x_lens):
        super().__init__(model)
        # [S, B, d], [S, B, d] <-- [S, B], [B]
        self.enc_outs, enc_hids = model.encode(x_seqs, x_lens, None)
        # [S, B, d]
        self.dec_hids = enc_hids

    def generate_next(self, past_ys, get_attn=False):
        last_ys = past_ys[:, -1]
        log_probs, self.dec_hids, attn = self.model.dec(self.enc_outs, last_ys, self.dec_hids)
        return (log_probs, attn) if get_attn else log_probs


class BiNMTGenerator(Seq2SeqGenerator):

    def __init__(self, model: BiNMT, x_seqs, x_lens, path):
        # pick a sub Seq2Seq model inside the BiNMT model as per the given path
        assert path
        super().__init__(model.paths[path], x_seqs, x_lens)
        self.path = path
        self.wrapper = model


class T2TGenerator(GeneratorFactory):

    def __init__(self, model: TransformerNMT, x_seqs, x_lens=None):
        super().__init__(model)
        self.x_mask = (x_seqs != PAD_IDX).unsqueeze(1)
        self.memory = self.model.encode(x_seqs, self.x_mask)

    def generate_next(self, past_ys):
        out = self.model.decode(self.memory, self.x_mask, past_ys, subsequent_mask(past_ys.size(1)))
        log_probs = self.model.generator(out[:, -1])
        return log_probs


class MTfmGenerator(GeneratorFactory):

    def __init__(self, model: TransformerNMT, x_seqs, x_lens=None):
        super().__init__(model)
        x_mask = (x_seqs != PAD_IDX).unsqueeze(1)
        self.sent_repr = self.model.encode(x_seqs, x_mask)

    def generate_next(self, past_ys):
        out = self.model.decode(self.sent_repr, past_ys, subsequent_mask(past_ys.size(1)))
        log_probs = self.model.generator(out[:, -1])
        return log_probs


class ComboGenerator(GeneratorFactory):
    from rtg.syscomb import Combo

    def __init__(self, model: Combo, x_seqs, *args, **kwargs):
        super().__init__(model)
        self.x_mask = (x_seqs != PAD_IDX).unsqueeze(1)
        self.memory = self.model.encode(x_seqs, self.x_mask)

    def generate_next(self, past_ys):
        y_mask = subsequent_mask(past_ys.size(1))
        log_probs = self.model.generate_next(self.memory, self.x_mask, past_ys, y_mask)
        return log_probs


class RnnLmGenerator(GeneratorFactory):

    def __init__(self, model: RnnLm, x_seqs, x_lens):
        super().__init__(model)
        self.dec_hids = None
        if INTERACTIVE:
            # interactive mode use input as prefix
            n = x_seqs.shape[1] - 1 if x_seqs[0, -1] == EOS_IDX[1] else x_seqs.shape[1]
            for i in range(n):
                self.log_probs, self.dec_hids, _ = self.model(None, x_seqs[:, i], self.dec_hids)
            self.consumed = False

    def generate_next(self, past_ys):
        if INTERACTIVE and not self.consumed:
            assert past_ys[0, -1] == BOS_IDX  # we are doing it right?
            self.consumed = True
            return self.log_probs

        last_ys = past_ys[:, -1]
        log_probs, self.dec_hids, _ = self.model(None, last_ys, self.dec_hids)
        return log_probs


class TfmLmGenerator(GeneratorFactory):

    def __init__(self, model: TfmLm, x_seqs, x_lens):
        super().__init__(model)
        if INTERACTIVE:
            self.x_seqs = x_seqs
            self.x_lens = x_lens
            for i in x_lens[1:]:
                # this feature was only meant to be used with a single sequence (probably beamed)
                # all seqs should've same length (else, padding assumption breaks in generate_next)
                assert x_lens[0] == i

    def generate_next(self, past_ys):
        if INTERACTIVE:
            # treat input (i.e. x_seqs) as a prefix for generation
            past_ys = torch.cat([self.x_seqs, past_ys], dim=1)

        seq_mask = subsequent_mask(past_ys.size(1))
        out = self.model(past_ys, seq_mask, gen_probs=False)
        # only generate probs for the last time step
        log_probs = self.model.generator(out[:, -1])
        return log_probs
