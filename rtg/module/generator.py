#!/usr/bin/env python
#
# Author: Thamme Gowda [tg (at) isi (dot) edu] 
# Created: 3/9/19

import abc
import torch
from rtg import log
from rtg.module.rnnmt import RNNMT
from rtg.lm.rnnlm import RnnLm
from rtg.lm.tfmlm import TfmLm
from rtg.module.tfmnmt import TransformerNMT
from rtg.data.dataset import subsequent_mask
from rtg.data.codec import Field

INTERACTIVE = False


# TODO: simplify the generators
class GeneratorFactory(abc.ABC):

    def __init__(self, model, field: Field, **kwargs):
        self.model = model
        self.field = field

    @abc.abstractmethod
    def generate_next(self, past_ys):
        pass


class Seq2SeqGenerator(GeneratorFactory):

    def __init__(self, model: RNNMT, field, x_seqs, x_lens):
        super().__init__(model, field=field)
        # [S, B, d], [S, B, d] <-- [S, B], [B]
        self.enc_outs, enc_hids = model.encode(x_seqs, x_lens, None)
        # [S, B, d]
        self.dec_hids = enc_hids

    def generate_next(self, past_ys, get_attn=False):
        last_ys = past_ys[:, -1]
        log_probs, self.dec_hids, attn = self.model.dec(self.enc_outs, last_ys, self.dec_hids)
        return (log_probs, attn) if get_attn else log_probs


class T2TGenerator(GeneratorFactory):

    multi_label_warned = False

    def __init__(self, model: TransformerNMT, field, x_seqs, x_lens=None, multi_label=False):
        super().__init__(model, field)
        self.x_mask = (x_seqs != field.pad_idx).unsqueeze(1)
        self.memory = self.model.encode(x_seqs, self.x_mask)
        self.multi_label = multi_label
        if multi_label and not type(self).multi_label_warned:
            log.warning(">>> Multi-label decoding mode enabled")
            type(self).multi_label_warned = True

    def generate_next(self, past_ys):
        out = self.model.decode(self.memory, self.x_mask, past_ys, subsequent_mask(past_ys.size(1)))
        if self.multi_label:
            log_probs = self.model.generator(out[:, -1], score='sigmoid').log()
        else:
            log_probs = self.model.generator(out[:, -1], score='log_softmax')
        return log_probs


class MTfmGenerator(GeneratorFactory):

    def __init__(self, model: TransformerNMT, field, x_seqs, x_lens=None):
        super().__init__(model, field)
        x_mask = (x_seqs != field.pad_idx).unsqueeze(1)
        self.sent_repr = self.model.encode(x_seqs, x_mask)

    def generate_next(self, past_ys):
        out = self.model.decode(self.sent_repr, past_ys, subsequent_mask(past_ys.size(1)))
        log_probs = self.model.generator(out[:, -1])
        return log_probs

class TfmExtEembGenerator(T2TGenerator):

    def __init__(self, model: TransformerNMT, x_seqs, x_lens=None):
        super().__init__(model, x_seqs, x_seqs)
        self.src_ext_emb = self.model.src_ext_emb(x_seqs)

    def generate_next(self, past_ys):
        tgt_ext_emb = self.model.tgt_ext_emb(past_ys)
        y_mask = subsequent_mask(past_ys.size(1))
        out = self.model.decode(self.memory, self.x_mask, past_ys, y_mask,
                                self.src_ext_emb, tgt_ext_emb)
        log_probs = self.model.generator(out[:, -1])
        return log_probs

class ComboGenerator(GeneratorFactory):
    from rtg.syscomb import Combo

    def __init__(self, model: Combo, field, x_seqs, *args, **kwargs):
        super().__init__(model, field)
        self.x_mask = (x_seqs != field.pad_idx).unsqueeze(1)
        self.memory = self.model.encode(x_seqs, self.x_mask)

    def generate_next(self, past_ys):
        y_mask = subsequent_mask(past_ys.size(1))
        log_probs = self.model.generate_next(self.memory, self.x_mask, past_ys, y_mask)
        return log_probs


class RnnLmGenerator(GeneratorFactory):

    def __init__(self, model: RnnLm, field, x_seqs, x_lens):
        super().__init__(model, field)
        self.dec_hids = None
        if INTERACTIVE:
            # interactive mode use input as prefix
            n = x_seqs.shape[1] - 1 if x_seqs[0, -1] == field.eos_idx else x_seqs.shape[1]
            for i in range(n):
                self.log_probs, self.dec_hids, _ = self.model(None, x_seqs[:, i], self.dec_hids)
            self.consumed = False

    def generate_next(self, past_ys):
        if INTERACTIVE and not self.consumed:
            assert past_ys[0, -1] == self.field.bos_idx  # we are doing it right?
            self.consumed = True
            return self.log_probs

        last_ys = past_ys[:, -1]
        log_probs, self.dec_hids, _ = self.model(None, last_ys, self.dec_hids)
        return log_probs


class TfmLmGenerator(GeneratorFactory):

    def __init__(self, model: TfmLm, field, x_seqs, x_lens):
        super().__init__(model, field)
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
