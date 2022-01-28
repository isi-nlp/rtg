#!/usr/bin/env python
#
import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC

from rtg.module.tfmnmt import (Decoder, MultiHeadedAttention, TransformerNMT, TransformerTrainer, Generator)
from rtg.emb.tfmcls import SentenceCompressor
from rtg.distrib import dtorch
from rtg.registry import register, MODEL, registry, CRITERION


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
    # DecoderFactory = SubClassDecoder
    GeneratorFactory = SubClassGenerator
    model_type = MODEL_NAME

    def __init__(self, *args, **kwargs):
        super(SubClassTfmNMT, self).__init__(*args, **kwargs)
        d_model = self.model_dim
        attn = MultiHeadedAttention(h=8, d_model=d_model)
        self.compressor = SentenceCompressor(d_model=d_model, attn=attn)

    def vocab_sub_selection(self, enc_outs, src_mask):
        sent_repr = self.compressor(enc_outs, src_mask)  # [B x D]
        tgt_vocabs = self.generator(sent_repr, score='sigmoid')  # [B x Vocab]
        return tgt_vocabs

    def forward(self, src, tgt, src_mask, tgt_mask, gen_probs=False, log_probs=True, sub_select=False):
        "Take in and process masked src and target sequences."
        enc_outs = self.encode(src, src_mask)                       # [B x T1 x D]
        feats = self.decode(enc_outs, src_mask, tgt, tgt_mask)      # [B x T2 x D]
        if not gen_probs:
            if sub_select:
                tgt_vocabs = self.vocab_sub_selection(enc_outs=enc_outs, src_mask=src_mask)
                return feats, tgt_vocabs
            else:
                return feats
        assert not sub_select, 'sub_select is not implemented yet'
        score = 'log_softmax' if log_probs else 'softmax'
        # tgt_sub_vocab, tgt_rev_idx = tgt.unique(return_inverse=True)
        return self.generator(feats, score=score)

    @classmethod
    def make_trainer(cls, *args, **kwargs):
        return SubClassNMTTrainer(*args, model_factory=cls.make_model, **kwargs)


class SubClassNMTTrainer(TransformerTrainer):

    def __init__(self, *args, **kwargs):
        super(SubClassNMTTrainer, self).__init__(*args, **kwargs)
        self.loss_func.subcls_gen = True
        self.vocab_loss_func = nn.BCELoss(reduce='mean')

    def _train_step(self, take_step: bool, x_mask, x_seqs, y_mask, y_seqs_in, y_seqs_out):
        # [Batch x Time x D], [Batch x V]
        out, vocab_guesses = self.model(x_seqs, y_seqs_in, x_mask, y_mask, sub_select=True)

        # multi-label classification on vocabulary
        vocab_truth = torch.zeros_like(vocab_guesses, dtype=torch.float)   # [B x V]
        vocab_truth.scatter_(1, y_seqs_out, 1.)                           # [B x V]
        loss2 = self.vocab_loss_func(vocab_guesses, vocab_truth)
        dtorch.backward(loss2, retain_graph=True)
        loss2 = loss2.item()

        # skip the last time step (the one with EOS as input)
        out = out[:, :-1, :]
        # assumption:  y_seqs has EOS, and not BOS
        loss1 = self.loss_func(out, y_seqs_out, train_mode=True, take_step=take_step)

        return loss1 + loss2
