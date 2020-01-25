#!/usr/bin/env python
#
# Author: Thamme Gowda [tg (at) isi (dot) edu] 
# Created: 2020-01-23

import torch
from torch import nn
import torch.nn.functional as F
from rtg.dataprep import PAD_TOK_IDX
import abc


class Criterion(nn.Module, abc.ABC):
    """Base class for Criterion functions"""

    def __init__(self, input_type: str, pad_idx: int = PAD_TOK_IDX):
        """

        :param input_type: what type of input is expected?
            example: logits, softmax, log_softmax, signmoid
            This choice should be compatible with Generator.forward(x, score=xxx)
        :param pad_idx: index of padding
        """
        super().__init__()
        self.pad_idx = pad_idx
        self.input_type = input_type

class CrossEntropy(Criterion):

    def __init__(self):
        super().__init__(input_type='logits')
        self.xent_loss = nn.CrossEntropyLoss(reduction='none')

    def forward(self, logits, targets, mask_pad=True):
        # logits: [B x V] targets: [B]
        assert targets.shape[0] == logits.shape[0]

        per_tok_loss = self.xent_loss(logits, targets)
        if mask_pad:
            pad_mask = targets == self.pad_idx
            per_tok_loss.masked_fill_(mask=pad_mask, value=0.)

        #num_toks = batch_size - pad_mask.sum()
        #mean_loss = per_tok_loss.sum() / num_toks
        return per_tok_loss


class BinaryCrossEntropy(Criterion):

    def __init__(self):
        super().__init__(input_type='logits')
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, logits, targets, mask_pad=True):
        # logits: [B x V] targets: [B]
        assert targets.shape[0] == logits.shape[0]
        targets = targets.unsqueeze(1)

        truth_full = torch.zeros_like(logits, requires_grad=False)
        truth_full.scatter_(1, targets, 1)

        per_time_per_class_loss = self.bce_loss(logits, truth_full)
        if mask_pad:
            pad_mask = targets == self.pad_idx
            per_time_per_class_loss.masked_fill_(mask=pad_mask, value=0.)

        #num_toks = batch_size - pad_mask.sum()
        #mean_loss = per_tok_loss.sum() / num_toks
        per_tok_loss = per_time_per_class_loss.sum(dim=-1)
        return per_tok_loss


class SmoothKLD(Criterion):
    """
    Label smoothing
    """

    def __init__(self, vocab_size: int, smoothing: float=0.1):
        super().__init__(input_type='log_softmax')
        self.size = vocab_size
        assert 0.0 <= smoothing <= 1.0

        # want elementwise_mean but due to padded tokens, we do the division ourselves
        self.criterion = nn.KLDivLoss(reduction='none')
        self.fill_val = smoothing / (vocab_size - 2)  # exclude 2  = padding, and expected word
        self.confidence = 1.0 - smoothing

    def forward(self, x, target, mask_pad = True):
        # 'x' is log probabilities, originally [B, T, V], but here [B.T, V]
        # 'target' is expected word Ids, originally [B, T] but here [B.T]
        assert x.shape[1] == self.size
        assert x.shape[0] == target.shape[0]
        target = target.unsqueeze(1)  # [B.T x 1]   # and it has class_idx of correct class

        smooth_truth = torch.full_like(x, fill_value=self.fill_val, requires_grad=False)
        smooth_truth[:, self.pad_idx] = 0
        smooth_truth.scatter_(1, target, self.confidence)

        if mask_pad:
            mask = target.eq(self.pad_idx)
            smooth_truth.masked_fill_(mask, 0)
            # note: x is log probs, smooth_truth is just probs
            # D (P || Q) = - \sum p(x) log[q(x)/p(x)]
            # mask is done by setting p(x)=0 for pad toks

        loss = self.criterion(x, smooth_truth)
        return loss


class TripletLoss(Criterion):
    ## Note: Triplet loss doesnt work fully yet; it sorta works and then overfits

    def __init__(self, embedding,  margin=1.0):
        super().__init__(input_type="embedding")
        self.embedding = embedding
        self.vocab_size = embedding.weight.shape[0]
        self.margin = margin

    @classmethod
    def dot(cls, a, b):
        # a, b: [B x D]
        B, D = a.shape[0], a.shape[1]
        # [B x 1 x 1] = [B x 1 x D] * [B x D x 1]
        dots = torch.bmm(a.view(B, 1, D), b.view(B, D, 1))
        return dots.view(B)

    @classmethod
    def distance(cls, a, b):
        # (a - b)^2 = a.a + b.b - 2.a.b
        dist_sq = cls.dot(a, a) + cls.dot(b, b) - 2 * cls.dot(a, b)
        # the root is ignored; do we really need it? I dont think so
        return dist_sq

    def forward(self, x, targets):
        # x: [B x D]   targets:[B]
        anchors = x # [B x D]
        pos_embs = self.embedding(targets)  # [B x D]
        neg_ids = torch.randint_like(targets, low=self.pad_idx + 1, high=self.vocab_size)
        neg_embs = self.embedding(neg_ids)   # [B x D]

        triplet_loss = self.distance(anchors, pos_embs) - self.distance(anchors, neg_embs)
        triplet_loss = F.relu(triplet_loss + self.margin)
        triplet_loss.masked_fill_(targets == self.pad_idx, 0)
        return triplet_loss