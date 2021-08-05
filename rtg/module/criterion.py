#!/usr/bin/env python
#
# Author: Thamme Gowda [tg (at) isi (dot) edu] 
# Created: 2020-01-23

import torch
from torch import nn
import torch.nn.functional as F
import abc
#from rtg.registry import CRITERION, register


class Criterion(nn.Module, abc.ABC):
    """Base class for Criterion functions"""

    def __init__(self, input_type: str, pad_idx: int):
        """
        :param input_type: what type of input is expected?
            example: logits, softmax, log_softmax, signmoid
            This choice should be compatible with Generator.forward(x, score=xxx)
        :param pad_idx: index of padding
        """
        super().__init__()
        self.pad_idx = pad_idx
        self.input_type = input_type


#@register(kind=CRITERION, name="cross_entropy")
class CrossEntropy(Criterion):

    def __init__(self, pad_idx):
        super().__init__(input_type='logits', pad_idx=pad_idx)
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


#@register(kind=CRITERION, name="binary_cross_entropy")
class BinaryCrossEntropy(Criterion):

    def __init__(self, pad_idx, smoothing=0.1):
        assert 0 <= smoothing < 1
        super().__init__(input_type='logits', pad_idx=pad_idx)
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='none')
        self.smoothing = smoothing

    def forward(self, logits, targets, mask_pad=True):
        # logits: [B x V] targets: [B]
        assert targets.shape[0] == logits.shape[0]
        targets = targets.unsqueeze(1)

        truth_full = torch.full_like(logits, fill_value=self.smoothing, requires_grad=False)
        #truth_full = torch.zeros_like(logits, requires_grad=False)
        truth_full.scatter_(1, targets, 1)

        per_time_per_class_loss = self.bce_loss(logits, truth_full)
        if mask_pad:
            pad_mask = targets == self.pad_idx
            per_time_per_class_loss.masked_fill_(mask=pad_mask, value=0.)

        #num_toks = batch_size - pad_mask.sum()
        #mean_loss = per_tok_loss.sum() / num_toks
        per_tok_loss = per_time_per_class_loss.sum(dim=-1)
        return per_tok_loss


#@register(kind=CRITERION, name="smooth_kld")
class SmoothKLD(Criterion):
    """
    Label smoothing
    """

    def __init__(self, vocab_size: int, pad_idx: int, smoothing: float = 0.1):
        super().__init__(input_type='log_softmax', pad_idx=pad_idx)
        self.size = vocab_size
        assert 0.0 <= smoothing <= 1.0

        # want elementwise_mean but due to padded tokens, we do the division ourselves
        self.criterion = nn.KLDivLoss(reduction='none')
        self.fill_val = smoothing / (vocab_size - 2)  # exclude 2  = padding, and expected word
        self.confidence = 1.0 - smoothing

    def forward(self, x, target, mask_pad=True):
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


#@register(kind=CRITERION, name="triplet_loss")
class TripletLoss(Criterion):
    # Note: Triplet loss doesnt work fully yet; it sorta works and then overfits

    def __init__(self, embedding, pad_idx,  margin: float = 0., neg_region: float = 0.05,
                 mode: str = 'dot', neg_sampling: str = 'random'):
        # TODO: whats the right margin?
        super().__init__(input_type='embedding', pad_idx=pad_idx)
        self.embedding = embedding
        self.embeddings = embedding.weight
        self.vocab_size = embedding.weight.shape[0]
        assert margin >= 0
        self.margin = margin
        assert mode in ('dot', 'l2')
        self.mode = mode
        assert neg_sampling in ('random', 'hard')
        self.neg_sampling = neg_sampling
        self.hard_neg_region = max(int(neg_region * self.vocab_size), 5)

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
        anchors = x      # [B x D]
        pos_embs = self.embedding(targets)  # [B x D]
        if self.neg_sampling == 'random':
            neg_ids = torch.randint_like(targets, low=self.pad_idx+1, high=self.vocab_size)  # [B]
        elif self.neg_sampling == 'hard':
            candidates = torch.einsum('bd,vd->bv', anchors, self.embeddings)    # [B, V]
            mask = candidates.new_full(candidates.shape, False, dtype=torch.bool) # [B, V]
            mask.scatter_(dim=1, index=targets.view(-1, 1), value=True)
            candidates = candidates.masked_fill(mask, -1)
            c, indexes = torch.sort(candidates)
            neg_ids = indexes[:, -self.hard_neg_region].contiguous().view(-1)  #[B]
        else:
            raise Exception(self.neg_sampling + ' not supported')
        neg_embs = self.embedding(neg_ids)   # [B x D]

        if self.mode == 'l2':
            triplet_loss = self.distance(anchors, pos_embs) - self.distance(anchors, neg_embs) + self.margin
            triplet_loss = F.relu(triplet_loss)
        elif self.mode == 'dot':
            triplet_loss = self.dot(anchors, pos_embs) - self.dot(anchors, neg_embs) + self.margin
            triplet_loss = F.relu(-triplet_loss)
        else:
            raise Exception(self.mode + ' not supported')

        triplet_loss = triplet_loss.masked_fill(targets == self.pad_idx, 0)
        return triplet_loss


#@register(kind=CRITERION, name="smooth_kld_and_triplet_loss")
class SmoothKLDAndTripletLoss(Criterion):

    def __init__(self, embedding, pad_idx, margin: float = 0., neg_region: float = 0.05,
                 mode: str = 'dot', neg_sampling: str = 'random',
                 smoothing: float = 0.1, alpha: float = 1.0):
        super().__init__(input_type='identity')
        self.embeddings = embedding.weight
        self.smoothKLD = SmoothKLD(embedding.weight.shape[0], smoothing=smoothing, pad_idx=pad_idx)
        self.tripletLoss = TripletLoss(embedding, margin=margin, neg_region=neg_region, mode=mode,
                                       neg_sampling=neg_sampling, pad_idx=pad_idx)
        self.alpha = alpha

    def forward(self, x, targets, mask_pad=True):
        smx = F.log_softmax(torch.einsum('bd,vd->bv', x, self.embeddings), dim=-1)
        sKLD = self.smoothKLD(smx, targets, mask_pad)
        tLoss = self.tripletLoss(x, targets)

        # Must sum here to match sizes
        return sKLD.sum() + self.alpha * tLoss.sum()
