#!/usr/bin/env python
#
# Author: Thamme Gowda [tg (at) isi (dot) edu] 
# Created: 2020-01-23

import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
import abc
from rtg.registry import CRITERION, register


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


def smooth_labels(labels, n_labels, smooth_rate, weight=None):
    """
    :param labels: labels [N] where each item is in range {0, 1, 2,... C-1}
    :param n_labels: total number of classes
    :param smooth_rate: the magnitude of smoothing
    :param weight: distribute epsilon as per the weights
    :return:
    """
    assert len(labels.shape) == 1
    assert labels.max() < n_labels
    assert 0 <= labels.min()
    assert 0 <= smooth_rate <= 1

    N = len(labels)
    labels = labels.view(N, 1)
    device = labels.device
    if weight is None:
        # take out epsilon and distribute evenly to all but one
        fill_value = smooth_rate / (n_labels - 1)
        # expand [N] -> [N, C]
        full = torch.full([N, n_labels], fill_value=fill_value, dtype=torch.float, device=device)
        full.scatter_(1, labels.type(torch.int64), 1 - smooth_rate)
    else:
        assert len(weight) == n_labels
        weight = weight.to(device)
        full = (weight * smooth_rate).expand(N, n_labels)  # [C] -> [N, C]
        peaks = torch.full([N, 1], fill_value=1 - smooth_rate, dtype=torch.float, device=device)
        #peaks = torch.tensor(1 - epsilon, device=device).expand(N, 1)  # [N, 1]
        full.scatter_add_(1, labels, peaks)  # inplace add
    return full


def dense_cross_entropy(input: Tensor, target: Tensor, reduction=None, mask_out=None, weight=None,
                        input_type='logits') -> Tensor:
    """
    :param input: input tensor, see input_type
    :param target: probability distribution of labels
    :param reduction: what reduction to perform
    :param mask_out: optional boolean tensor same shape as input and target.
        True positions results in exclusion of item-class loss from total loss
    :param weight: optional float tensor, whose items are multiplied with item-class before
        calling reduce operations on result. shape of weight should be broadcastable to input.
        See https://pytorch.org/docs/stable/notes/broadcasting.html
        e.g. input is [N, C] implies batch has N items and C classes
          weight of [N, C] => weight for each item-class pair
          weight of [N, 1] => weight for each item, all classes have same weight
          weight of [1, C] => weight for each class, all items have same weight
    :param: input_type: to specify what kind of values are in tensor.
             Valid options are: logits, probs, log_probs
    :return:
    """
    assert reduction in ('per_item', 'per_class', 'none', None, 'sum', 'micro', 'macro')
    N, C = input.shape
    #target is dense; i.e not one-hot
    assert input.shape == target.shape,  f'input shape: {input.shape}, target is {target.shape}'
    if mask_out is not None:
        assert mask_out.shape in [(N, C), (1, C), (N, 1)],\
            f'input: {input.shape} and mask_out: {mask_out.shape} are not broadcastable'
    if weight is not None:
        assert weight.shape in [(N, C), (1, C), (N, 1)],\
            f'input: {input.shape} weight: {weight.shape} are not broadcastable'

    if input_type == 'log_probs':
        log_probs = input
    elif input_type == 'probs':
        log_probs = input.log()   # TODO: handle log zero
    elif input_type == 'logits':
        log_probs = input.log_softmax(dim=1)
    else:
        raise Exception(f'input_type={input_type} unknown; know: logits, probs, log_probs')

    tot_items, tot_classes = N, C
    #[N, C] :  -y_c * log(p_c)
    table = -torch.mul(target, log_probs)  # loss per item-class
    if mask_out is not None:   # overwrite positions with mask=True to zero
        table.masked_fill_(mask_out, value=0.0)
        if mask_out.shape == (N, 1): # [N, 1] => sum items are excluded e.g. pad tokens
            tot_items -= mask_out.sum()
    if weight is not None:
        table.mul_(weight) # assumption: weight is broadcastable

    if reduction in (None, 'none'):
        return table
    elif reduction == 'per_item':
        return table.sum(dim=1)
    elif reduction == 'per_class':
        return table.sum(dim=0)
    elif reduction == 'sum':
        return table.sum()
    elif reduction == 'micro':
        # micro: first get loss per_item (get rid of class dim), and average over items
        return table.sum(dim=1).sum() / tot_items
    elif reduction == 'macro':
        # micro: first get loss per_class, normalize as per their frequencies and then sum
        eps = 1e-9   # to avoid divide by zero
        target_per_class = target.sum(dim=0) + eps
        input_per_class = table.sum(dim=0)
        per_class_normalized = input_per_class / target_per_class
        return per_class_normalized.sum()
    else:
        raise ValueError(f'reduce={reduction} not supported')


@register(kind=CRITERION, name="cross_entropy")
class CrossEntropy(Criterion):

    def __init__(self, pad_idx: int, label_smoothing=0., reduction='micro'):
        super().__init__(input_type='logits', pad_idx=pad_idx)
        assert 0 <= label_smoothing <= 1
        self.label_smoothing = label_smoothing
        assert reduction in ('micro', 'macro')
        self.reduction = reduction
        if reduction == 'macro':
            assert self.label_smoothing > 0., 'reduce=macro requires label_smoothing > 0'

    def forward(self, inputs, targets, mask_pad=True):
        # logits: [N x C] targets: [N]
        N, C = inputs.shape
        assert targets.shape[0] == inputs.shape[0]
        mask_out = None
        if mask_pad:
            mask_out = (targets == self.pad_idx).unsqueeze(1) # [N] -> [N, 1]
        if self.label_smoothing > 0:
            dense_targets = smooth_labels(labels=targets, n_labels=C, smooth_rate=self.label_smoothing)
        else:
            dense_targets = torch.full([N, C], fill_value=0.0, dtype=torch.float,
                                       device=inputs.device)
            dense_targets.scatter_(1, targets.type(torch.int64), 1.0)

        weight = self.get_weights(inputs, targets)
        loss = dense_cross_entropy(input=inputs, target=dense_targets, reduction=self.reduction,
                                   weight=weight, mask_out=mask_out, input_type=self.input_type)
        return loss

    def get_weights(self, inputs, targets):
        return None  # nothing interesting here yet


class MaskableSmoothableCriterion(Criterion):

    def __init__(self, pad_idx, input_type='logits', label_smoothing=0.0, reduction='micro'):
        super(MaskableSmoothableCriterion, self).__init__(pad_idx=pad_idx, input_type=input_type)
        self.label_smoothing = label_smoothing
        self.reduction = reduction

    def forward(self, inputs, targets, mask_pad=True, reduce=None):
        N, C = inputs.shape
        reduce = reduce or self.reduction
        assert targets.shape[0] == inputs.shape[0]
        mask_out = None
        if mask_pad:
            mask_out = (targets == self.pad_idx).unsqueeze(1)  # [N] -> [N, 1]
        if self.label_smoothing > 0.0:
            dense_targets = smooth_labels(labels=targets, n_labels=C, smooth_rate=self.label_smoothing)
        else:
            dense_targets = torch.full([N, C], fill_value=0.0, dtype=torch.float,
                                       device=inputs.device)
            dense_targets.scatter_(1, targets.type(torch.int64), 1.0)
        loss = dense_cross_entropy(input=inputs, target=targets, reduction=reduce, mask_out=mask_out,
                                   input_type=self.input_type)
        return loss


@register(kind=CRITERION, name="focal_loss")
class FocalLoss(Criterion):

    def __init__(self, pad_idx: int, label_smoothing=0., reduction='micro', gamma=0.0):
        super(FocalLoss, self).__init__(pad_idx=pad_idx, input_type='probs')
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.reduction = reduction
        assert gamma >= 0.0

    def forward(self, inputs, targets, mask_pad=True):
        probs = inputs
        N, C = probs.shape
        assert targets.shape[0] == probs.shape[0]
        mask_out = None
        if mask_pad:
            mask_out = (targets == self.pad_idx).unsqueeze(1)  # [N] -> [N, 1]
        if self.label_smoothing > 0:
            dense_targets = smooth_labels(labels=targets, n_labels=C, smooth_rate=self.label_smoothing)
        else:
            dense_targets = torch.full([N, C], fill_value=0.0, dtype=torch.float,
                                       device=probs.device)
            dense_targets.scatter_(1, targets.type(torch.int64), 1.0)
        weight = None
        if self.gamma > 0.0:
            probs = inputs.softmax(dim=1)
            weight = (1 - probs).pow(self.gamma)

        loss = dense_cross_entropy(input=probs, target=dense_targets, reduction=self.reduction,
                                   weight=weight, mask_out=mask_out, input_type=self.input_type)
        return loss


@register(kind=CRITERION, name="binary_cross_entropy")
class BinaryCrossEntropy(Criterion):

    def __init__(self, pad_idx: int, label_smoothing=0.1):
        assert 0 <= label_smoothing < 1
        super().__init__(input_type='logits', pad_idx=pad_idx)
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='none')
        self.smoothing = label_smoothing

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


@register(kind=CRITERION, name="smooth_kld")
class SmoothKLD(Criterion):
    """
    Label smoothing
    """

    def __init__(self, pad_idx: int, n_classes: int, label_smoothing: float = 0.1):
        super().__init__(input_type='log_softmax', pad_idx=pad_idx)
        self.size = n_classes
        assert 0.0 <= label_smoothing <= 1.0

        # want elementwise_mean but due to padded tokens, we do the division ourselves
        self.criterion = nn.KLDivLoss(reduction='none')
        self.fill_val = label_smoothing / (n_classes - 2)  # exclude 2  = padding, and expected word
        self.confidence = 1.0 - label_smoothing

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


@register(kind=CRITERION, name="triplet_loss")
class TripletLoss(Criterion):
    # Note: Triplet loss doesnt work fully yet; it sorta works and then overfits

    def __init__(self, pad_idx: int, embedding: nn.Embedding, margin: float = 0.,
                 neg_region: float = 0.05, mode: str = 'dot', neg_sampling: str = 'random'):
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


@register(kind=CRITERION, name="smooth_kld_and_triplet_loss")
class SmoothKLDAndTripletLoss(Criterion):

    def __init__(self, pad_idx: int, embedding: nn.Embedding, margin: float = 0.,
                 neg_region: float = 0.05, mode: str = 'dot', neg_sampling: str = 'random',
                 label_smoothing: float = 0.1, alpha: float = 1.0):
        super().__init__(input_type='identity')
        self.embeddings = embedding.weight
        self.smoothKLD = SmoothKLD(n_classes=embedding.weight.shape[0],
                                   label_smoothing=label_smoothing, pad_idx=pad_idx)
        self.tripletLoss = TripletLoss(embedding=embedding, margin=margin, neg_region=neg_region,
                                       mode=mode, neg_sampling=neg_sampling, pad_idx=pad_idx)
        self.alpha = alpha

    def forward(self, x, targets, mask_pad=True):
        smx = F.log_softmax(torch.einsum('bd,vd->bv', x, self.embeddings), dim=-1)
        sKLD = self.smoothKLD(smx, targets, mask_pad)
        tLoss = self.tripletLoss(x, targets)

        # Must sum here to match sizes
        return sKLD.sum() + self.alpha * tLoss.sum()
