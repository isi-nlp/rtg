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
from rtg import log
from rtg.exp import TranslationExperiment as Experiment
from pathlib import Path


def batch_dot(a, b):
    """Computes dot product in batch mode"""

    # return torch.mul(a, b).sum(dim=-1)
    assert a.shape == b.shape
    n_items, n_dims = a.shape
    result = torch.bmm(b.view(n_items, 1, n_dims), b.view(n_items, n_dims, 1))
    return result.squeeze_(1).squeeze_(1)


class Criterion(nn.Module, abc.ABC):
    """Base class for Criterion functions"""
    infinitesimal = 1e-8

    def __init__(self, input_type: str, exp: Experiment, reduction='micro'):
        """
        :param input_type: what type of input is expected?
            example: logits, softmax, log_softmax, signmoid
            This choice should be compatible with Generator.forward(x, score=xxx)
        :param pad_idx: index of padding
        """
        super().__init__()
        self.exp = exp
        self.pad_idx = self.exp.tgt_vocab.pad_idx
        self.input_type = input_type
        self.reduction = reduction


def smooth_labels(labels, n_labels, smooth_rate, ignore_idx=-1, weight=None):
    """
    :param labels: labels [N] where each item is in range {0, 1, 2,... C-1}
    :param n_labels: total number of classes
    :param smooth_rate: the magnitude of smoothing
    :param weight: distribute epsilon as per the weights
    :param ignore_idx: ignore this index (e.g. padding)
    :return:
    """
    assert len(labels.shape) == 1
    assert labels.max() < n_labels
    assert 0 <= labels.min()
    assert 0 <= labels.min()
    assert 0 <= smooth_rate < 1

    N = len(labels)
    labels = labels.view(N, 1)
    device = labels.device
    if weight is None:
        # take out epsilon and distribute evenly to all but one; exclude pad_idx
        fill_value = smooth_rate / (n_labels - (2 if ignore_idx >= 0 else 1))
        # expand [N] -> [N, C]
        full = torch.full([N, n_labels], fill_value=fill_value, dtype=torch.float, device=device)
        full.scatter_(1, labels.type(torch.int64), 1 - smooth_rate)
        if ignore_idx >= 0:
            full[:, ignore_idx] = 0.0
    else:
        assert len(weight) == n_labels
        weight = weight.to(device)
        full = (weight * smooth_rate).expand(N, n_labels)  # [C] -> [N, C]
        peaks = torch.full([N, 1], fill_value=1 - smooth_rate, dtype=torch.float, device=device)
        # peaks = torch.tensor(1 - epsilon, device=device).expand(N, 1)  # [N, 1]
        full.scatter_add_(1, labels, peaks)  # inplace add
    return full


def get_dense_targets(labels, n_labels, label_smoothing, ignore_idx=-1, weight=None):
    if label_smoothing > 0.0:
        dense_targets = smooth_labels(labels=labels, n_labels=n_labels, smooth_rate=label_smoothing,
                                      ignore_idx=ignore_idx, weight=weight)
    else:
        assert weight is None, 'weight is supported only if label_smoothing > 0'
        dense_targets = F.one_hot(labels, num_classes=n_labels).float().to(labels.device)
    return dense_targets


def dense_cross_entropy_old(inputs: Tensor, targets: Tensor, reduction=None, mask_out=None, weight=None,
                            input_type='logits') -> Tensor:
    """
    :param inputs: input tensor, see input_type
    :param targets: probability distribution of labels
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
    N, C = inputs.shape
    infinitesimal = 1e-9  # to avoid divide by zero
    # target is dense; i.e not one-hot
    assert inputs.shape == targets.shape, f'input shape: {inputs.shape}, target is {targets.shape}'
    if mask_out is not None:
        assert mask_out.shape in [(N, C), (1, C), (N, 1)], \
            f'input: {inputs.shape} and mask_out: {mask_out.shape} are not broadcastable'
    if weight is not None:
        assert weight.shape in [(N, C), (1, C), (N, 1)], \
            f'input: {inputs.shape} weight: {weight.shape} are not broadcastable'

    if input_type == 'log_probs':
        log_probs = inputs
    elif input_type == 'probs':
        # adding small num to avoid zeros
        log_probs = (inputs + infinitesimal).log()
    elif input_type == 'logits':
        log_probs = inputs.log_softmax(dim=1)
    else:
        raise Exception(f'input_type={input_type} unknown; know: logits, probs, log_probs')

    tot_items, tot_classes = N, C
    # [N, C] :  -y_c * log(p_c)
    # table = -torch.mul(targets, log_probs)  # loss per item per class
    table = torch.kl_div(inputs, targets)
    if mask_out is not None:  # overwrite positions with mask=True to zero
        table.masked_fill_(mask_out, value=0.0)
        if mask_out.shape == (N, 1):  # [N, 1] => sum items are excluded e.g. pad tokens
            tot_items -= mask_out.sum()
    assert tot_items > 0, f'tot_items should be positive; N={N}, C={C}, |mask_out|={mask_out.sum()}'
    if weight is not None:
        table.mul_(weight)  # assumption: weight is broadcastable

    if reduction is None or reduction == 'none':
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
        target_per_class = targets.sum(dim=0) + infinitesimal
        input_per_class = table.sum(dim=0)
        per_class_normalized = input_per_class / target_per_class
        return per_class_normalized.sum()
    elif reduction == 'macro+micro' or reduction == 'micro+macro':
        micro = table.sum(dim=1).sum() / tot_items
        macro = (table.sum(dim=0) / (targets.sum(dim=0) + infinitesimal)).sum()
        return macro + micro
    else:
        raise ValueError(f'reduce={reduction} not supported')


def dense_cross_entropy(inputs: Tensor, targets: Tensor, reduction='none', mask_out=None, weight=None,
                        input_type='log_probs', infinitesimal=1e-8) -> Tensor:
    assert input_type == 'log_probs'
    assert inputs.shape == targets.shape
    tot_items, tot_classes = inputs.shape
    losses = torch.kl_div(input=inputs, target=targets)
    if mask_out is not None:
        losses.masked_fill_(mask_out, value=0.0)
        if mask_out.shape == (tot_items, 1):  # [N, 1] => sum items are excluded e.g. pad tokens
            tot_items -= mask_out.sum()
    assert tot_items > 0, f'tot_items should be positive; input={inputs.shape}, |mask_out|={mask_out.sum()}'
    if weight is not None:
        losses.mul_(weight)  # assumption: weight is broadcastable

    if reduction == 'none':
        return losses
    if reduction == 'micro':
        # micro: first get loss per_item (get rid of class dim), and average over items
        return losses.sum(dim=1).sum() / tot_items
    elif reduction == 'macro':
        # micro: first get loss per_class, normalize as per their frequencies and then sum
        target_mass_per_class = targets.sum(dim=0) + infinitesimal
        # KL div per item (i.e. sum of each row in table) is guaranteed to be positive,
        # however per-item-per-class (i.e. each cell in the table) is not guaranteed to +ve
        loss_per_class = losses.pow(2).sum(dim=0)
        assert loss_per_class.shape == target_mass_per_class.shape
        loss_per_class_normalized = loss_per_class / target_mass_per_class
        return loss_per_class_normalized.mean()
    elif reduction == 'macro+micro':
        micro = losses.sum(dim=1).sum() / tot_items
        macro = (losses.pow(2).sum(dim=0) / (targets.sum(dim=0) + infinitesimal)).mean()
        return macro + micro
    else:
        raise ValueError(f'reduce={reduction} not supported; try: none micro macro macro+micro')


@register(kind=CRITERION, name="sparse_cross_entropy")
class SparseCrossEntropy(Criterion):

    def __init__(self, exp: Experiment, input_type='log_probs', weight=None, reduction='micro'):
        super().__init__(input_type=input_type, exp=exp, reduction=reduction)
        self.exp = exp
        self.weight_by = weight
        self._weight = None
        self.eos_idx = exp.tgt_vocab.eos_idx

    def forward(self, inputs, targets, mask_pad=True):
        # logits: [N x C] targets: [N]
        assert self.reduction == 'micro'
        assert targets.shape[0] == inputs.shape[0]
        weights = self.get_weights(inputs=inputs, targets=targets)
        losses = F.cross_entropy(input=inputs, target=targets, reduction='none', weight=weights)
        tot_items = targets.shape[0]
        if mask_pad:
            mask_out = targets.eq(self.pad_idx)
            losses.masked_fill_(mask_out, 0)
            tot_items -= mask_out.sum()
        return losses.sum() / tot_items

    def get_weights(self, inputs, targets=None):
        n_classes = inputs.shape[1]
        if not self.weight_by:
            return None  # nothing interesting here yet
        if self._weight is None:
            assert self.exp
            work_dir: Path = self.exp.work_dir
            wt_file = work_dir / 'classes.weights.tsv'
            if not wt_file.exists() or wt_file.stat().st_size == 0:
                cls_freqs = self.exp.get_class_freqs()
                assert len(cls_freqs) == n_classes
                freqs = [f for idx, c, f in cls_freqs]
                assert all(f >= 0 for f in freqs), 'frequency cannot be negative, but some are found to be negative'
                freqs = torch.tensor(freqs)
                high = freqs.max()
                if self.weight_by == 'inv_freq':
                    weight = high / freqs
                elif self.weight_by == 'inv_sqrt_freq':
                    weight = torch.sqrt(high) / freqs.sqrt()
                elif self.weight_by == 'inv_log_freq':
                    weight = torch.log(high) / freqs.log()
                else:
                    raise Exception(f'{self.weight_by} is not supported')
                min_weight = 1.0
                bad_pos = weight.isnan() | weight.isinf() | (weight < min_weight)
                weight.masked_fill_(bad_pos, min_weight)
                if self.eos_idx >= 0:
                    weight[self.eos_idx] = weight.max()
                assert len(weight) == len(cls_freqs)
                with wt_file.open('w', encoding='utf-8', errors='replace') as out:
                    for (idx, cls_name, freq), wt in zip(cls_freqs, weight):
                        line = f'{idx}\t{cls_name}\t{freq}\t{wt:g}\n'
                        out.write(line)
                log.info(f"created {wt_file}")

            with wt_file.open() as lines:
                recs = (line.strip().split('\t') for line in lines)
                # idx, name, freq, weight
                weights = [float(rec[3]) for rec in recs]
                assert len(weights) == inputs.shape[1]
                self._weight = torch.tensor(weights, dtype=inputs.dtype, device=inputs.device)

        assert (self._weight < 0).sum() == 0, 'weight cannot be negative'
        return self._weight


@register(kind=CRITERION, name="dense_cross_entropy")
class DenseCrossEntropy(SparseCrossEntropy):

    def __init__(self, exp: Experiment, label_smoothing=0., reduction='micro',
                 input_type='log_probs', weight=None):
        super().__init__(input_type=input_type, reduction=reduction, exp=exp, weight=weight)
        assert 0 <= label_smoothing <= 1
        self.exp = exp
        self.label_smoothing = label_smoothing
        assert reduction in ('micro', 'macro', 'macro+micro')
        if 'macro' in reduction:
            assert self.label_smoothing > 0., 'reduce=macro requires label_smoothing > 0'

    def forward(self, inputs, targets, mask_pad=True):
        # logits: [N x C] targets: [N]
        N, C = inputs.shape
        assert targets.shape[0] == inputs.shape[0]
        mask_out = None
        if mask_pad:
            mask_out = (targets == self.pad_idx).unsqueeze(1)  # [N] -> [N, 1]
        dense_targets = get_dense_targets(labels=targets, n_labels=C, label_smoothing=self.label_smoothing,
                                          ignore_idx=self.pad_idx)
        weight = self.get_weights(inputs, targets)
        return dense_cross_entropy(inputs=inputs, targets=dense_targets, reduction=self.reduction,
                                   weight=weight, mask_out=mask_out, input_type=self.input_type,
                                   infinitesimal=self.infinitesimal)


@register(kind=CRITERION, name="focal_loss")
class FocalLoss(DenseCrossEntropy):

    def __init__(self, exp: Experiment, label_smoothing=0., reduction='micro', gamma=0.0):
        super(FocalLoss, self).__init__(exp=exp, reduction=reduction, label_smoothing=label_smoothing,
                                        input_type='log_probs')
        assert gamma >= 0.0
        self.gamma = gamma

    def get_weights(self, inputs, targets=None):
        if self.input_type == 'probs':
            probs = inputs
        elif self.input_type == 'log_probs':
            probs = torch.exp(inputs)
        else:
            raise Exception(f'{self.input_type} not supported')
        weight = (1 - probs).pow(self.gamma)
        return weight


@register(kind=CRITERION, name="binary_cross_entropy")
class BinaryCrossEntropy(Criterion):

    def __init__(self, exp: Experiment, label_smoothing=0.1):
        assert 0 <= label_smoothing < 1
        super().__init__(input_type='logits', exp=exp)
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='none')
        self.smoothing = label_smoothing

    def forward(self, logits, targets, mask_pad=True):
        # logits: [B x V] targets: [B]
        assert targets.shape[0] == logits.shape[0]
        targets = targets.unsqueeze(1)

        truth_full = torch.full_like(logits, fill_value=self.smoothing, requires_grad=False)
        # truth_full = torch.zeros_like(logits, requires_grad=False)
        truth_full.scatter_(1, targets, 1)

        per_time_per_class_loss = self.bce_loss(logits, truth_full)
        if mask_pad:
            pad_mask = targets == self.pad_idx
            per_time_per_class_loss.masked_fill_(mask=pad_mask, value=0.)

        # num_toks = batch_size - pad_mask.sum()
        # mean_loss = per_tok_loss.sum() / num_toks
        per_tok_loss = per_time_per_class_loss.sum(dim=-1)
        return per_tok_loss


@register(kind=CRITERION, name="smooth_kld")
class SmoothKLD(Criterion):
    """
    Label smoothing
    """

    def __init__(self, exp: Experiment, n_classes: int, label_smoothing: float = 0.1):
        super().__init__(input_type='log_softmax', exp=exp, reduction='micro')
        self.size = n_classes
        assert 0.0 <= label_smoothing <= 1.0
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
        tot_toks = x.shape[0]
        if mask_pad:
            mask = target.eq(self.pad_idx)
            smooth_truth.masked_fill_(mask, 0)
            # note: x is log probs, smooth_truth is just probs
            # D (P || Q) = - \sum p(x) log[q(x)/p(x)]
            # mask is done by setting p(x)=0 for pad toks
            tot_toks -= mask.sum()
        loss_per_item = F.kl_div(x, smooth_truth, reduction='none')
        # micro reduction
        loss = loss_per_item.sum() / tot_toks
        return loss


@register(kind=CRITERION, name="triplet_loss")
class TripletLoss(Criterion):
    # Note: Triplet loss doesnt work fully yet; it sorta works and then overfits

    def __init__(self, exp: Experiment, embedding: nn.Embedding, margin: float = 0.,
                 neg_region: float = 0.05, mode: str = 'dot', neg_sampling: str = 'random'):
        # TODO: whats the right margin?
        super().__init__(input_type='embedding', exp=exp)
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
        anchors = x  # [B x D]
        pos_embs = self.embedding(targets)  # [B x D]
        if self.neg_sampling == 'random':
            neg_ids = torch.randint_like(targets, low=self.pad_idx + 1, high=self.vocab_size)  # [B]
        elif self.neg_sampling == 'hard':
            candidates = torch.einsum('bd,vd->bv', anchors, self.embeddings)  # [B, V]
            mask = candidates.new_full(candidates.shape, False, dtype=torch.bool)  # [B, V]
            mask.scatter_(dim=1, index=targets.view(-1, 1), value=True)
            candidates = candidates.masked_fill(mask, -1)
            c, indexes = torch.sort(candidates)
            neg_ids = indexes[:, -self.hard_neg_region].contiguous().view(-1)  # [B]
        else:
            raise Exception(self.neg_sampling + ' not supported')
        neg_embs = self.embedding(neg_ids)  # [B x D]

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

    def __init__(self, exp: Experiment, embedding: nn.Embedding, margin: float = 0.,
                 neg_region: float = 0.05, mode: str = 'dot', neg_sampling: str = 'random',
                 label_smoothing: float = 0.1, alpha: float = 1.0):
        super().__init__(input_type='identity', exp=exp)
        self.embeddings = embedding.weight
        self.smoothKLD = SmoothKLD(n_classes=embedding.weight.shape[0],
                                   label_smoothing=label_smoothing, exp=exp)
        self.tripletLoss = TripletLoss(embedding=embedding, margin=margin, neg_region=neg_region,
                                       mode=mode, neg_sampling=neg_sampling, exp=exp)
        self.alpha = alpha

    def forward(self, x, targets, mask_pad=True):
        smx = F.log_softmax(torch.einsum('bd,vd->bv', x, self.embeddings), dim=-1)
        sKLD = self.smoothKLD(smx, targets, mask_pad)
        tLoss = self.tripletLoss(x, targets)

        # Must sum here to match sizes
        return sKLD.sum() + self.alpha * tLoss.sum()


@register(kind=CRITERION, name="dice_loss")
class DiceLoss(Criterion):

    def __init__(self, exp: Experiment = None, reduction='micro', gamma=0.0, additive_smoothing=1,
                 label_smoothing=0.0, squared_denominator=False):
        super().__init__(exp=exp, input_type='logits', reduction=reduction)
        if gamma > 0.0:
            log.warning(">>>>> gamma, ɣ > 0,  is not well tested;")
        self.gamma = gamma
        self.squared_denominator = squared_denominator
        self.additive_smoothing = additive_smoothing
        self.label_smoothing = label_smoothing

    def forward(self, inputs, targets, mask_pad=True):
        probs = inputs.softmax(dim=1)
        # probs = inputs.sigmoid()
        N, C = probs.shape
        assert targets.shape[0] == probs.shape[0]
        mask_in = None
        if mask_pad:
            mask_in = (targets != self.pad_idx).unsqueeze(1).float()  # [N] -> [N, 1]
        dense_targets = get_dense_targets(labels=targets, n_labels=C, label_smoothing=self.label_smoothing,
                                          ignore_idx=self.pad_idx)
        weight = None
        if self.gamma > 0.0:
            weight = (1 - probs).pow(self.gamma)
        loss = self.dice_loss(inputs=inputs, targets=dense_targets, mask_in=mask_in, weight=weight)
        return loss

    def dice_loss(self, inputs: Tensor, targets: Tensor, mask_in=None, weight=None):
        assert inputs.shape == targets.shape, f'{inputs.shape} == {targets.shape}?'
        assert len(inputs.shape) == 2  # two dims: [Batch, Classes]
        assert weight is None, f'weight is not supported as of now'
        assert self.reduction == 'micro', f'reduction={self.reduction} is not supported; only micro is supported now'
        # y: target p: input  both are distributions
        # intersection = y . p     ; batch dot product
        intersection = torch.mul(targets, inputs).sum(dim=-1)  # [N, C] [N, C] --> [N]
        total_items = inputs.shape[0]
        if mask_in is not None:
            inputs = inputs * mask_in
            targets = targets * mask_in
            total_items = mask_in.sum()
        numerator = 2 * intersection + self.additive_smoothing
        if self.squared_denominator:
            denominator = inputs.pow(2).sum(dim=-1) + targets.pow(2).sum(dim=-1)  # [N]
        else:
            denominator = inputs.sum(dim=-1) + targets.sum(dim=-1)  # [N]
        similarity = numerator / (numerator + denominator)
        loss = 1 - similarity  # [N]
        loss = loss.sum() / total_items  # [N] --> [1]
        return loss


@register(kind=CRITERION, name="squared_error")
class SquaredError(Criterion):

    def __init__(self, exp: Experiment, label_smoothing: float = 0.1, reduction='micro'):
        super().__init__(exp=exp, input_type='probs', reduction=reduction)
        self.label_smoothing = label_smoothing
        assert 0.0 <= label_smoothing <= 1.0
        assert reduction in ('micro', 'macro')

    def forward(self, x, target, mask_pad=True):
        # 'x' is log probabilities, originally [B, T, V], but here [B.T, V]
        # 'target' is expected word Ids, originally [B, T] but here [B.T]
        assert x.shape[0] == target.shape[0]
        n_labels = x.shape[1]
        smooth_truth = smooth_labels(labels=target, n_labels=n_labels, smooth_rate=self.label_smoothing,
                                     ignore_idx=self.pad_idx)
        tot_toks = x.shape[0]
        if mask_pad:
            mask = target.eq(self.pad_idx).unsqueeze(1)
            smooth_truth.masked_fill_(mask, 0)
            x = x.masked_fill(mask, 0)
            tot_toks -= mask.sum()
        assert tot_toks > 0
        dist_sq = (x - smooth_truth).pow(2)
        loss_per_item_per_cls = dist_sq
        if self.reduction == 'micro':
            loss_per_item = loss_per_item_per_cls.sum(dim=1)  # sum along rows
            loss = loss_per_item.sum() / tot_toks
        elif self.reduction == 'macro':
            mass_per_cls = smooth_truth.sum(dim=0) + self.infinitesimal
            loss_per_cls = loss_per_item_per_cls.sum(dim=0)  # sum along rows
            loss_per_cls /= mass_per_cls + self.infinitesimal
            return loss_per_cls.sum() / n_labels
        else:
            raise Exception(f'Reduction {self.reduction} not supported; try micro or macro')
        return loss
