#!/usr/bin/env python
#
# Author: Thamme Gowda [tg (at) isi (dot) edu] 
# Created: 1/3/19
from typing import List, Union
from rtg.exp import TranslationExperiment
from rtg.module import NMTModel
from rtg import device
from pathlib import Path
import torch
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm
from rtg.module.tfmnmt import LabelSmoothing
from rtg.dataprep import PAD_TOK_IDX


class Combo(nn.Module):
    """
    This module combines multiple models by ensembling the last layer.
    It performs a simple linear combination of distributions produced by the softmax layer
    (i.e. last layer)

    The weights of a model should be learned from a dataset held out from train and test.
    """

    def __init__(self, models: List[NMTModel], w=None):
        super().__init__()
        assert type(models) is list
        # TODO: check if list breaks the computation graph? we don't want to propagate the loss
        self.models = models
        self.n_models = len(models)
        if w is None:
            # equal weight to begin with
            w_init = [1.0 / self.n_models] * self.n_models
        else:
            assert len(w) == len(models)
            assert all(x >= 0 for x in w)
            assert abs(sum(w) - 1.0) < 0.00001
            w_init = w
        self.weight = nn.Parameter(torch.tensor(w_init, device=device, dtype=torch.float))
        self.tgt_vocab_size = models[0].vocab_size
        for m in models:
            assert m.vocab_size == self.tgt_vocab_size
        self.vocab_size = models[0].vocab_size

    def forward(self, batch):
        # [n=models x batch x time ]
        w_probs = F.softmax(self.weight, dim=0)
        result_distr = None
        for i, model in enumerate(self.models):
            distr_i = model(batch.x_seqs, batch.y_seqs, batch.x_mask, batch.y_mask, gen_probs=True)
            # assumption: model did not give log_probs, it gave raw probs (sums to 1)
            distr_i = w_probs[i] * distr_i.detach()
            if i == 0:
                result_distr = distr_i
            else:
                result_distr += distr_i
        # assumption: we have raw probs, need to return log_probs
        return result_distr.log()

    def encode(self, x_seqs, x_len):
        x_mask = (x_seqs != PAD_TOK_IDX).unsqueeze(1)
        return [model.encode(x_seqs, x_mask) for model in self.models]

    def generate_next(self, x_mems, x_mask, past_ys, y_mask):
        assert len(x_mems) == self.n_models
        batch_size = x_mask.shape[0]
        result = torch.zeros(batch_size, self.tgt_vocab_size, device=device)
        for x_mem, model, w in zip(x_mems, self.models, self.model_weights):
            y_feats = model.decode(x_mem, x_mask, past_ys, y_mask)
            probs = model.generator(y_feats[:, -1], log=False)
            result += w * probs
        return result.log()

    @property
    def model_weights(self):
        return F.softmax(self.weight.data, dim=0)


class SysCombTrainer:

    def __init__(self, combo: Union[Combo, List[Path]], exp: Union[Path, TranslationExperiment],
                 lr: float = 1e-4, smoothing=0.1):
        if isinstance(exp, Path):
            exp = TranslationExperiment(exp)
        if type(combo) is list:
            from rtg.module.decoder import load_models
            combo = Combo(load_models(combo, exp))
        self.combo = combo
        self.exp = exp
        self.optim = torch.optim.Adam(combo.parameters(), lr=lr)
        self.criterion = LabelSmoothing(vocab_size=combo.vocab_size,
                                        padding_idx=PAD_TOK_IDX,
                                        smoothing=smoothing)

    def train(self, steps: int, batch_size: int):
        batches = self.exp.get_combo_data(batch_size=batch_size, steps=steps)
        with tqdm(batches, total=steps, unit='step') as data_bar:
            for i, batch in enumerate(data_bar):
                y_probs = self.combo(batch)  # B x T x V

                loss = self.loss_func(y_probs, y_seqs=batch.y_seqs, norm=batch.y_toks)
                progress_msg = f'loss={loss}, weights={self.combo.model_weights}'
                data_bar.set_postfix_str(progress_msg, refresh=False)

    def loss_func(self, y_probs, y_seqs, norm, train_mode=True):
        scores = y_probs.contiguous().view(-1, y_probs.size(-1))  # B x T x V --> B.T x V
        truth = y_seqs.contiguous().view(-1)  # B x T --> B.T
        assert norm != 0
        loss = self.criterion(scores, truth).sum() / norm
        if train_mode:  # don't do this for validation set
            loss.backward()
            self.optim.step()
            self.optim.zero_grad()
        return loss.item() * norm
