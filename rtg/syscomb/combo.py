#!/usr/bin/env python
#
# Author: Thamme Gowda [tg (at) isi (dot) edu] 
# Created: 1/3/19
from typing import List, Union, Optional
from rtg.exp import TranslationExperiment
from rtg.module import NMTModel
from rtg import device
from pathlib import Path
import torch
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm
from rtg.module.criterion import SmoothKLD
from rtg import log, yaml
from rtg.utils import IO
from rtg.lm.rnnlm import RnnLm
from rtg.lm.tfmlm import TfmLm


class RnnLmWrapper(nn.Module):
    """
    Wraps a RNN language model to provide a translation model like API for Sys Comb
    """

    def __init__(self, model: RnnLm):
        super().__init__()
        self.model = model
        self.rnn_state = None
        self.prev_seq_len = 0

    @property
    def vocab_size(self):
        return self.model.vocab_size

    @property
    def model_type(self):
        return self.model_type

    def encode(self, x_seqs, x_lens):
        pass  # No Op

    def decode(self, ys, gen_probs=True, log_probs=False):

        seq_len = ys.shape[1]
        if seq_len == 1:
            # Note: we need to keep track of rnn_state; but the API was designed for the transformer
            #  model which doesnt have the state concept. So this is a hacky way to get stuff done
            #  without redesigning the API
            # new sequence --> reset the state
            self.rnn_state = None
            self.prev_seq_len = 0
        assert seq_len == self.prev_seq_len + 1
        last_ys = ys[:, -1]
        out_probs, self.rnn_state, _ = self.model(None, last_ys, last_hidden=self.rnn_state,
                                                  gen_probs=gen_probs, log_probs=log_probs)
        self.prev_seq_len += 1
        return out_probs

    def forward(self, x_seqs, y_seqs, x_mask, y_mask, gen_probs: bool = True, log_probs=False):
        assert gen_probs
        # x_seqs and x_mask are useless for a Language Model
        batch, max_time = y_seqs.shape
        result = torch.zeros(batch, max_time, self.model.vocab_size, device=device)
        rnn_state = None
        x_mem = None
        for i in range(max_time):
            result[:, i, :], rnn_state, _ = self.model(x_mem, y_seqs[:, i], last_hidden=rnn_state,
                                                       gen_probs=gen_probs, log_probs=log_probs)
        return result


class TfmLmWrapper(nn.Module):
    """
    Wraps a Transformer language model to provide a translation model like API for  Sys Comb
    """

    def __init__(self, model: TfmLm):
        super().__init__()
        self.model = model
        self.generator = model.generator

    @property
    def vocab_size(self):
        return self.model.vocab_size

    @property
    def model_type(self):
        return self.model_type

    def encode(self, x_seqs, x_lens):
        pass  # No Op

    def decode(self, x_mem, x_mask, past_ys, y_mask):
        return self.model(past_ys, y_mask, gen_probs=False)

    def forward(self, x_seqs, y_seqs, x_mask, y_mask, gen_probs: bool = True, log_probs=False):
        return self.model(y_seqs, y_mask, gen_probs=gen_probs, log_probs=log_probs)


class Combo(nn.Module):
    """
    This module combines multiple models by ensembling the last layer.
    It performs a simple linear combination of distributions produced by the softmax layer
    (i.e. last layer)

    The weights of a model should be learned from a dataset held out from train and test.
    """

    wrappers = dict(rnnlm=RnnLmWrapper, tfmlm=TfmLmWrapper)

    def __init__(self, models: List[NMTModel], model_paths: Optional[List[Path]] = None,
                 w: Optional[List[float]] = None):
        super().__init__()
        assert type(models) is list
        # TODO: check if list breaks the computation graph? we don't want to propagate the loss
        # Optionally wrap models in wrappers
        models = [self.wrappers[m.model_type](m) if m.model_type in self.wrappers
                  else m for m in models]
        self.models = models
        self.model_paths = model_paths
        self.n_models = len(models)
        if w is None:
            # equal weight to begin with
            w_init = [1.0 / self.n_models] * self.n_models
        else:
            assert len(w) == len(models)
            assert all(x >= 0 for x in w)
            assert abs(sum(w) - 1.0) < 0.00001
            w_init = w
        self.weight = nn.Parameter(torch.tensor(w_init, dtype=torch.float))
        self.tgt_vocab_size = models[0].vocab_size
        for m in models:
            assert m.vocab_size == self.tgt_vocab_size
        self.vocab_size = models[0].vocab_size

    def to(self, device):
        super().to(device)
        # self.weights = self.weights.to(device)
        # the python list  cuts the pytorch graph, so we need to do this
        self.models = [m.to(device) for m in self.models]
        return self

    def forward(self, batch):
        # [n=models x batch x time ]
        w_probs = F.softmax(self.weight, dim=0)
        result_distr = None

        x_seqs = batch.x_seqs
        x_mask = (batch.x_seqs != batch.pad_val).unsqueeze(1)
        bos_step = torch.full((len(batch), 1), fill_value=batch.bos_val, dtype=torch.long,
                              device=device)
        y_seqs_with_bos = torch.cat([bos_step, batch.y_seqs], dim=1)
        y_mask = batch.make_autoreg_mask(y_seqs_with_bos)

        for i, model in enumerate(self.models):
            distr_i = model(x_seqs, y_seqs_with_bos, x_mask, y_mask,
                            gen_probs=True, log_probs=False)
            distr_i = distr_i[:, :-1, :]  # Skip the output after the EOS time step
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
            if isinstance(model, RnnLmWrapper):
                probs = model.decode(past_ys, gen_probs=True, log_probs=False)
            else:
                # model is a TransformerNMT
                y_feats = model.decode(x_mem, x_mask, past_ys, y_mask)
                probs = model.generator(y_feats[:, -1], log_probs=False)
            result += w * probs
        return result.log()

    @property
    def model_weights(self):
        return F.softmax(self.weight.data, dim=0)


class SysCombTrainer:

    def __init__(self, models: List[Path], exp: Union[Path, TranslationExperiment],
                 lr: float = 1e-4, smoothing=0.1):
        if isinstance(exp, Path):
            exp = TranslationExperiment(exp)
        self.w_file = exp.work_dir / f'combo-weights.yml'

        wt = None
        if self.w_file.exists():
            with IO.reader(self.w_file) as rdr:
                combo_spec = yaml.load(rdr)
            weights = combo_spec['weights']
            assert len(weights) == len(models)  # same models as before: no messing allowed
            model_path_strs = [str(m) for m in models]
            for m in model_path_strs:
                assert m in weights, f'{m} not found in weights file.'
            wt = [weights[str(m)] for m in model_path_strs]
            log.info(f"restoring previously stored weights {wt}")

        from rtg.module.decoder import load_models
        combo = Combo(load_models(models, exp), model_paths=models, w=wt)
        self.combo = combo.to(device)
        self.exp = exp
        self.optim = torch.optim.Adam(combo.parameters(), lr=lr)
        self.criterion = SmoothKLD(vocab_size=combo.vocab_size,
                                   padding_idx=exp.tgt_vocab.pad_idx,
                                   smoothing=smoothing)

    def train(self, steps: int, batch_size: int):
        log.info(f"Going to train for {steps}")
        batches = self.exp.get_combo_data(batch_size=batch_size, steps=steps)
        with tqdm(batches, total=steps, unit='step', dynamic_ncols=True) as data_bar:
            for i, batch in enumerate(data_bar):
                batch = batch.to(device)
                y_probs = self.combo(batch)  # B x T x V
                loss = self.loss_func(y_probs, y_seqs=batch.y_seqs, norm=batch.y_toks)
                wt_str = ','.join(f'{wt:g}' for wt in self.combo.weight)
                progress_msg = f'loss={loss:g}, weights={wt_str}'
                data_bar.set_postfix_str(progress_msg, refresh=False)

        weights = dict(zip([str(x) for x in self.combo.model_paths],
                           self.combo.model_weights.tolist()))
        log.info(f" Training finished. {weights}")
        with IO.writer(self.w_file) as wtr:
            yaml.dump(dict(weights=weights), wtr, default_flow_style=False)

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
