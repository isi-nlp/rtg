#!/usr/bin/env python
#
# Author: Thamme Gowda [tg (at) isi (dot) edu] 
# Created: 3/16/19
from typing import Optional, Callable, List, Iterable
import copy

import torch
from torch import nn
import torch.nn.functional as F
#import rtg.dataprep as prep
from rtg.data.codec import Field
from rtg.data.dataset import SqliteFile, LoopingIterable, TSVData, IdExample
from rtg import device
from rtg.module import Model
from rtg.module.trainer import SteppedTrainer
from rtg import log
from dataclasses import dataclass
from rtg import TranslationExperiment as Experiment
from rtg.utils import IO
from tqdm import tqdm
import pickle
import numpy as np


class CBOW(Model):
    """
    continuous bag of words by Mikolov et al
    https://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf
    """

    def __init__(self, emb_dim, vocab_size, pad_idx, dropout=0.1):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        self.l1 = nn.Linear(emb_dim, emb_dim)
        self.l2 = nn.Linear(emb_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)
        self._model_dim = emb_dim
        self._vocab_size = vocab_size

    @property
    def model_dim(self):
        return self._model_dim

    @property
    def model_type(self):
        return 'wv_cbow'

    @property
    def vocab_size(self):
        return self._vocab_size

    def forward(self, ctx_ids):
        # [B x C x D] <- [B x C]
        ctx_embs = self.emb(ctx_ids)
        ctx_embs = self.dropout(ctx_embs)
        # [ B x D] <- [B x C x D]
        ctx_sum = ctx_embs.sum(dim=1)
        # [B x V] <- [B x D] <- B x D]
        nxt_word_weights = self.l2(self.dropout(self.l1(ctx_sum)))
        nxt_word_lprobs = F.softmax(nxt_word_weights, dim=-1)
        return nxt_word_lprobs

    @classmethod
    def make_model(cls, emb_dim, vocab_size, exp):
        model = cls(emb_dim, vocab_size, pad_idx=exp.tgt_vocab.pad_idx)
        model.init_params()

        args = dict(emb_dim=emb_dim, vocab_size=vocab_size, )
        return model, args

    @classmethod
    def make_trainer(cls, *args, **kwargs):
        return CBOWTrainer(*args, **kwargs)


@dataclass
class CBOWBatchReader:
    data: Iterable[IdExample]
    batch_size: int
    ctx_size: int
    side: str
    field: Field
    add_bos: bool = True
    add_eos: bool = True


    def __post_init__(self):
        assert self.side in {'src', 'tgt', 'src+tgt'}

    def _read_all_seqs(self):
        for ex in self.data:
            if 'src' in self.side:
                yield ex.x
            if 'tgt' in self.side:
                yield ex.y

    def _make_ctxs(self, seq: np.ndarray):
        # left_ctx + word + right_ctx
        if self.add_eos or self.add_bos:
            seq = list(seq)  # ndarray to list or a copy of list
            if self.add_bos and seq[0] != self.field.bos_idx:
                seq.insert(0, self.field.bos_idx)
            if self.add_eos and seq[-1] != self.field.eos_idx:
                seq.append(self.field.eos_idx)
        full_window = self.ctx_size + self.ctx_size
        for i in range(len(seq) - full_window):
            word = seq[i + self.ctx_size]
            ctx = seq[i:i + self.ctx_size] + seq[i + self.ctx_size + 1: i + 2 * self.ctx_size + 1]
            yield (ctx, word)

    def _make_tensors(self, batch):
        xs = torch.zeros(len(batch), self.ctx_size * 2, dtype=torch.long)
        ys = torch.zeros(len(batch), dtype=torch.long)
        for i, (x, y) in enumerate(batch):
            xs[i] = torch.tensor(x, dtype=torch.long)
            ys[i] = y
        return xs, ys

    def __iter__(self):
        batch = []
        for seq in self._read_all_seqs():
            for ex in self._make_ctxs(seq):
                batch.append(ex)
                if len(batch) == self.batch_size:
                    yield self._make_tensors(batch)
                    batch.clear()
        if batch:
            yield self._make_tensors(batch)


@dataclass
class DataReader:
    exp: Experiment
    side: str

    def get_training_data(self, batch_size, ctx_size, n_batches):
        train_db = SqliteFile(self.exp.train_db)
        reader = CBOWBatchReader(train_db, batch_size=batch_size, ctx_size=ctx_size, side=self.side,
                                 field=self.exp.src_vocab)
        return LoopingIterable(reader, batches=n_batches)

    def get_val_data(self, batch_size, ctx_size):
        data = TSVData(self.exp.valid_file, longest_first=False)
        return CBOWBatchReader(data, batch_size=batch_size, ctx_size=ctx_size, side=self.side,
                               field=self.exp.src_vocab)


class CBOWTrainer(SteppedTrainer):

    def __init__(self, *args, model_factory=CBOW.make_model, **kwargs):
        super().__init__(*args, model_factory=model_factory, **kwargs)
        assert isinstance(self.model, CBOW)  # type check
        self.model: CBOW = self.model   # type ann
        self.loss_func = nn.NLLLoss()

    def run_valid_epoch(self, data_iter: Iterable) -> float:
        log.info("Running validation")
        total_loss, n = 0.0, 0
        with tqdm(data_iter, unit='batch', dynamic_ncols=True) as data_bar:
            for i, (xs, ys) in enumerate(data_bar):
                xs, ys = xs.to(device), ys.to(device)
                log_probs = self.model(xs)
                loss = self.loss_func(log_probs, ys)
                total_loss += loss.item()
                n += len(ys)
        return total_loss / n

    def save_embeddings(self, step, train_loss, val_loss, txt=True):
        matrix = self.model.emb.weight
        vocab = self.exp.shared_vocab
        words = [vocab.id_to_piece(i) for i in range(len(vocab))]
        self.tbd.add_embedding(matrix, metadata=words, global_step=step)
        ext = 'txt.gz' if txt else 'pkl'
        path = self.exp.model_dir / f'embeddings_{step}_{train_loss:.6f}_{val_loss:.6f}.{ext}'
        log.info(f"writing  embedding after step {step} to {path}")
        if txt:
            with IO.writer(path) as w:
                w.write(f'{matrix.shape[0]} {matrix.shape[1]}\n')
                for i in range(matrix.shape[0]):
                    word = words[i]
                    vect = ' '.join(f'{x:g}' for x in matrix[i])
                    w.write(f'{word} {vect}\n')
        else:
            with path.open('wb') as f:
                data = {'words': words, 'vectors': matrix.numpy}
                pickle.dump(data, f)

    def train(self, steps: int, check_point: int, batch_size: int,
              check_pt_callback: Optional[Callable] = None, side='tgt', ctx_size=2, **args):
        log.info(f"using side={side}, ctx_size={ctx_size}")
        reader = DataReader(self.exp, side=side)
        rem_steps = steps - self.start_step
        if rem_steps <= 0:
            log.warning(f"Already trained upto {self.start_step-1}. Skipped")
            return
        train_data = reader.get_training_data(batch_size=batch_size, n_batches=rem_steps,
                                              ctx_size=ctx_size)
        val_data = reader.get_val_data(batch_size=batch_size, ctx_size=ctx_size)
        train_loss, n = 0.0, 0

        def _make_checkpt(step, train_loss):
            with torch.no_grad():
                val_loss = self.run_valid_epoch(val_data)
            log.info(f"Checkpoint at {step}")
            self.save_embeddings(step, train_loss, val_loss, txt=True)
            self.tbd.add_scalars('losses', {'training': train_loss,
                                            'valid_loss': val_loss}, global_step=step)

        with tqdm(train_data, initial=self.start_step, total=rem_steps+1, unit='batch',
                  dynamic_ncols=True) as data_bar:
            for i, (xs, ys) in enumerate(data_bar, start=self.start_step):
                self.model.zero_grad()
                xs, ys = xs.to(device), ys.to(device)
                log_probs = self.model(xs)
                loss = self.loss_func(log_probs, ys)
                self.tbd.add_scalars('training', {'step_loss': loss.item(),
                                                  'learn_rate': self.opt.curr_lr},
                                     self.opt.curr_step)
                progress_msg = f', loss={loss:g} LR={self.opt.curr_lr:g}'
                data_bar.set_postfix_str(progress_msg, refresh=False)
                train_loss += loss.item()
                n += len(ys)

                loss.backward()
                self.opt.step()
                self.opt.zero_grad()

                if i > 0 and i % check_point == 0:
                    _make_checkpt(i, train_loss / n)
                    train_loss, n = 0.0, 0
        if n > 0:
            _make_checkpt(steps, train_loss / n)


if __name__ == '__main__':
    # refer to examples/cbow.conf.yml for the spec
    pass
