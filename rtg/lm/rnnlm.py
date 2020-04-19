#!/usr/bin/env python
#
# Author: Thamme Gowda [tg (at) isi (dot) edu] 
# Created: 1/31/19

import torch
from typing import Callable, Optional
from rtg import log, device, my_tensor as tensor
from rtg.data.dataset import padded_sequence_mask

import inspect
from rtg.module.rnnmt import Embedder, Generator, SeqDecoder
from rtg import TranslationExperiment as Experiment
import random
from rtg.module.trainer import TrainerState, SteppedTrainer
from tqdm import tqdm
from . import LanguageModel


class RnnLm(SeqDecoder, LanguageModel):

    def __init__(self, embedder: Embedder, generator: Generator, n_layers: int = 1,
                 dropout: float = 0.1):
        super().__init__(prev_emb_node=embedder, generator=generator, n_layers=n_layers,
                         dropout=dropout)
        assert embedder.emb_size == generator.vec_size
        self._model_dim = embedder.emb_size
        self._vocab_size = generator.vocab_size

    @property
    def model_dim(self):
        return self._model_dim

    @property
    def model_type(self):
        return 'rnnlm'

    @property
    def vocab_size(self):
        return self._vocab_size

    @classmethod
    def make_model(cls, lang: str, vocab_size: int, model_dim: int = 300, n_layers: int = 1,
             dropout: float = 0.1, exp: Experiment=None):
        # get all args for reconstruction at a later phase
        _, _, _, kwargs = inspect.getargvalues(inspect.currentframe())
        for exclusion in ['cls', 'exp']:
            del kwargs[exclusion]  # exclude some args
        emb = Embedder(name=lang, vocab_size=vocab_size, emb_size=model_dim)
        gen = Generator(name=lang, vec_size=model_dim, vocab_size=vocab_size)
        # Todo: tying stuff here
        model = cls(embedder=emb, generator=gen, n_layers=n_layers, dropout=dropout)
        model.init_params()
        return model, kwargs

    def batch_forward(self, batch):
        assert batch.batch_first
        batch_size = len(batch)

        assert not batch.has_y
        seqs = batch.x_seqs
        max_seq_len = batch.max_x_len

        prev_out = tensor([[batch.bos_val]] * batch_size, dtype=torch.long)
        last_hidden = None
        outp_probs = torch.zeros((max_seq_len - 1, batch_size), device=device)

        for t in range(1, max_seq_len):
            word_probs, last_hidden, _ = self(enc_outs=None, prev_out=prev_out,
                                              last_hidden=last_hidden)

            # expected output;; log probability for these indices should be high
            expct_word_idx = seqs[:, t].view(batch_size, 1)
            expct_word_log_probs = word_probs.gather(dim=1, index=expct_word_idx)
            outp_probs[t - 1] = expct_word_log_probs.squeeze()

            # Randomly switch between gold and the prediction next word
            if random.choice((False, True)):
                prev_out = expct_word_idx  # Next input is current target
            else:
                pred_word_idx = word_probs.argmax(dim=1)
                prev_out = pred_word_idx.view(batch_size, 1)
        return outp_probs.t()


class RnnLmTrainer(SteppedTrainer):

    def __init__(self, exp: Experiment, model: RnnLm=None, model_factory=RnnLm.make_model, **optim_args):
        super().__init__(exp=exp, model=model, model_factory=model_factory, **optim_args)

    def simple_loss_func(self, log_probs, seq_lens, tot_toks=None, max_seq_len=None,
                    train_mode: bool=True) -> float:
        per_tok_loss = -log_probs
        if max_seq_len is None:
            max_seq_len = seq_lens.max()
        if tot_toks is None:
            tot_toks = seq_lens.sum()
        norm = tot_toks
        tok_mask = padded_sequence_mask(seq_lens, max_seq_len - 1)
        loss = (per_tok_loss * tok_mask.float()).sum().float() / norm
        if train_mode:
            loss.backward()
            self.opt.step()
            self.opt.zero_grad()
        return loss.item() * norm

    def run_valid_epoch(self, data_iter):
        state = TrainerState(self.model, -1)
        with tqdm(data_iter, total=data_iter.num_batches, unit='batch',
                  dynamic_ncols=True) as data_bar:
            for i, batch in enumerate(data_bar):
                batch = batch.to(device)
                # Step clear gradients
                self.model.zero_grad()
                # Step Run forward pass.
                outp_log_probs = self.model.batch_forward(batch)
                loss = self.simple_loss_func(outp_log_probs, seq_lens=batch.x_len,
                                             tot_toks=batch.x_toks, max_seq_len=batch.max_x_len,
                                             train_mode=False)
                bar_msg, _ = state.step(batch.x_toks, loss)
                data_bar.set_postfix_str(bar_msg, refresh=False)
                del batch
        return state.running_loss()

    def train(self, steps: int, check_point: int, batch_size: int,
              check_pt_callback: Optional[Callable] = None, **args):
        train_state = TrainerState(self.model, check_point=check_point)
        train_state.train_mode(True)
        if self.start_step >= steps:
            log.warning(f"Already trained to  {self.start_step}. Considering it as done.")
            return
        rem_steps = steps - self.start_step
        side = 'tgt'     # TODO: this should be inferrable or configurable instead of hardcoded

        train_data = self.exp.get_mono_data('train', side, batch_size=batch_size,
                                            batch_first=True, sort_dec=True,
                                            num_batches=rem_steps, shuffle=True)
        val_data = self.exp.get_mono_data('valid', side, batch_size=batch_size,
                                          batch_first=True, sort_dec=True)

        keep_models = 8
        unsaved_state = False
        with tqdm(train_data, initial=self.start_step, total=steps, unit='batch',
                  dynamic_ncols=True) as data_bar:
            for batch in data_bar:
                batch.to(device)
                outp_log_probs = self.model.batch_forward(batch)
                loss = self.simple_loss_func(outp_log_probs, seq_lens=batch.x_len,
                                             tot_toks=batch.x_toks, max_seq_len=batch.max_x_len,
                                             train_mode=True)
                unsaved_state = True
                bar_msg, is_check_pt = train_state.step(batch.x_toks, loss)
                data_bar.set_postfix_str(bar_msg, refresh=True)
                del batch       # TODO: force free memory
                if is_check_pt:
                    train_loss = train_state.reset()
                    train_state.train_mode(False)
                    val_loss = self.run_valid_epoch(val_data)
                    self.make_check_point(train_loss, val_loss=val_loss, keep_models=keep_models)
                    if check_pt_callback:
                        check_pt_callback(model=self.model,
                                          step=self.opt.curr_step,
                                          train_loss=train_loss)
                    train_state.train_mode(True)
                    unsaved_state = False

        log.info("End of training session")
        if unsaved_state:
            # End of training
            train_loss = train_state.reset()
            train_state.train_mode(False)
            val_loss = self.run_valid_epoch(val_data)
            self.make_check_point(train_loss, val_loss=val_loss, keep_models=keep_models)


def test_lm():
    #model, args = RnnLm.make('eng', 8000)
    work_dir = '/Users/tg/work/me/rtg/saral/runs/1S-rnnlm-basic'
    exp = Experiment(work_dir)
    trainer = RnnLmTrainer(exp=exp)
    trainer.train(steps=2000, check_point=100, batch_size=64)


if __name__ == '__main__':
    test_lm()
