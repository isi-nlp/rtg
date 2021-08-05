#!/usr/bin/env python
#
# Author: Thamme Gowda [tg (at) isi (dot) edu] 
# Created: 2/7/19

import inspect
import copy
from typing import Optional, Callable

import torch
from torch import nn
from rtg import log, device, BatchIterable, Batch
from rtg.lm import LanguageModel
from rtg import TranslationExperiment as Experiment
from rtg.module.tfmnmt import (Generator, Embeddings, PositionalEncoding,
                               MultiHeadedAttention, PositionwiseFeedForward, TransformerTrainer)

from rtg.module.trainer import TrainerState
from tqdm import tqdm
import time


""""In NMT, DecoderLayer also has source attention.
But here, decoder layer is just like Encoder layer: self_attn and feed forward"""
from rtg.module.tfmnmt import EncoderLayer as LMDecoderLayer
from rtg.module.tfmnmt import Encoder as LMDecoder


class TfmLm(LanguageModel):

    def __init__(self, decoder: LMDecoder, embedder, generator: Generator):
        super().__init__()
        self.decoder: LMDecoder = decoder
        self.embed = embedder
        self.generator: Generator = generator

    @property
    def model_dim(self):
        return self.generator.d_model

    @property
    def model_type(self):
        return 'tfmlm'

    @property
    def vocab_size(self):
        return self.generator.vocab

    def forward(self, y_seqs, y_mask, gen_probs=False, log_probs=False):
        feats = self.decoder(self.embed(y_seqs), y_mask)
        return self.generator(feats, log_probs=log_probs) if gen_probs else feats

    @classmethod
    def make_model(cls, vocab_size, n_layers=6, hid_size=512, ff_size=2048,
                   n_heads=8, dropout=0.1, tied_emb=True, exp: Experiment = None):
        # get all args for reconstruction at a later phase
        _, _, _, args = inspect.getargvalues(inspect.currentframe())
        for exclusion in ['cls', 'exp']:
            del args[exclusion]  # exclude some args
        # In case you are wondering, why I didnt use **kwargs here:
        #   these args are read from conf file where user can introduce errors, so the parameter
        #   validation and default value assignment is implicitly done by function call for us :)

        c = copy.deepcopy
        attn = MultiHeadedAttention(n_heads, hid_size)
        ff = PositionwiseFeedForward(hid_size, ff_size, dropout)
        dec_layer = LMDecoderLayer(hid_size, c(attn), c(ff), dropout)
        decoder = LMDecoder(dec_layer, n_layers)
        embedr = nn.Sequential(Embeddings(hid_size, vocab_size),
                               PositionalEncoding(hid_size, dropout))
        generator = Generator(hid_size, vocab_size)

        model = TfmLm(decoder, embedr, generator)
        if tied_emb:
            log.info("Tying the embedding weights, two ways: (TgtIn == TgtOut)")
            model.generator.proj.weight = model.embed[0].lut.weight

        model.init_params()
        return model, args


class TfmLmTrainer(TransformerTrainer):

    def __init__(self, *args, model_factory=TfmLm.make_model, **kwargs):
        super().__init__(*args, model_factory=model_factory, **kwargs)
        self.model: TfmLm = self.model  # type annotation

    def run_valid_epoch(self, data_iter: BatchIterable) -> float:
        start = time.time()
        total_tokens = 0
        total_loss = 0.0
        with tqdm(data_iter, total=data_iter.num_batches, unit='batch',
                  dynamic_ncols=True) as data_bar:
            for i, batch in enumerate(data_bar):
                batch = batch.to(device)
                num_toks = batch.x_toks
                seqs = batch.x_seqs
                bos_step = torch.full((len(batch), 1), fill_value=batch.bos_val, dtype=torch.long,
                                      device=device)
                seqs_with_bos = torch.cat([bos_step, seqs], dim=1)
                seq_mask = batch.make_autoreg_mask(seqs_with_bos)
                out = self.model(seqs_with_bos, seq_mask, gen_probs=False)
                # [Batch x Time x D]
                # skip the last time step (the one with EOS as input)
                out = out[:, :-1, :]
                # assumption:  y_seqs has EOS, and not BOS
                loss = self.loss_func(out, seqs, num_toks, False)
                total_loss += loss
                total_tokens += num_toks
                elapsed = time.time() - start
                data_bar.set_postfix_str(
                    f'Loss:{loss:.4f}, {int(num_toks / elapsed)}toks/s', refresh=False)
                start = time.time()
                # force free memory
                del batch

        score = total_loss / total_tokens
        return score

    def train(self, steps: int, check_point: int, batch_size: int,
              check_pt_callback: Optional[Callable] = None, keep_models=4, **args):
        log.info(f'Going to train for {steps} epochs; batch_size={batch_size}; '
                 f'check point size:{check_point}')

        rem_steps = steps - self.start_step
        if rem_steps <= 0:
            raise Exception(f'The model was already trained to {self.start_step} steps. '
                            f'Please increase the steps or clear the existing models')
        side = 'tgt'  # TODO: this should be inferrable or configurable instead of hardcoded

        train_data = self.exp.get_mono_data('train', side, batch_size=batch_size,
                                            batch_first=True, sort_dec=False,
                                            num_batches=rem_steps, shuffle=True)
        val_data = self.exp.get_mono_data('valid', side, batch_size=batch_size,
                                          batch_first=True, sort_dec=False)

        train_state = TrainerState(self.model, check_point=check_point)
        train_state.train_mode(True)
        unsaved_state = False
        with tqdm(train_data, initial=self.start_step, total=steps, unit='batch',
                  dynamic_ncols=True) as data_bar:
            for batch in data_bar:
                self.model.zero_grad()
                assert batch.eos_x   # must have EOS
                batch = batch.to(device)
                num_toks = batch.x_toks
                seqs = batch.x_seqs
                bos_step = torch.full((len(batch), 1), fill_value=self.exp.tgt_vocab.bos_idx,
                                      dtype=torch.long, device=device)
                seqs_with_bos = torch.cat([bos_step, batch.x_seqs], dim=1)
                seq_mask = batch.make_autoreg_mask(seqs_with_bos)
                out = self.model(seqs_with_bos, seq_mask, gen_probs=False)
                # [Batch x Time x D]
                # skip the last time step (the one with EOS as input)
                out = out[:, :-1, :]
                # assumption:  y_seqs has EOS, and not BOS
                loss = self.loss_func(out, seqs, num_toks, True)
                unsaved_state = True
                self.tbd.add_scalars('training', {'step_loss': loss,
                                                  'learn_rate': self.opt.curr_lr},
                                     self.opt.curr_step)

                progress_msg, is_check_pt = train_state.step(num_toks, loss)
                progress_msg += f', LR={self.opt.curr_lr:g}'

                data_bar.set_postfix_str(progress_msg, refresh=False)
                del batch

                if is_check_pt:
                    train_loss = train_state.reset()
                    train_state.train_mode(False)
                    val_loss = self.run_valid_epoch(val_data)
                    self.make_check_point(train_loss, val_loss, keep_models=keep_models)
                    if check_pt_callback:
                        check_pt_callback(model=self.model,
                                          step=self.opt.curr_step,
                                          train_loss=train_loss)
                    train_state.train_mode(True)
                    unsaved_state = False

        if unsaved_state:
            # End of training
            train_loss = train_state.reset()
            train_state.train_mode(False)
            val_loss = self.run_valid_epoch(val_data)
            self.make_check_point(train_loss, val_loss, keep_models=keep_models)
