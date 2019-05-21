#!/usr/bin/env python
#
# Author: Thamme Gowda [tg at isi dot edu] 
# Created: 10/17/18
import torch
import torch.nn as nn
import rtg
from rtg import log, TranslationExperiment as Experiment, device, BatchIterable
from rtg.module import NMTModel
from rtg.utils import IO

from abc import abstractmethod
from typing import Optional, Callable
from dataclasses import dataclass
import time
from tensorboardX import SummaryWriter

from torch import optim
from torch.optim import Optimizer
from enum import Enum
import inspect
from pathlib import Path


class NoamOpt(Optimizer):
    """
    Optimizer wrapper that implements learning rate as a function of step.
    """

    def __init__(self, model_size, factor, warmup, optimizer: Optimizer, step=0):
        super().__init__(params=optimizer.param_groups, defaults=dict(warmup=warmup, step=step))
        self.optimizer = optimizer
        self._step = step
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0
        log.info(f"model_size={model_size}, factor={factor}, warmup={warmup}, step={step}")

    def step(self, closure=None):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step(closure=closure)

    @property
    def curr_step(self):
        return self._step

    @property
    def curr_lr(self):
        return self._rate

    def zero_grad(self):
        self.optimizer.zero_grad()

    def rate(self, step=None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.factor * (
                self.model_size ** (-0.5) * min(step ** (-0.5), step * self.warmup ** (-1.5)))

    @staticmethod
    def get_std_opt(model):
        return NoamOpt(model.src_embed[0].d_model, 2, 4000,
                       torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))


class Optims(Enum):
    ADAM = optim.Adam
    SGD = optim.SGD

    def new(self, parameters, lr=0.001, **args):
        log.info(f"Creating {self.value} optimizer with lr={lr} and extra args:{args}")
        log.info(f"   {self.value}, default arguments {inspect.signature(self.value)}")
        return self.value(parameters, lr=lr, **args)

    @staticmethod
    def names():
        return list(Optims.__members__.keys())


@dataclass
class TrainerState:
    """
    A dataclass for storing any running stats the trainer needs to keep track during training
    """

    model: NMTModel
    check_point: int
    total_toks: int = 0
    total_loss: float = 0.0
    steps: int = 0
    start: float = time.time()

    def running_loss(self):
        return self.total_loss / self.total_toks if self.total_toks != 0 else float('inf')

    def reset(self):
        loss = self.running_loss()
        self.total_toks = 0
        self.total_loss = 0.0
        self.steps = 0
        self.start = time.time()
        return loss

    def train_mode(self, mode: bool):
        torch.set_grad_enabled(mode)
        self.model.train(mode)

    def step(self, toks, loss):
        self.steps += 1
        self.total_toks += toks
        self.total_loss += loss
        return self.progress_bar_msg(), self.is_check_point()

    def progress_bar_msg(self):
        elapsed = time.time() - self.start
        return f'Loss:{self.total_loss / self.total_toks:.4f},' \
            f' {int(self.total_toks / elapsed)}toks/s'

    def is_check_point(self):
        return self.steps == self.check_point


class NoOpSummaryWriter(SummaryWriter):
    """
    A No-Op TensorBordX for tests and such experiments that doesnt want to leave
    footprints on file system.
    Note: that this does not extend all methods of SummaryWriter
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def add_text(self, *args, **kwargs):
        pass

    def add_scalar(self, *args, **kwargs):
        pass

    def add_scalars(self, *args, **kwargs):
        pass

    def add_embedding(self, *args, **kwargs):
        pass


class SteppedTrainer:
    """
    A base class for Trainers that use step based training (not epoch based training)
    """

    def __init__(self, exp: Experiment,
                 model: Optional[NMTModel] = None,
                 model_factory: Optional[Callable] = None,
                 optim: str = 'ADAM',
                 **optim_args):
        self.start_step = 0
        self.last_step = -1
        self.exp = exp
        optim_state = None
        if model:
            self.model = model
        else:
            args = exp.model_args
            assert args
            assert model_factory
            log.info(f"Creating model with args: {args}")
            self.model, args = model_factory(exp=exp, **args)
            exp.model_args = args
            last_model, self.last_step = self.exp.get_last_saved_model()
            if last_model:
                self.start_step = self.last_step + 1
                log.info(f"Resuming training from step:{self.start_step}, model={last_model}")
                state = torch.load(last_model)
                model_state = state['model_state'] if 'model_state' in state else state
                if 'optim_state' in state:
                    optim_state = state['optim_state']
                self.model.load_state_dict(model_state)
            else:
                log.info("No earlier check point found. Looks like this is a fresh start")

        # making optimizer
        optim_args['lr'] = optim_args.get('lr', 0.1)
        optim_args['betas'] = optim_args.get('betas', [0.9, 0.98])
        optim_args['eps'] = optim_args.get('eps', 1e-9)

        warm_up_steps = optim_args.pop('warmup_steps', 8000)
        self._smoothing = optim_args.pop('label_smoothing', 0.1)
        noam_factor = 2

        self.model = self.model.to(device)

        inner_opt = Optims[optim].new(self.model.parameters(), **optim_args)
        if optim_state:
            log.info("restoring optimizer state from checkpoint")
            inner_opt.load_state_dict(optim_state)
        self.opt = NoamOpt(self.model.model_dim, noam_factor, warm_up_steps, inner_opt,
                           step=self.start_step)

        optim_args['warmup_steps'] = warm_up_steps
        optim_args['label_smoothing'] = self._smoothing
        if self.exp.read_only:
            self.tbd = NoOpSummaryWriter()
        else:
            self.tbd = SummaryWriter(log_dir=str(exp.work_dir / 'tensorboard'))

        self.exp.optim_args = optim, optim_args
        if not self.exp.read_only:
            self.exp.persist_state()
        self.samples = None
        if exp.samples_file.exists():
            with IO.reader(exp.samples_file) as f:
                self.samples = [line.strip().split('\t') for line in f]
                log.info(f"Found {len(self.samples)} sample records")
                if self.start_step == 0:
                    for samp_num, sample in enumerate(self.samples):
                        self.tbd.add_text(f"sample/{samp_num}", " || ".join(sample), 0)

            from rtg.module.decoder import Decoder
            self.decoder = Decoder.new(self.exp, self.model)

        if self.start_step == 0:
            self.init_embeddings()
        self.model = self.model.to(device)


    def init_embeddings(self):
        def load_matrix(path: Path):
            return torch.load(path) if path.exists() else None

        src_emb_mat = load_matrix(self.exp.emb_src_file)
        if src_emb_mat is None:
            log.info("NOT initializing pre-trained source embedding")
        else:
            self.model.init_src_embedding(src_emb_mat)

        tgt_emb_mat = load_matrix(self.exp.emb_tgt_file)
        if tgt_emb_mat is None:
            log.info("NOT Initializing pre-trained target embeddings")
        else:
            self.model.init_tgt_embedding(tgt_emb_mat)

    def show_samples(self, beam_size=3, num_hyp=3, max_len=30):
        """
        Logs the output of model (at this stage in training) to a set of samples
        :param beam_size: beam size
        :param num_hyp: number of hypothesis to output
        :param max_len: maximum length to decode
        :return:
        """
        if not self.samples:
            log.info("No samples are chosen by the experiment")
            return
        for i, (line, ref) in enumerate(self.samples):
            step_num = self.opt.curr_step
            result = self.decoder.decode_sentence(line, beam_size=beam_size, num_hyp=num_hyp,
                                                  max_len=max_len)
            outs = [f"hyp{j}: {score:.3f} :: {out}" for j, (score, out) in enumerate(result)]
            self.tbd.add_text(f'sample/{i}', " || ".join(outs), step_num)
            outs = '\n'.join(outs)
            log.info(f"==={i}===\nSRC:{line}\nREF:{ref}\n{outs}")

    def make_check_point(self, train_loss: float, val_loss: float, keep_models: int):
        """
        Check point the model
        :param train_loss: training loss value
        :param val_loss: loss on validation set
        :param keep_models: how many checkpoints to keep on file system
        :return:
        """

        step_num = self.opt.curr_step
        if step_num == self.last_step:
            log.warning("Ignoring checkpt request")
            return  # calling multiple times doesnt save
        log.info(f"Checkpoint at step {step_num}. Training Loss {train_loss:g},"
                 f" Validation Loss:{val_loss:g}")
        self.show_samples()

        self.tbd.add_scalars(f'losses', {'train_loss': train_loss,
                                         'valid_loss': val_loss}, step_num)
        # Unwrap model state from DataParallel and persist
        model = (self.model.module if isinstance(self.model, nn.DataParallel) else self.model)
        state = {
            'model_state': model.state_dict(),
            'optim_state': self.opt.optimizer.state_dict(),
            'step': step_num,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'time': time.time(),
            'rtg_version': rtg.__version__,
            'model_type': self.exp.model_type,
            'model_args': self.exp.model_args,
        }

        self.exp.store_model(step_num, state, train_score=train_loss,
                             val_score=val_loss, keep=keep_models)
        self.last_step = step_num

    @abstractmethod
    def run_valid_epoch(self, data_iter: BatchIterable) -> float:
        """
        Run a validation epoch
        :param data_iter: data iterator, either training or validation
        :return: score which is a loss
        """
        raise NotImplementedError()

    @abstractmethod
    def train(self, steps: int, check_point: int, batch_size: int,
              check_pt_callback: Optional[Callable] = None, **args):
        """
        Train the model
        :param steps: number of steps to train
        :param check_point: how often to take check points
        :param batch_size: what is the batch size to use (depends on GPU RAM and model size)
        :param check_pt_callback: the function to call when a check point is taken
        :param args: any extra args
        :return:
        """
        raise NotImplementedError()
