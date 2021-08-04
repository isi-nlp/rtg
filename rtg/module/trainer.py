#!/usr/bin/env python
#
# Author: Thamme Gowda [tg at isi dot edu] 
# Created: 10/17/18
import torch
import rtg
from rtg import log, yaml, TranslationExperiment as Experiment, device, BatchIterable
from rtg.module import NMTModel
from rtg.utils import IO
from rtg.module import criterion as criteria

from abc import abstractmethod
from typing import Optional, Callable, List
from dataclasses import dataclass, field
import time

from torch import optim
from torch.optim.optimizer import Optimizer
from torch.utils.tensorboard import SummaryWriter
from enum import Enum
import inspect
from pathlib import Path
from rtg.distrib import DistribTorch



dtorch = DistribTorch.instance()



class NoamOpt(Optimizer):
    """
    Optimizer wrapper that implements learning rate as a function of step.

    If inv_sqrt==True:
    - Linear warmup followed by inverse sqrt decay.
    - Uses learning rate in conf.yml as maximum learning rate after warmup

        Modeled after FairSeq's Inverse Square Root LR Scheduler:
            https://github.com/pytorch/fairseq/blob/master/fairseq/optim/lr_scheduler/
                inverse_square_root_schedule.py

    Else:
    - Independent of learning rate set in conf.yml

        Modeled after The Annotated Transformer's LR Scheduler:
            https://nlp.seas.harvard.edu/2018/04/03/attention.html
    """

    def __init__(self, model_size, factor, warmup, optimizer: Optimizer, step=0, inv_sqrt=False):
        super().__init__(params=optimizer.param_groups, defaults=dict(warmup=warmup, step=step))
        self.optimizer = optimizer
        self._step = step
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size

        self.inv_sqrt = inv_sqrt
        lr = optimizer.defaults['lr']
        self.warmup_rate = lr / warmup
        self.decay_factor = lr * warmup ** 0.5

        self._rate = 0
        log.info(f"model_size={model_size}, factor={factor}, warmup={warmup}, step={step}, "
                 f"inv_sqrt={inv_sqrt}")

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
        if self.inv_sqrt:
            if step < self.warmup:
                lr = self.warmup_rate * step
            else:
                lr = self.decay_factor * step ** (-0.5)
        else:
            lr = self.factor * self.model_size ** (-0.5) * min(step ** (-0.5),
                                                               step * self.warmup ** (-1.5))
        return lr

    @staticmethod
    def get_std_opt(model):
        return NoamOpt(model.src_embed[0].d_model, 2, 4000,
                       torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))


class Optims(Enum):
    ADAM = optim.Adam
    ADAMW = optim.AdamW
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
    unit:str = 'tok'

    def running_loss(self):
        return self.total_loss / self.steps if self.steps > 0 else float('inf')

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
        return f'Loss:{self.running_loss():.4f},' \
               f' {int(self.total_toks / elapsed)}{self.unit}/s'

    def is_check_point(self):
        return self.steps == self.check_point


@dataclass
class EarlyStopper:
    """
    A data model to track early stopping state
    """
    enabled: bool = True
    by: str = 'loss'
    patience: int = 15
    min_steps: int = 0
    cur_step: int = 0
    signi_round: int = 4   # integer either positive or negative
    # these many digits are significant round(100, -1) => 30.0  round(100, 1) => 33.3
    measures: List[float] = field(default_factory=list)  # could be loss or accuracy

    buf = 3  # take average of these many points; avoids weird dips and surges as stop
    minimizing = True  # minimize loss, maximize accuracy

    def __post_init__(self):
        if self.enabled:
            assert self.patience > 0, f'early_stop.patience > 0 ? given={self.patience}'
            assert 1 <= self.buf <= self.patience
            log.info(f"Early Stop Enabled;")

        if self.by in {'loss'}:
            self.minimizing = True
        elif self.by in {'bleu', 'accuracy'}:
            self.minimizing = False  # maximizing
        else:
            raise Exception(f'{self.by} is not supported')

    def step(self):
        self.cur_step += 1
        return self.cur_step

    def validation(self, val):
        self.measures.append(val)

    def is_stop(self):
        if not self.enabled:
            return False
        if self.cur_step < self.min_steps:
            # hasn't reached minimum steps; dont stop
            return False
        if len(self.measures) < (self.patience + self.buf + 1):
            # hasn't accumulated enough data points, dont stop
            return False

        # The old value; with some buffer around to avoid weird dips and surges
        old = (self.measures[-self.patience - self.buf: -self.patience])
        old = sum(old) / len(old)  # mean
        recent = self.measures[-self.patience:]  # the patience of seeing the post mark

        if self.minimizing:
            # older value is smaller than or same as best of recent => time to stop
            should_stop = round(old, self.signi_round) <= round(min(recent), self.signi_round)
        else:
            # older value is bigger than or same as best of recent => time to stop
            should_stop = round(old, self.signi_round) >= round(max(recent), self.signi_round)
        return should_stop


class NoOpSummaryWriter(SummaryWriter):
    """
    A No-Op TensorBordX for tests and such experiments that doesnt want to leave
    footprints on file system.
    Note: that this does not extend all methods of SummaryWriter
    """

    def __init__(self, *args, **kwargs):
        #super().__init__(*args, **kwargs)
        # super will create dirs, which we dont want
        pass

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
    default_optim_args = {
        'lr': 0.01,
        'betas': [0.9, 0.98],
        'eps': 1e-9,
        'amsgrad': False,
        'weight_decay': 0,
        'criterion': 'smooth_kld',
        'label_smoothing': 0.1,
        'warmup_steps': 8000,
        'inv_sqrt': False,
        'constant': 2
    }

    def __init__(self, exp: Experiment,
                 model: Optional[NMTModel] = None,
                 model_factory: Optional[Callable] = None,
                 optim: str = 'ADAM',
                 **optim_args):
        self.last_step = -1
        self.exp = exp
        optim_state = None
        if model:
            self.model = model
        else:
            args = exp.model_args
            assert args
            assert model_factory
            self.model, args = model_factory(exp=exp, **args)
            exp.model_args = args
            last_model, self.last_step = self.exp.get_last_saved_model()
            if last_model:
                log.info(f"Resuming training from step:{self.last_step}, model={last_model}")
                state = torch.load(last_model, map_location=device)  
                model_state = state['model_state'] if 'model_state' in state else state

                if 'optim_state' in state:
                    optim_state = state['optim_state']
                self.model.load_state_dict(model_state)
                if 'amp_state' in state and dtorch.fp16:
                    log.info("Restoring  AMP state")
                    dtorch._scaler.load_state_dict(state['amp_state'])
            else:
                log.info("No earlier check point found. Looks like this is a fresh start")

        # optimizer : default args for missing fields
        for k, v in self.default_optim_args.items():
            optim_args[k] = optim_args.get(k, v)

        self.n_gpus = torch.cuda.device_count()
        self.device_ids = list(range(self.n_gpus))

        inner_opt_args = {k: optim_args[k] for k in
                          ['lr', 'betas', 'eps', 'weight_decay', 'amsgrad']}

        self.core_model = self.model.to(device)

        
        trainable_params = self.exp.config['optim'].get('trainable', {})
        if trainable_params:
            if dtorch.is_distributed: # model is wrapped in DP or DistributedDP
                log.warning(f">> Using more than 1 GPU with 'trainable' params is NOT tested")
            trainable_params = self.core_model.get_trainable_params(
                include=trainable_params.get('include'), exclude=trainable_params.get('exclude'))
        else:
            trainable_params = self.model.parameters()

        inner_opt = Optims[optim].new(trainable_params, **inner_opt_args)
        self.model = dtorch.maybe_distributed(self.core_model)
        
        if optim_state:
            log.info("restoring optimizer state from checkpoint")
            try:
                inner_opt.load_state_dict(optim_state)  
            except Exception:
                log.exception("Unable to restore optimizer, skipping it.")
        self.opt = NoamOpt(self.core_model.model_dim, optim_args['constant'], optim_args['warmup_steps'],
                           inner_opt, step=self.start_step, inv_sqrt=optim_args['inv_sqrt'])

        if self.exp.read_only:
            self.tbd = NoOpSummaryWriter()
        else:
            self.tbd = SummaryWriter(log_dir=str(exp.work_dir / 'tensorboard' ))

        self.exp.optim_args = optim, optim_args
        if not self.exp.read_only:
            self.exp.persist_state()
        self.samples = None
        if exp.samples_file and exp.samples_file.exists():
            with IO.reader(exp.samples_file) as f:
                self.samples = [line.strip().split('\t') for line in f]
                log.info(f"Found {len(self.samples)} sample records")
                if self.start_step == 0:
                    for samp_num, sample in enumerate(self.samples):
                        self.tbd.add_text(f"sample/{samp_num}", " || ".join(sample), 0)

            from rtg.module.decoder import Decoder
            self.decoder = Decoder.new(self.exp, self.core_model)

        if self.start_step <= 1:
            self.maybe_init_model()

        self.criterion = self.create_criterion(optim_args['criterion'])

    @property
    def start_step(self):
        _, step = self.exp.get_last_saved_model()
        if self.exp._trained_flag.exists():
            # noinspection PyBroadException
            try:
                step = max(step, yaml.load(self.exp._trained_flag.read_text())['steps'])
            except Exception as _:
                pass
        assert step >= 0
        return step

    def create_criterion(self, criterion):
        log.info(f"Criterion = {criterion}")

        optim_args = self.exp.optim_args[1]
        smoothing = optim_args.get('label_smoothing', 0.0)
        margin = optim_args.get('margin', 0.0)
        mode = optim_args.get('mode', 'dot')
        neg_sampling = optim_args.get('neg_sampling', 'random')
        neg_region = optim_args.get('neg_region', 0.05)
        alpha = optim_args.get('alpha', 1.0)

        pad_idx = self.exp.tgt_vocab.pad_idx
        if criterion == 'smooth_kld':
            return criteria.SmoothKLD(vocab_size=self.core_model.vocab_size, smoothing=smoothing,
                                      pad_idx=pad_idx)
        elif criterion == 'cross_entropy':
            return criteria.CrossEntropy(pad_idx=pad_idx)
        elif criterion == 'binary_cross_entropy':
            return criteria.BinaryCrossEntropy(smoothing=smoothing, pad_idx=pad_idx)
        elif criterion == 'triplet_loss':
            tgt_embedding = self.core_model.tgt_embed[0].lut
            return criteria.TripletLoss(embedding=tgt_embedding, margin=margin,
                                        neg_region=neg_region,
                                        mode=mode, neg_sampling=neg_sampling, pad_idx=pad_idx)
        elif criterion == 'smooth_kld_and_triplet_loss':
            tgt_embedding = self.core_model.tgt_embed[0].lut
            return criteria.SmoothKLDAndTripletLoss(
                embedding=tgt_embedding, margin=margin, neg_region=neg_region, mode=mode,
                neg_sampling=neg_sampling, smoothing=smoothing, alpha=alpha, pad_idx=pad_idx)
        else:
            raise Exception(f'criterion={criterion} is not supported')

    def maybe_init_model(self):
        def load_matrix(path: Path):
            return torch.load(path) if path.exists() else None

        src_emb_mat = load_matrix(self.exp.emb_src_file)
        if src_emb_mat is None:
            log.info("NOT initializing pre-trained source embedding")
        else:
            self.core_model.init_src_embedding(src_emb_mat)

        tgt_emb_mat = load_matrix(self.exp.emb_tgt_file)
        if tgt_emb_mat is None:
            log.info("NOT Initializing pre-trained target embeddings")
        else:
            self.core_model.init_tgt_embedding(tgt_emb_mat)
        self.core_model.maybe_init_from_parent(exp=self.exp)

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

    def make_check_point(self, train_loss: float, val_loss: float, keep_models: int,
                         log_embedding=False):
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
        log.info(f"Checkpoint at optimizer step {step_num}. Training Loss {train_loss:g},"
                 f" Validation Loss:{val_loss:g}")
        self.show_samples()

        self.tbd.add_scalars(f'losses', {'train_loss': train_loss,
                                         'valid_loss': val_loss}, step_num)
        if log_embedding:
            # TODO: add metadata (text) of each subword
            # TODO: Update tag to include tie configuration
            self.tbd.add_embedding(self.model.generator.proj.weight,
                                   global_step=step_num, tag=f'Target embeddings')

        # Unwrap model state from DataParallel and persist
        model = (self.model.module if hasattr(self.model, 'module') else self.model)
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
        if dtorch.fp16:
            state['amp_state'] = dtorch._scaler.state_dict()

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
