#!/usr/bin/env python
#
# Author: Thamme Gowda [tg (at) isi (dot) edu]
# Created: 4/26/21
from dataclasses import dataclass

from torch import optim

from rtg import log
from ..registry import register, SCHEDULE
from .distrib import DistribTorch

dtorch = DistribTorch.instance()

__all__ = ['LRSchedule', 'Noam', 'InverseRoot', 'InverseSqrt', 'ScheduledOptimizer']


@dataclass
class LRSchedule:
    def __call__(self, *args, **kwargs) -> float:
        return self.rate(*args, **kwargs)

    def rate(self, step) -> float:
        raise NotImplementedError()


@register(SCHEDULE, 'noam')
@dataclass
class Noam(LRSchedule):
    warmup: int
    constant: int
    model_dim: int

    def rate(self, step) -> float:
        return self.constant * self.model_dim**-0.5 * min(step**-0.5, step * self.warmup**-1.5)


@register(SCHEDULE, 'inverse_root')
@dataclass
class InverseRoot(LRSchedule):
    # for a visualization, see https://www.desmos.com/calculator/qadmmahb2t

    warmup: int
    peak_lr: float
    init_lr: 0.0
    constant: int = 1
    decay_power: float = 0.5

    def __post_init__(self):
        assert self.init_lr < self.peak_lr, f'init_lr must be lower than peak_lr'
        assert self.constant > 0
        assert 0 < self.decay_power < 1, f'set 0.5 for sqrt, 0.3 for cube root etc. given {self.decay_power}'

    def rate(self, step) -> float:
        return self.constant * min(
            self.init_lr + step * (self.peak_lr - self.init_lr) / self.warmup,
            self.peak_lr * self.warmup**self.decay_power * step**-self.decay_power,
        )


@register(SCHEDULE, 'inverse_sqrt')
@dataclass
class InverseSqrt(LRSchedule):
    warmup: int
    peak_lr: float
    init_lr: 0.0
    constant: int = 1

    def __post_init__(self):
        assert self.init_lr < self.peak_lr, f'init_lr must be lower than peak_lr'
        assert self.constant > 0

    def rate(self, step) -> float:
        if step <= self.warmup:
            lr = self.init_lr + step * (self.peak_lr - self.init_lr) / self.warmup
        else:
            lr = self.peak_lr * self.warmup**0.5 * step**-0.5
        return self.constant * lr


@dataclass
class ScheduledOptimizer:
    start_step: int
    schedule: LRSchedule
    optimizer: optim.Optimizer

    def __post_init__(self):
        self._step = self.start_step
        self._rate = -1
        if self.schedule is None:
            log.warning("Learning rate schedule is not configured; letting optimizer handle itself")

    def step(self, closure=None):
        "Update parameters and rate"
        self._step += 1
        if self.schedule is not None:
            rate = self.schedule.rate(step=self._step)
            # if dtorch.world_size > 1:
            # assumption: batch_size was divided across workers
            # Refer to: https://arxiv.org/pdf/1706.02677.pdf
            # Usually, batch_size is multiplied when more GPUs are added
            #           rate *= dtorch.world_size   #  <-- this is harmful
            # but, in RTG, we stick to same batch_size specified in conf.yml (for reproducibility)
            # and hence divide batch_size across workers, so we divide learning rate instead of multiplying
            # rate /= dtorch.world_size   #<-- this should have been okay, but see update:
            # update: we have scaled the minibatch normalizer instead, so we dont need to mess with learning rate
            for p in self.param_groups:
                p['lr'] = rate
            self._rate = rate
        else:  # extract learning rate from optimizer
            for param_group in self.param_groups:
                self._rate = param_group['lr']
                break
        self.optimizer.step(closure=closure)

    @property
    def param_groups(self):
        return self.optimizer.param_groups

    @property
    def curr_step(self):
        return self._step

    @property
    def curr_lr(self):
        return self._rate

    def zero_grad(self):
        self.optimizer.zero_grad()

    @classmethod
    def get_vaswani_etal_opt(cls, model_params, model_dim=512):
        """The optimizer used in Attention is all you need"""
        return cls(
            start_step=0,
            schedule=Noam(warmup=4000, constant=2, model_dim=model_dim),
            optimizer=optim.Adam(model_params, lr=0, betas=(0.9, 0.98), eps=1e-9),
        )
