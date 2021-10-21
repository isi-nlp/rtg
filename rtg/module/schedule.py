#!/usr/bin/env python
#
# Author: Thamme Gowda [tg (at) isi (dot) edu] 
# Created: 4/26/21
from dataclasses import dataclass

from rtg import log
from rtg.registry import register, SCHEDULE
from torch import optim


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
        return self.constant * self.model_dim ** -0.5 * min(step ** -0.5,
                                                            step * self.warmup ** -1.5)


@register(SCHEDULE, 'inverse_sqrt')
@dataclass
class InverseSqrt(LRSchedule):
    warmup: int
    peak_lr: float

    def rate(self, step) -> float:
        return min(step * self.peak_lr / self.warmup,
                   self.peak_lr * self.warmup ** 0.5 * step ** -0.5)


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
        return cls(start_step=0,
                   schedule=Noam(warmup=4000, constant=2, model_dim=model_dim),
                   optimizer=optim.Adam(model_params, lr=0, betas=(0.9, 0.98), eps=1e-9))
