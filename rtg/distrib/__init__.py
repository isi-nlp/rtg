#!/usr/bin/env python
#
# Author: Thamme Gowda [tg (at) isi (dot) edu] 
# Created: 7/10/20
import os
import socket
from dataclasses import dataclass
from typing import ClassVar
from torch import nn
from torch.optim.optimizer import Optimizer
from torch.cuda.amp import GradScaler
import torch.distributed as dist

import torch
from torch import nn

from rtg import log

get_env = os.environ.get


@dataclass()
class DistribTorch:

    host_name: str = socket.gethostname()
    pid: int = os.getpid()
    global_rank: int = int(get_env('RANK', '-1'))
    local_rank: int = int(get_env('LOCAL_RANK', '-1'))
    world_size: int = int(get_env('WORLD_SIZE', '-1'))
    master_addr: str = get_env('MASTER_ADDR', '')
    master_port: int = int(get_env('MASTER_PORT', '-1'))

    gpu_count: int = torch.cuda.device_count()
    visible_devices: str = get_env('CUDA_VISIBLE_DEVICES', '')
    max_norm = 10
    fp16 = False  # Manually enable by calling enable_fp16()

    _scaler = None
    _is_backend_ready = False
    # singleton instance; lazy initialization
    _instance: ClassVar['DistribTorch'] = None
    _model: nn.Module = None

    def setup(self):
        log.info("DistribTorch setup()")
        if self.world_size > 1:
            assert self.global_rank >= 0
            assert self.local_rank >= 0
            assert self.master_addr
            assert self.master_port > 1024
            backend = 'nccl' if self.gpu_count > 0 else 'gloo'
            log.info(f"Initializing PyTorch distributed with '{backend}' backend:\n {self}")
            torch.distributed.init_process_group(init_method='env://', backend=backend)
            self._is_backend_ready = True
        return self

    def enable_fp16(self):
        if not self.fp16:   # conditional import
            self.fp16 = True
            self._scaler = GradScaler(enabled=self.fp16)
            log.info("Enabling FP16  /Automatic Mixed Precision training")
        else:
            log.warning(" fp16 is already enabled")
            
    def close(self):
        if self._is_backend_ready:
            log.warning("destroying distributed backend")
            torch.distributed.destroy_process_group()
            self._is_backend_ready = False

    @classmethod
    def instance(cls) -> 'DistribTorch':
        """
        :return: gets singleton instance of class, lazily initialized
        """
        if not cls._instance:
            cls._instance = cls()
        return cls._instance

    def maybe_distributed(self, module: nn.Module):
        if self.world_size > 1:
            if not self._is_backend_ready:
                self.setup()
            self._model = module
            #return torch.nn.parallel.DistributedDataParallel(module)
        return module    # don't wrap

    @property
    def is_distributed(self):
        return self.world_size > 1

    @property
    def is_global_main(self) -> bool:
        return self.global_rank <= 0

    @property
    def is_local_main(self) -> bool:
        return self.local_rank <= 0

    def barrier(self):
        if self.is_distributed:
            torch.distributed.barrier()
        # else we dont need it

    def backward(self, loss):
        if torch.isnan(loss):
            log.warning('loss is nan; backward() skipped')
            return
        if self.fp16:
            loss = self._scaler.scale(loss)
            # to apply norm: TODO: unscale gradients ; refer to docs
            # torch.nn.utils.clip_grad_norm_(self._amp.master_params(opt.optimizer), self.max_norm)
        loss.backward()

    def average_gradients(self, model):
        size = float(self.world_size)
        for param in model.parameters():
            dist.all_reduce(param.grad.data, op=dist.reduce_op.SUM)
            # TODO: ring reduce https://pytorch.org/tutorials/intermediate/dist_tuto.html#our-own-ring-allreduce
            param.grad.data /= size

    def step(self, optimizer: Optimizer):
        if self.is_distributed:
            self.average_gradients(self._model)
            #TODO: Maybe we dont need to average every step ?
        if self.fp16:
            self._scaler.step(optimizer)
            self._scaler.update()
        else:
            optimizer.step()
        optimizer.zero_grad()
