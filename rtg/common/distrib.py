#!/usr/bin/env python
#
# Author: Thamme Gowda [tg (at) isi (dot) edu]
# Created: 7/10/20
import os
import socket
from dataclasses import dataclass
from typing import ClassVar

import torch
import torch.distributed as dist
from torch import nn
from torch.cuda.amp import GradScaler
from torch.optim.optimizer import Optimizer

from rtg import log

__all__ = ['dtorch', 'DistribTorch']


class SkipBatchException(Exception):
    """
    This exception is raised when a batch is skipped. e.g. due to nan loss
    """

    pass


@dataclass()
class DistribTorch:
    host_name: str = socket.gethostname()
    pid: int = os.getpid()
    get_env = os.environ.get
    global_rank: int = int(get_env('RANK', '-1'))
    local_rank: int = int(get_env('LOCAL_RANK', '-1'))
    world_size: int = int(get_env('WORLD_SIZE', '-1'))
    master_addr: str = get_env('MASTER_ADDR', '')
    master_port: int = int(get_env('MASTER_PORT', '-1'))

    gpu_count: int = torch.cuda.device_count()
    visible_devices: str = get_env('CUDA_VISIBLE_DEVICES', '')
    max_norm = 10
    fp16 = False  # Manually enable by calling enable_fp16()
    fp16_dtype = torch.float16
    grad_accum = 1  # grad accumulation over these many batches
    max_skips = 5  # max number of skips due to nan loss

    _scaler = None
    _is_backend_ready = False
    # singleton instance; lazy initialization
    _instance: ClassVar['DistribTorch'] = None
    _model: nn.Module = None
    _clip_grad_max_norm = None

    _n_skips = 0

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

    def init_trainer_args(self, args: dict):
        if 'clip_grad_norm' in args:
            self.clip_grad_norm(args['clip_grad_norm'])
        if args.get('grad_accum'):
            self.set_grad_accum(args['grad_accum'])
        if args.get('fp16'):
            self.enable_fp16()

    def enable_fp16(self):
        try:
            if torch.cuda.is_bf16_supported():
                log.info('BFLOAT16 is supported; upgrading...')
                self.fp16_dtype = torch.bfloat16  # if supported
        except:
            log.info('BFLOAT16 is not supported. using FLOAT16')

        if not self.fp16:
            self.fp16 = True
            self._scaler = GradScaler(enabled=self.fp16)
            log.info("Enabling FP16  /Automatic Mixed Precision training")
        else:
            log.warning(" fp16 is already enabled")

    def set_grad_accum(self, interval: int):
        if interval < 1:
            log.warning(f"grad_accum is set to {interval}; updating to 1")
            interval = 1
        self.grad_accum = interval
        log.info(f"Gradient accumulation interval set to {interval}")

    def clip_grad_norm(self, max_norm):
        assert max_norm
        log.info(f"Gradient clipping max_norm={max_norm}")
        self._clip_grad_max_norm = max_norm

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
        return module  # don't wrap

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

    def backward(self, loss, retain_graph=False):
        if self.fp16:
            loss = self._scaler.scale(loss)
        if torch.isnan(loss) or torch.isinf(loss) or loss < 0:  # nan, inf or negative
            err_msg = f"Loss value:{loss} (device: {loss.device}); is_NaN: {torch.isnan(loss)}; is_inf:{torch.isinf(loss)}; is_neg:{loss < 0}; skipping..."
            log.error(err_msg)
            raise SkipBatchException(err_msg)
            # else crash
            raise Exception(
                '''Loss is nan; enable debug mode to know more (export NMT_DEBUG=true);
    Or, here are some tips:
    1. reduce the learning rate
    2. reduce batch size
    3. set trainer.init_args.clip_grad_norm to a small number e.g. 5.0'''
            )

        loss.backward(retain_graph=retain_graph)

    def average_gradients(self, model):
        size = float(self.world_size)
        # dist.all_reduce_coalesced(list(model.parameters()), op=dist.ReduceOp.SUM)  # unavailable
        futures = []

        skipped_params = []
        for name, param in model.named_parameters():
            if param.grad is None:
                skipped_params.append(name)
                continue
            work = dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM, async_op=True)
            futures.append((work, param))
            # TODO: ring reduce https://pytorch.org/tutorials/intermediate/dist_tuto.html#our-own-ring-allreduce
            # param.grad.data /= size
        if skipped_params:
            log.warning_once(
                "Skipped averaging of %d parameters gradients because they dont have gradients: %s",
                len(skipped_params),
                ', '.join(skipped_params),
            )

        for work, param in futures:
            work.wait()  # if not complete
            param.grad.data /= size

    def step(self, optimizer: Optimizer):
        if self.is_distributed:
            self.average_gradients(self._model)
            # TODO: Maybe we dont need to average every step ?

        if self._clip_grad_max_norm:
            if self.fp16:
                # Unscales the gradients of optimizer's assigned params in-place
                self._scaler.unscale_(optimizer)
            # Since the gradients of optimizer's assigned params are unscaled, clips as usual:
            torch.nn.utils.clip_grad_norm_(
                self._model.parameters(), self._clip_grad_max_norm
            )  #  error_if_nonfinite=False

        if self.fp16:
            self._scaler.step(optimizer)
            self._scaler.update()
        else:
            optimizer.step()
        optimizer.zero_grad()

    @property
    def batch_size_scaler(self):
        return max(self.world_size, 1) * max(self.grad_accum, 1)


dtorch = DistribTorch.instance()
