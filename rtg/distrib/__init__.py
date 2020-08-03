#!/usr/bin/env python
#
# Author: Thamme Gowda [tg (at) isi (dot) edu] 
# Created: 7/10/20
import os
import socket
from dataclasses import dataclass
from typing import ClassVar

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

    _is_backend_ready = False
    # singleton instance; lazy initialization
    _instance: ClassVar['DistribTorch'] =  None

    def setup(self):
        if self.world_size > 1:
            assert self.global_rank >= 0
            assert self.local_rank >= 0
            assert self.master_addr
            assert self.master_port > 1024
            backend = 'nccl' if self.gpu_count > 0 else 'gloo'
            log.info(f"Initializing PyTorch distributed with '{backend}' backend:\n {self}")
            torch.distributed.init_process_group(init_method='env://', backend=backend)
            self._is_backend_ready = True

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
            return nn.parallel.DistributedDataParallel(module, broadcast_buffers=True)
        return module # dont wrap

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