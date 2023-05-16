__version__ = '0.7.3-dev'


import os
import logging
from pathlib import Path
import multiprocessing as mp


debug_mode = os.environ.get('NMT_DEBUG', False)

import torch

device_name = 'cuda:0' if torch.cuda.is_available() else 'cpu'
device = torch.device(device_name)
cpu_device = torch.device('cpu')
cpu_count = int(os.environ.get('RTG_CPUS', str(max(1, mp.cpu_count() - 2))))
RTG_PATH = Path(__file__).resolve().parent.parent

from rtg.log import Logger
from ruamel.yaml import YAML

log = Logger(console_level=logging.DEBUG if debug_mode else logging.INFO)
yaml = YAML()

log.info(f"rtg v{__version__} from {RTG_PATH}; default device: {device}")
profiler = None

if os.environ.get('NMT_PROFILER') == 'memory':
    import memory_profiler

    profiler = memory_profiler.profile
    log.info('Setting memory profiler')


def my_tensor(*args, **kwargs):
    return torch.tensor(*args, device=device, **kwargs)


def profile(func, *args):
    """
    :param func: function to profile
    :param args: any addtional args for profiler
    :return:
    """
    if not profiler:
        return func
    return profiler(func, *args)


from rtg.utils import *
from rtg.registry import *
from rtg.common import *
from rtg.data import *

# avoid name space collision
from . import (
    nmt,
    lm,
    classifier,
    comet,
)
