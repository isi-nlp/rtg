__version__ = '0.5.2'

import os
import logging
from rtg.tool.log import Logger

debug_mode = os.environ.get('NMT_DEBUG', False)
log = Logger(console_level=logging.DEBUG if debug_mode else logging.INFO)

import torch
import multiprocessing as mp

device_name = 'cuda:0' if torch.cuda.is_available() else 'cpu'
device = torch.device(device_name)
cpu_device = torch.device('cpu')
cpu_count = int(os.environ.get('RTG_CPUS', str(max(1, mp.cpu_count() - 2))))

from ruamel.yaml import YAML

yaml = YAML()

log.debug(f'device: {device}')
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


from rtg.data.dataset import BatchIterable, Batch
from rtg.exp import TranslationExperiment
from rtg.module import tfmnmt, decoder
from pathlib import Path

RTG_PATH = Path(__file__).resolve().parent.parent

log.info(f"rtg v{__version__} from {RTG_PATH}")
