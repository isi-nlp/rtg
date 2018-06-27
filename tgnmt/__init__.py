import os
import logging as log
debug_mode = os.environ.get('NMT_DEBUG', False)
log.basicConfig(level=log.DEBUG if debug_mode else log.INFO)
log.debug(f"NMT_DEBUG={debug_mode}")

import torch
device_name = 'cuda:0' if torch.cuda.is_available() else 'cpu'
device = torch.device(device_name)
log.debug(f'device: {device}')


def my_tensor(*args, **kwargs):
    return torch.tensor(*args, device=device, **kwargs)


from .dataprep import Field, BatchIterable, Batch
from tgnmt.exp import TranslationExperiment
from tgnmt.module import seq2seq