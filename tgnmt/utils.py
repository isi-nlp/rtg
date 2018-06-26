import gc
from tgnmt import log
import torch
from functools import reduce
import operator as op


def log_tensor_sizes(writer=log.info, min_size=1024):
    """
    Forces garbage collector and logs all the current tensors
    :return:
    """
    log.info("Collecting tensor allocations")
    gc.collect()
    tensors = (obj for obj in gc.get_objects()
               if torch.is_tensor(obj)
               # or (hasattr(obj, 'data') and torch.is_tensor(obj.data))
               )
    stats = ((reduce(op.mul, obj.size()) if len(obj.size()) > 0 else 0,
              type(obj), obj.size(), hex(id(obj))) for obj in tensors)
    stats = (x for x in stats if x[0] > min_size)
    sorted_stats = sorted(stats, key=lambda x: x[0])
    lines = (f'{i:4}\t{size:12,}\t{typ}\t{shape}\t{_id}' for i, (size, typ, shape, _id) in enumerate(sorted_stats))
    log.info("==== Tensors and memories === ")
    for i, l in enumerate(lines):
        writer(l)
