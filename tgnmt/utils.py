import gc
from tgnmt import log
import torch
from functools import reduce
import operator as op


# Size of each element in tensor
tensor_size = {
    'torch.Tensor': 4,
    'torch.FloatTensor': 4,
    'torch.DoubleTensor': 8,
    'torch.HalfTensor': 2,
    'torch.ByteTensor': 1,
    'torch.CharTensor': 1,
    'torch.ShortTensor': 2,
    'torch.IntTensor': 4,
    'torch.LongTensor': 8
}
tensor_size.update({t.replace('torch.', 'torch.cuda.'): size for t, size in tensor_size.items()})


def log_tensor_sizes(writer=log.info, min_size=1024):
    """
    Forces garbage collector and logs all the current tensors
    :return:
    """
    log.info("Collecting tensor allocations")
    gc.collect()

    def is_tensor(obj):
        if torch.is_tensor(obj):
            return True
        try:    # some native objects raise exceptions
            return hasattr(obj, 'data') and torch.is_tensor(obj.data)
        except:
            return False

    tensors = filter(is_tensor, gc.get_objects())
    stats = ((reduce(op.mul, obj.size()) if len(obj.size()) > 0 else 0,
              obj.type(), tuple(obj.size()), hex(id(obj))) for obj in tensors)
    stats = ((n*tensor_size[typ], n, typ, *blah) for n, typ, *blah in stats)
    stats = (x for x in stats if x[0] > min_size)
    sorted_stats = sorted(stats, key=lambda x: x[0])

    writer("####\tApprox Bytes\tItems       \tShape   \tObject ID")
    lines = (f'{i:4}\t{size:12,}\t{n:12,}\t{typ}\t{shape}\t{_id}'
             for i, (size, n, typ, shape, _id) in enumerate(sorted_stats))
    log.info("==== Tensors and memories === ")
    for i, l in enumerate(lines):
        writer(l)

    total = sum(rec[0] for rec in sorted_stats)
    log.info(f'Total Bytes by tensors  bigger than {min_size} is (approx):{total:,}')
