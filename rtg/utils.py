import gc
import gzip
import operator as op
from functools import reduce
from pathlib import Path
import torch
from rtg import log
import inspect
import shutil
from datetime import datetime


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


def line_count(path, ignore_blanks=False):
    """count number of lines in file
    :param path: file path
    :param ignore_blanks: ignore blank lines
    """
    with IO.reader(path) as reader:
        count = 0
        for line in reader:
            if ignore_blanks and not line.strip():
                continue
            count += 1
        return count


def get_my_args(exclusions=None):
    """
    get args of your call. you = a function
    :type exclusions: List of arg names that should be excluded from return dictionary
    :return: dictionary of {arg_name: argv_value} s
    """
    _, _, _, args = inspect.getargvalues(inspect.currentframe().f_back)
    for excl in ['self', 'cls'] + (exclusions or []):
        if excl in  args:
            del args[excl]
    return args


class IO:
    """File opener and automatic closer"""

    def __init__(self, path, mode='r', encoding=None, errors=None):
        self.path = path if type(path) is Path else Path(path)
        self.mode = mode
        self.fd = None
        self.encoding = encoding if encoding else 'utf-8' if 't' in mode else None
        self.errors = errors if errors else 'replace'

    def __enter__(self):

        if self.path.name.endswith(".gz"):   # gzip mode
            self.fd = gzip.open(self.path, self.mode, encoding=self.encoding, errors=self.errors)
        else:
            if 'b' in self.mode:  # binary mode doesnt take encoding or errors
                self.fd = self.path.open(self.mode)
            else:
                self.fd = self.path.open(self.mode, encoding=self.encoding, errors=self.errors)
        return self.fd

    def __exit__(self, _type, value, traceback):
        self.fd.close()

    @classmethod
    def reader(cls, path, text=True):
        return cls(path, 'rt' if text else 'rb')

    @classmethod
    def writer(cls, path, text=True, append=False):
        return cls(path, ('a' if append else 'w') + ('t' if text else 'b'))

    @classmethod
    def get_lines(cls, path, col=0, delim='\t', line_mapper=None):
        with cls.reader(path) as inp:
            if col >= 0:
                inp = (line.split(delim)[col].strip() for line in inp)
            if line_mapper:
                inp = (line_mapper(line) for line in inp)
            yield from inp

    @classmethod
    def write_lines(cls, path: Path, text):
        if isinstance(text, str):
            text = [text]
        with cls.writer(path) as out:
            for line in text:
                out.write(line)
                out.write('\n')

    @classmethod
    def copy_file(cls, src: Path, dest: Path):
        assert src.resolve() != dest.resolve()
        log.info(f"Copy {src} → {dest}")
        with IO.reader(src) as inp, IO.writer(dest) as out:
            shutil.copyfileobj(inp, out)

    @classmethod
    def maybe_backup(cls, file: Path):
        if file.exists():
            time = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
            dest = file.with_suffix(f'.{time}')
            log.info(f"Backup {file} → {dest}")
            file.rename(dest)
