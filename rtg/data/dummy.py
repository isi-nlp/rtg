#!/usr/bin/env python
# Generates dummy data for testing
from rtg.data.dataset import Batch, Example
import argparse
from rtg import log, TranslationExperiment as Experiment
from rtg.utils import IO
from rtg.data.dataset import LoopingIterable

import numpy as np
from pathlib import Path
from typing import Optional, Dict, Union, Any


class BatchIterable:
    # TODO: How to specify Type Hint for this as Iterable[Batch]
    """Dummy equivalent of dataprep.BatchIterable"""

    def __init__(self, vocab_size, batch_size, n_batches, min_seq_len=5, max_seq_len=20,
                 n_reserved_toks=Batch.eos_val + 1, reverse=True, batch_first=False):
        """
         "Generate random data for a src-tgt copy task."
         :param vocab_size: Vocabulary size
         :param batch_size:
         :param n_batches: number of batches to produce
         :param n_reserved_toks:  number of reserved tokens (such as pad, EOS, BOS, UNK etc)
         :param reverse: reverse the target
         :param batch_first: first dimension is batch
         :return:
         """

        self.vocab_size = vocab_size
        self.batch_size = batch_size
        self.num_batches = n_batches
        self.min_seq_len = min_seq_len
        self.max_seq_len = max_seq_len
        self.n_reserved_toks = n_reserved_toks
        self.reverse = reverse
        self.batch_first = batch_first

    def make_an_ex(self):
        seq_len = np.random.randint(self.min_seq_len, self.max_seq_len)
        data = np.random.randint(self.n_reserved_toks, self.vocab_size, size=(seq_len,))
        tgt = self.vocab_size + (self.n_reserved_toks - 1) - data if self.reverse else data
        return Example(data.tolist(), tgt.tolist())

    def __iter__(self):
        for i in range(self.num_batches):
            exs = [self.make_an_ex() for _ in range(self.batch_size)]
            yield Batch(exs, sort_dec=True, batch_first=self.batch_first)


class DummyExperiment(Experiment):
    """
    A dummy experiment for testing;
    this produces random data and leaves no trace on disk
    """

    def __init__(self, work_dir: Union[str, Path], read_only=True,
                 config: Optional[Dict[str, Any]] = None, vocab_size: int = 20,
                 train_batches=30, val_batches=5):
        super().__init__(work_dir, read_only, config)
        self.vocab_size = vocab_size
        self.train_batches = train_batches
        self.val_batches = val_batches

    def get_train_data(self, batch_size: int, steps: int = 0, sort_desc=True, sort_by='random',
                       batch_first=True, shuffle=False, copy_xy=False, fine_tune=False):
        train_data = BatchIterable(self.vocab_size, batch_size, self.train_batches,
                                   reverse=False, batch_first=batch_first)
        if steps > 0:
            train_data = LoopingIterable(train_data, steps)
        return train_data

    def get_val_data(self, batch_size: int, sort_desc=True, batch_first=True,
                     shuffle=False, copy_xy=False):
        assert not shuffle, 'Not supported'
        assert not copy_xy, 'Not supported'
        val_data = BatchIterable(self.vocab_size, batch_size, self.val_batches,
                                 reverse=False, batch_first=batch_first)
        return val_data


def parse_args():
    p = argparse.ArgumentParser(description='Generates random data for testing',
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('exp', type=Path, help="path to experiment directory")
    p.add_argument('-s', '--seed', type=int, default=0,
                   help='seed for reproducing the randomness. 0 is no seed.')
    p.add_argument('-mn', '--min-len', type=int, default=4, help='Minimum length sequence')
    p.add_argument('-mx', '--max-len', type=int, default=15, help='Maximum length sequence')
    p.add_argument('-v', '--vocab', dest='vocab_size', type=int, default=200,
                   help='Vocabulary size')
    p.add_argument('-r', '--reserved', dest='num_reserved',
                   type=int, default=4, help='Reserved tokens')
    p.add_argument('-nt', '--num-train', type=int, default=1000, help='Number of train sequences')
    p.add_argument('-nv', '--num-val', type=int, default=500, help='Number of validation seqs')
    p.add_argument('--rev-vocab', action="store_true",
                   help="Reverse the target side vocabulary")
    p.add_argument('--rev-seq', action="store_true",
                   help="Reverse the target side sequence order")
    return vars(p.parse_args())


def generate_parallel(min_len, max_len, vocab_size, num_reserved, num_exs, rev_vocab, rev_seq):
    lower, higher = num_reserved, vocab_size

    for _ in range(num_exs):
        _len = np.random.randint(min_len, max_len)
        src_seq = np.random.randint(lower, higher, size=_len)
        tgt_seq = src_seq.copy()
        if rev_vocab:
            tgt_seq = higher + num_reserved - src_seq
        if rev_seq:
            tgt_seq = np.flip(tgt_seq, axis=0)
        yield src_seq, tgt_seq


def write_tsv(data, out):
    count = 0
    for src_seq, tgt_seq in data:
        src_seq, tgt_seq = ' '.join(map(str, src_seq)), ' '.join(map(str, tgt_seq))
        out.write(f'{src_seq}\t{tgt_seq}\n')
        count += 1
    log.info(f"Wrote {count} records")


def write_parallel(data, src_file, tgt_file):
    count = 0
    with IO.writer(src_file) as src_f, IO.writer(tgt_file) as tgt_f:
        for src_seq, tgt_seq in data:
            src_seq = ' '.join(map(str, src_seq))
            tgt_seq = ' '.join(map(str, tgt_seq))
            src_f.write(f'{src_seq}\n')
            tgt_f.write(f'{tgt_seq}\n')
            count += 1
    log.info(f"Wrote {count} records to {src_file} and {tgt_file}")


def main(args):
    work_dir: Path = args.pop('exp')

    work_dir.mkdir(exist_ok=True, parents=True)
    log.info(f"Setting up a dummy experiment at {work_dir}")
    num_train, num_val = args.pop('num_train'), args.pop('num_val')

    train_data = generate_parallel(**args, num_exs=num_train)
    val_data = generate_parallel(**args, num_exs=num_val)

    train_files = str(work_dir / 'train.raw.src'), str(work_dir / 'train.raw.tgt')
    val_files = str(work_dir / 'valid.raw.src'), str(work_dir / 'valid.raw.tgt')
    write_parallel(train_data, *train_files)
    write_parallel(val_data, *val_files)

    config = {
        'prep': {
            'train_src': train_files[0],
            'train_tgt': train_files[1],
            'valid_src': val_files[0],
            'valid_tgt': val_files[1],
            'pieces': 'word',
            'truncate': True,
            'src_len': args['max_len'],
            'tgt_len': args['max_len'],
        }}

    if args.get('rev_vocab'):
        # shared vocabulary would be confusing
        config['prep'].update({
            'shared_vocab': False,
            'max_src_types': args['vocab_size'],
            'max_tgt_types': args['vocab_size']
        })
    else:
        config['prep'].update({
            'shared_vocab': True,
            'max_types': args['vocab_size']
        })
    exp = Experiment(work_dir, config=config)
    exp.store_config()


if __name__ == '__main__':
    args = parse_args()
    log.info(f"Args {args}")
    seed = args.pop('seed')
    if seed:
        np.random.seed(seed)
    main(args)
