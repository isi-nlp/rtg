#!/usr/bin/env python
#
# A script to augment parallel datasets
# Author: Thamme Gowda
# Created: 11/2/21

import argparse
import collections
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, Set, List, Sequence, Tuple
import itertools
import resource
import numpy as np
from tqdm import tqdm
import logging as log


log.basicConfig(level=log.INFO)


def read_raw_parallel_lines(*streams):
    assert len(streams) >= 2
    for lines in itertools.zip_longest(*streams):
        assert all(x is not None for x in lines), 'Input has unequal number of segments; parallel segments expected'
        yield tuple(line.strip() for line in lines)


def max_RSS(who=resource.RUSAGE_SELF) -> Tuple[int, str]:
    """Gets memory usage of current process, maximum so far.
    Maximum so far, since the system call API doesnt provide "current"
    :returns (int, str)
       int is a value from getrusage().ru_maxrss
       str is human friendly value (best attempt to add right units)
    """
    mem = resource.getrusage(who).ru_maxrss
    h_mem = mem
    if 'darwin' in sys.platform:  # "man getrusage 2" says we get bytes
        h_mem /= 10 ** 3  # bytes to kilo
    unit = 'KB'
    if h_mem >= 10 ** 3:
        h_mem /= 10 ** 3  # kilo to mega
        unit = 'MB'
    return mem, f'{int(h_mem):,}{unit}'


@dataclass
class RandomAugment:
    rate: float

    def __post_init__(self):
        assert 0 <= self.rate <= 1

    def __call__(self, *args, **kwargs):
        raise NotImplementedError()


class RandomWordAugment(RandomAugment):
    def __call__(self, toks):
        assert isinstance(toks, list)
        n_augs = int(len(toks) * self.rate)
        if n_augs > 0:
            aug_pos = {random.randrange(0, len(toks)) for _ in range(n_augs)}
            toks = self.augment(toks, aug_pos)
            return toks
        else:
            return None

    def augment(self, toks: List[Any], positions: Set[int]):
        raise NotImplementedError('Not implemented')


class WordDropout(RandomWordAugment):

    def augment(self, toks: List[Any], positions: Set[int]):
        return [x for i, x in enumerate(toks) if i not in positions]


class WordShuffle(RandomWordAugment):

    def augment(self, toks: List[Any], positions: Set[int]):
        from_pos = list(positions)
        to_pos = [random.randrange(0, len(toks)) for _ in positions]
        for x, y in zip(from_pos, to_pos):
            toks[x], toks[y] = toks[y], toks[x]  # swap
        return toks


@dataclass
class WordReplace(RandomWordAugment):
    replacements: Sequence[Any]

    def augment(self, toks: List[Any], positions: Set[int]):
        for pos in positions:
            toks[pos] = random.choices(self.replacements)
        return toks


@dataclass
class Transforms:
    chain: List[RandomWordAugment]

    def __call__(self, toks):
        x = toks
        for it in self.chain:
            if x is None:
                return None
            x = it(x)
        return x


@dataclass
class Augmentor:
    src_in: Path
    tgt_in: Path
    src_out: Path
    tgt_out: Path
    meta_out: Path
    n_inp_recs: int = 0
    n_out_recs: int = 0
    args: Dict[str, Any] = field(default_factory=dict)
    # ##########
    _inp_recs = None
    _src_freqs = None
    _tgt_freqs = None

    def __post_init__(self):
        paths = [self.src_out, self.tgt_out, self.meta_out]
        for p in paths:
            p.parent.mkdir(exist_ok=True)
        self.outs = [open(p, mode='w', encoding='utf-8') for p in paths]

    @property
    def inp_recs(self):
        if not self._inp_recs:
            log.warning(f"Going to buffer data; this may consume all the memory crash. Current usage={max_RSS()[1]}.")
            with self.src_in.open(encoding='utf8') as src, self.tgt_in.open(encoding='utf8') as tgt:
                recs = read_raw_parallel_lines(src, tgt)
                self._inp_recs = list(recs)
            self.n_inp_recs = len(self._inp_recs)
            log.warning(f"Buffered {self.n_inp_recs:,} records Current memory usage={max_RSS()[1]}")
        return self._inp_recs

    def _get_freqs(self, lines):
        stats = collections.Counter()
        for line in tqdm(lines, desc="Counting frequencies"):
            stats.update(line.split())
        stats = {k: v for k, v in sorted(stats.items(), key=lambda x: x[1], reverse=True)}
        return stats

    @property
    def src_freqs(self):
        if not self._src_freqs:
            self._src_freqs = self._get_freqs(lines=(s for s, t in self.inp_recs))
        return self._src_freqs

    @property
    def tgt_freqs(self):
        if not self._tgt_freqs:
            self._tgt_freqs = self._get_freqs(lines=(s for s, t in self.inp_recs))
        return self._tgt_freqs

    def write_rec(self, src: str, tgt: str, tag: str):
        self.outs[0].write(f"{src.strip()}\n")
        self.outs[1].write(f"{tgt.strip()}\n")
        self.outs[2].write(f"{tag.strip()}\n")
        self.n_out_recs += 1

    def close(self):
        log.info(f"Written {self.n_out_recs} records to {self.src_out, self.tgt_out}")
        for o in self.outs:
            o.close()

    def buffered_cartesian(self, recs: List[Tuple[str, str]], max_src_len: int, max_tgt_len: int):
        n = len(recs)
        mem = set()
        tot = n ** 2
        while len(mem) < tot:
            # generate a random number [0, n**2]
            # think of number is like a cell in big square, wrap the cell_idx to row_idx and col_idx
            r = np.random.randint(0, tot, dtype=np.int64)
            if r in mem:
                continue
            mem.add(r)
            x, y = r // n, r % n
            s1, t1 = recs[x]
            s2, t2 = recs[y]
            if len(s1) + len(s2) + 1 > max_src_len or (len(s2) + len(t2) + 1 > max_tgt_len):
                continue
            src = recs[x][0] + ' ' + recs[y][0]
            tgt = recs[x][1] + ' ' + recs[y][1]
            yield src, tgt

    def run(self, copy=False, noise_src=False, denoise_tgt=False, concat=0, reverse_src=False, reverse_tgt=False, **args):
        assert copy or noise_src or denoise_tgt or concat or reverse_src or reverse_tgt, 'no augmentations are enabled'
        _ = self.inp_recs  # load recs
        if copy:
            log.info("copying input to output")
            for src, tgt in tqdm(self.inp_recs, total=self.n_inp_recs, desc="Copy"):
                self.write_rec(src, tgt, 'ORIG_INP')
        if reverse_src or reverse_tgt:
            log.info(f"Reversing words src:{reverse_src} tgt:{reverse_tgt} ")
            for src, tgt in tqdm(self.inp_recs, total=self.n_inp_recs, desc="Reversing"):
                if reverse_src:
                    src2 = ' '.join(reversed(src.split()))
                    self.write_rec(src2, tgt, 'REV_SRC')
                if reverse_tgt:
                    tgt2 = ' '.join(reversed(tgt.split()))
                    self.write_rec(tgt2, tgt, 'REV_TGT')

        for enabled, side in [(noise_src, 'src'), (denoise_tgt, 'tgt')]:
            if not enabled:
                continue  # skip
            chain = []
            if args.get('word_drop'):
                chain.append(WordDropout(rate=args.get('word_drop')))
            if args.get('word_shuffle'):
                chain.append(WordShuffle(rate=args.get('word_shuffle')))
            if args.get('word_replace'):
                replacements = list((self.src_freqs if side == 'src' else self.tgt_freqs).keys())
                chain.append(WordReplace(rate=args.get('word_replace'), replacements=replacements))
            if args.get('word_mask'):
                chain.append(WordReplace(rate=args.get('word_replace'), replacements=['<MASK>']))
            transform = Transforms(chain=chain)

            for src, tgt in tqdm(self.inp_recs, total=self.n_inp_recs, desc=f"Writing noisy {side} recs"):
                src, tgt = src.split(), tgt.split()
                if side == 'src':
                    src = transform(src)
                    tag = 'NOISY_SRC'
                else:
                    assert side == 'tgt'
                    src = transform(tgt)
                    tag = 'DENOISE_TGT'
                if not src or not tgt: # this was not augmented, so skip
                    continue
                src, tgt = ' '.join(src), ' '.join(tgt)
                self.write_rec(src, tgt, tag)

        if concat > 0:
            max_src_len = args['max_src_len']
            max_tgt_len = args['max_tgt_len']
            assert max_src_len > 0
            assert max_tgt_len > 0
            short_recs = [(s, t) for s, t in self.inp_recs if
                          (len(s) < int(0.7 * max_src_len)) and (len(t) < int(0.7 * max_tgt_len))]

            n_samples = int(self.n_inp_recs * concat)
            total_samples = len(short_recs) ** 2 - len(short_recs)  # approximation
            fraction = n_samples / total_samples
            log.info(f"Sampling {self.n_inp_recs:,} x {concat} = {n_samples:,} out of {total_samples:g}"
                     f" total possible concats (approx.). fraction={fraction:g}")
            recs = self.buffered_cartesian(recs=short_recs, max_src_len=max_src_len, max_tgt_len=max_tgt_len)
            i = 0
            for src, tgt in tqdm(recs, desc="Writing concats", total=n_samples):
                self.write_rec(src, tgt, 'CONCAT1')
                i += 1
                if i >= n_samples:
                    log.info(f"Stopping at {i:,} samples...")
                    break
            if i < n_samples:
                log.warning(f"Expected to get {n_samples} recs but got only {i}")


def main(**kwargs):
    args = kwargs or vars(parse_args())
    props = ['src_in', 'src_out', 'tgt_in', 'tgt_out', 'meta_out']
    props = {pn: args.pop(pn) for pn in props}
    aug = Augmentor(**props)
    try:
        aug.run(**args)
    finally:
        aug.close()


def bound_float(s, mn=0.0, mx=1.0):
    f = float(s)
    assert mn <= f <= mx


def parse_args():
    parser = argparse.ArgumentParser(description="Augment parallel sentences")
    io_p = parser.add_argument_group("input-out")
    io_p.add_argument("-si", '--src-in', type=Path, help="Source input file", required=True)
    io_p.add_argument("-so", '--src-out', type=Path, help="Source output file path", required=True)
    io_p.add_argument("-ti", '--tgt-in', type=Path, help="target input file", required=True)
    io_p.add_argument("-to", '--tgt-out', type=Path, help="target output file path", required=True)
    io_p.add_argument("-mo", '--meta-out', type=Path, help="Metadata output file path", required=True)

    copy_p = parser.add_argument_group("copy")
    copy_p.add_argument("-cp", "--copy", action="store_true", help="Copy input to output. default=%(default)s")
    noise_p = parser.add_argument_group("noise")
    noise_p.add_argument("-ns", "--noise-src", action="store_true",
                         help="augment (noise(source), target) records "
                              " See -wd and -ws to control noise rate. default=%(default)s")
    noise_p.add_argument("-dt", "--denoise-tgt", action="store_true",
                         help="Augment (noise(target),target) records."
                              " See -wd and -ws to control noise rate. default=%(default)s")
    noise_p.add_argument("-wd", "--word-drop", metavar="RATE", type=float, default=0.1,
                         help="What percent of words are to be dropped.")
    noise_p.add_argument("-ws", "--word-shuffle", metavar="RATE", type=bound_float, default=0.1,
                         help="What percent of words to shuffle? Range: [0, 1], default=%(default)s")
    noise_p.add_argument("-wr", "--word-random", metavar="RATE", type=bound_float, default=0.1,
                         help="What percent of words to randomly replace? default=%(default)s")

    cat_p = parser.add_argument_group("concatenation")
    cat_p.add_argument("-cat", "--cat", "--concat", dest='concat', metavar='RATIO', type=float, default=0.0,
                       help="RATIO greater than 0 enables random parallel sentence concatenation. default=%(default)s."
                            " Number of augmentations = RATIO*|INPUT|. See -msl and -mtl to control lengths")
    cat_p.add_argument("-msl", "--max-src-len", metavar='N_CHARS', type=int, default=200,
                       help="Maximum chars in source sequence. Active iff -cat > 0. default=%(default)s")
    cat_p.add_argument("-mtl", "--max-tgt-len", metavar='N_CHARS', type=int, default=200,
                       help="Maximum chars in target sequence. Active iff -cat > 0. default=%(default)s")

    rev_p = parser.add_argument_group("reverse")
    rev_p.add_argument("-rs", "--reverse-src", dest='reverse_src', action='store_true',
                       help="Enables augmentation of (reversed(src.split()), tgt). Words are split by white space."
                            " default=%(default)s. Number of augmentations = |INPUT|.")
    rev_p.add_argument("-rt", "--reverse-tgt", dest='reverse_tgt', action='store_true',
                       help="Enables augmentation having (reversed(tgt.split()), tgt). Words are split by white space."
                            " default=%(default)s. Number of augmentations = |INPUT|.")
    return parser.parse_args()


if __name__ == '__main__':
    main()
