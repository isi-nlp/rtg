#!/usr/bin/env python
#
# Author: Thamme Gowda [tg (at) isi (dot) edu] 
# Created: 2019-07-18


import argparse
import sys
import logging as log
import collections as coll
from typing import Tuple, Iterator

log.basicConfig(level=log.INFO)
ParallelRec = Tuple[str, str]
ParallelRecs = Iterator[ParallelRec]
ParallelLines = Iterator[str]


class Corpus:

    def __init__(self, recs: ParallelRecs):
        recs = self.dedup(recs)
        self.recs = [(src.split(), tgt.split()) for src, tgt in recs]

    def __len__(self):
        return len(self.recs)

    def dedup(self, recs):
        table = coll.Counter(tuple(rec) for rec in recs if len(rec) == 2)
        n_uniq = len(table)
        total = sum(table.values())
        log.info(f'Found {n_uniq} unique records')
        log.info(
            f'Found {total - n_uniq} dupes out of {total} total; ratio = {1 - n_uniq / total:.6f}')
        uniq_recs = list(table.keys())
        return uniq_recs

    def filter(self, recs=None, smoothing=1, low_deviations: float = 0, high_deviations: float = 0,
               invert=False):
        """
        :param smoothing: how much smoothing to add to the lengths
        :param {low,high}_deviations: how many deviations from mean
        :param invert: False means return inside; True means outside
        :return:
        """
        recs = recs or self.recs
        log.info(f"Smoothing={smoothing}; total recs={len(recs)}")
        assert low_deviations or high_deviations  # atleast one of them
        import numpy as np
        lens = ((len(src) + smoothing, len(tgt) + smoothing) for src, tgt in recs)
        ratios = [x / y for x, y in lens]
        np_ratios = np.array(ratios)
        mean, std = np_ratios.mean(), np_ratios.std()

        low_deviations = low_deviations if low_deviations else float('inf')
        high_deviations = high_deviations if high_deviations else float('inf')
        low, high = mean - low_deviations * std, mean + high_deviations * std
        log.info(f"mean:{mean}, std:{std:.4f} ;; low:{low:.4f} high:{high:.4f} ;; invert:{invert}")
        # invert => outside the [low, high] range; else inside the range
        ratio_filter = (lambda x: x < low or high < x) if invert else (lambda x: low <= x <= high)
        pairs = [rec for rec, ratio in zip(recs, ratios) if ratio_filter(ratio)]
        return pairs



def main(inp, out, low_deviations, high_deviations, invert=False, smoothing=0):
    recs = [line.rstrip('\n').split('\t') for line in inp]
    data = Corpus(recs)
    if high_deviations or low_deviations:
        out_recs = data.filter(invert=invert, low_deviations=low_deviations,
                                    high_deviations=high_deviations, smoothing=smoothing)
        selected, total = len(out_recs), len(data)
        removed = total - selected
        log.info(f"selected:{selected} total:{total}; removed={removed} ratio={removed/total:.6f}")
    else:
        out_recs = data.recs
        log.warning("No length based filtering done.")

    lines = (f"{' '.join(src)}\t{' '.join(tgt)}\n" for src, tgt in out_recs)
    count = 0
    for line in lines:
        out.write(line)
        count += 1
    log.info(f"Wrote {count} lines to {out}" )


if __name__ == '__main__':
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('-i', '--inp', type=argparse.FileType('r'), default=sys.stdin,
                   help='Input file path. Format: SRC\\tTGT')

    p.add_argument('-o', '--out', type=argparse.FileType('w'), default=sys.stdout,
                   help='Output file path Format: SRC\\tTGT')
    p.add_argument('-lrd', '--low-ratio-dev', dest='low_deviations', type=float, default=0,
                   help='How many standard deviations are allowed below the mean len ratio;'
                        ' zero or None disables it')
    p.add_argument('-hrd', '--high-ratio-dev', dest='high_deviations', type=float, default=0,
                   help='How many standard deviations are allowed above the mean len ratio;'
                        ' zero or None disables it')
    p.add_argument('-v', '--invert', dest='high_deviations', action='store_true',
                   help='invert the length filter i.e., select outside the deviation range')
    args = vars(p.parse_args())
    main(**args)
