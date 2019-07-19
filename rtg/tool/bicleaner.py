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

    @staticmethod
    def _check_format(recs):
        assert isinstance(recs, list)  # corpus
        assert isinstance(recs[0], list) or isinstance(recs[0], tuple)  # pair or bitext record
        assert isinstance(recs[0][0], list)  # tokenized sequence
        assert isinstance(recs[0][0][0], str)  # token

    def len_filter(self, recs, min_src_len=1, min_tgt_len=1, max_src_len: int = 300,
                   max_tgt_len: int = 300, invert=False):
        log.info(f"Lengths src:[{min_src_len}, {max_src_len}] tgt:[{min_tgt_len}, {max_tgt_len}]")
        self._check_format(recs)
        lens = ((len(src), len(tgt)) for src, tgt in recs)
        pairs = ((pair, min_src_len <= s <= max_src_len and min_tgt_len <= t <= max_tgt_len)
                 for pair, (s, t) in zip(recs, lens))
        if invert:
            pairs = ((x, not y) for x, y in pairs)
        pairs = [pair for pair, flag in pairs if flag]
        return pairs

    def ratio_filter(self, recs, smoothing=0, low_deviations: float = 0,
                     high_deviations: float = 0,
                     invert=False):
        """
        :param smoothing: how much smoothing to add to the lengths
        :param {low,high}_deviations: how many deviations from mean
        :param invert: False means return inside; True means outside
        :return:
        """
        assert low_deviations or high_deviations  # atleast one of them
        self._check_format(recs)
        log.info(f"Smoothing={smoothing}; total recs={len(recs)}: low_dev:{low_deviations}"
                 f" high_dev:{high_deviations}")
        import numpy as np
        lens = ((len(src) + smoothing, len(tgt) + smoothing) for src, tgt in recs)
        ratios = [x / y for x, y in lens]
        np_ratios = np.array(ratios)
        mean, std = np_ratios.mean(), np_ratios.std()

        low_deviations = low_deviations if low_deviations else float('inf')
        high_deviations = high_deviations if high_deviations else float('inf')
        low, high = max(0.1, mean - low_deviations * std), mean + high_deviations * std
        log.info(f"mean:{mean}, std:{std:.4f} ;; low:{low:.4f} high:{high:.4f} ;; invert:{invert}")
        # invert => outside the [low, high] range; else inside the range
        if invert:
            pairs = [rec for rec, ratio in zip(recs, ratios) if not (low <= ratio <= high)]
        else:
            pairs = [rec for rec, ratio in zip(recs, ratios) if low <= ratio <= high]
        return pairs


def main(inp, out, low_deviations, high_deviations, min_src_len, max_src_len, min_tgt_len, max_tgt_len, invert=False,
         smoothing=0):
    recs = [line.rstrip('\n').split('\t') for line in inp]
    data = Corpus(recs)
    recs = data.recs
    if recs and (min_src_len or min_tgt_len or max_src_len or max_tgt_len):
        total = len(recs)
        len_args=dict(min_src_len=min_src_len, max_src_len=max_src_len, min_tgt_len=min_tgt_len,
                      max_tgt_len=max_tgt_len, invert=invert)
        recs = data.len_filter(recs=recs, **len_args)
        selected, dropped = len(recs), total - len(recs)
        ratio = dropped / total if total > 0 else float('inf')
        log.info(f"Length: selected:{selected} total:{total}; dropped={dropped} ratio={ratio:.6f}")
    else:
        log.warning("len based filtering not done.")

    if recs and (high_deviations or low_deviations):
        total = len(recs)
        recs = data.ratio_filter(recs=recs, invert=invert, low_deviations=low_deviations,
                                     high_deviations=high_deviations, smoothing=smoothing)
        selected, dropped = len(recs), total - len(recs)
        ratio = dropped / total if total > 0 else float('inf')
        log.info(f"Ratio :: selected:{selected} total:{total}; dropped={dropped} ratio={ratio:.6f}")
    else:
        log.warning("ratio based filtering not done.")

    lines = (f"{' '.join(src)}\t{' '.join(tgt)}\n" for src, tgt in recs)
    count = 0
    for line in lines:
        out.write(line)
        count += 1
    log.info(f"Wrote {count} lines to {out}")


if __name__ == '__main__':
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('-i', '--inp', type=argparse.FileType('r'), default=sys.stdin,
                   help='Input file path. Format: SRC\\tTGT')

    p.add_argument('-o', '--out', type=argparse.FileType('w'), default=sys.stdout,
                   help='Output file path Format: SRC\\tTGT')
    p.add_argument('-lrd', '--low-ratio-dev', dest='low_deviations', type=float, default=2,
                   help='How many standard deviations are allowed below the mean len ratio;'
                        ' zero or None disables it')
    p.add_argument('-hrd', '--high-ratio-dev', dest='high_deviations', type=float, default=3,
                   help='How many standard deviations are allowed above the mean len ratio;'
                        ' zero or None disables it')
    p.add_argument('-v', '--invert', action='store_true',
                   help='invert the length filter i.e., select outside the deviation range')
    p.add_argument('-mst', '--min-src-len', type=int, default=1, help='Min source tokens')
    p.add_argument('-mtt', '--min-tgt-len', type=int, default=1, help='Min target tokens')
    p.add_argument('-xst', '--max-src-len', type=int, default=300, help='Max source tokens')
    p.add_argument('-xtt', '--max-tgt-len', type=int, default=300, help='Max target tokens')
    args = vars(p.parse_args())
    main(**args)
