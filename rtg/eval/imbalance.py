#!/usr/bin/env python
#
# Author: Thamme Gowda [tg (at) isi (dot) edu] 
# Created: 5/7/20
import logging as log
from pathlib import Path
from rtg import TranslationExperiment as Experiment
from rtg.data.dataset import SqliteFile
import collections as coll
import numpy as np

log.basicConfig(level=log.INFO)


def main(args=None):
    args = args or parse_args()
    exp = Experiment(args.exp)
    assert exp.train_db.exists()
    train_data = SqliteFile(exp.train_db)
    print(f"Experiment: {exp.work_dir} shared_vocab:{exp.src_vocab is exp.tgt_vocab}")
    for side, var in [('src', 'x'), ('tgt', 'y')]:
        term_freqs = coll.Counter(tok for rec in train_data.get_all([var]) for tok in rec[var])
        lens = np.array(list(rec[f'{var}_len'] for rec in train_data.get_all([f'{var}_len'])))
        tot_toks = sum(lens)
        n_types = dict(src=len(exp.src_vocab), tgt=len(exp.tgt_vocab))[side]
        assert sum(term_freqs.values()) == tot_toks

        uniform = 1 / n_types
        probs = [freq / tot_toks for freq in term_freqs.values()]
        div = 0.5 * sum(abs(uniform - prob) for prob in probs)
        print(f"{side} types: {n_types} toks: {tot_toks:,} len_mean: {np.mean(lens):.4f} "
              f"len_median: {np.median(lens)} imbalance: {div:.4f}")
    print(f'n_segs: {len(lens):,}')


def parse_args():
    import argparse
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('exp', type=Path, help='Path to experiment directory')
    return p.parse_args()


if __name__ == '__main__':
    main()
