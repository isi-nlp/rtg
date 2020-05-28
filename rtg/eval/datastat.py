#!/usr/bin/env python
#
# Author: Thamme Gowda [tg (at) isi (dot) edu] 
# Created: 5/27/20
import logging as log
from pathlib import Path
from rtg import TranslationExperiment
import collections as coll
import numpy as np
from tqdm import tqdm

log.basicConfig(level=log.INFO)

def get_stats(data):
    freqs = coll.Counter()
    n_seqs = 0
    lens = []
    for seq in tqdm(data):
        n_seqs += 1
        freqs.update(seq)
        lens.append(len(seq))
    lens = np.array(lens)
    res = dict(n_seqs=n_seqs,
               total_toks=sum(freqs.values()),
               total_toks2=np.sum(lens),
               mean_len=np.mean(lens),
               median_len=np.median(lens)
               )
    return freqs, res

def main(args=None):
    args = args or parse_args()
    exp = TranslationExperiment(args.exp, read_only=True)

    side = args.side
    def get_data():
        batch_size = 4096 # tokens
        data = exp.get_train_data(batch_size=batch_size, steps=0, batch_first=True, fine_tune=False,
                                  shuffle=False, keep_in_mem=False, sort_by=None)
        for batch in data:
            seqs, lens = (batch.x_seqs, batch.x_len) if side == 'src' \
                else (batch.y_seqs, batch.y_len)
            for i in range(len(batch)):
                yield seqs[i][:lens[i]].tolist()
    freqs, stats = get_stats(get_data())
    #
    fmt_num = lambda x: f'{x:.4f}' if isinstance(x, float) else f'{x:,}'
    header = " ".join(f'{k}: {fmt_num(v)}' for k, v in stats.items())
    field = exp.src_vocab if side == 'src' else exp.tgt_vocab
    out = args.out
    out.write(f'#{args.side} {header}\n\n')
    tfs = list(sorted(freqs.items(), reverse=True, key=lambda x: x[1]))
    for term_id, freq in tqdm(tfs):
        term = field.decode_ids([term_id])
        out.write(f'{term_id}\t{term}\t{freq:g}\n')
    out.close()

def parse_args():
    import argparse
    import sys
    import io
    stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='ignore')
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('exp', type=Path, help='Path to experiment')
    p.add_argument('side', choices=['src', 'tgt'])
    p.add_argument('-o', '--out', type=argparse.FileType('w'), default=stdout,
                   help='Output file path')
    return p.parse_args()

if __name__ == '__main__':
    main()
