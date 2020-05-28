#!/usr/bin/env python
#
# Author: Thamme Gowda [tg (at) isi (dot) edu] 
# Created: 5/27/20
from pathlib import Path
from rtg import TranslationExperiment, log
import numpy as np
from tqdm import tqdm


def get_stats(data, n, limit=-1):
    n_seqs = 0
    lens = []
    freqs = np.zeros(n, dtype=np.int32)
    for seq in tqdm(data):
        n_seqs += 1
        for i in seq:
            freqs[i] += 1
        lens.append(len(seq))
        if limit > 0 and n_seqs > limit:
            log.warning(f"Aborting at {n_seqs} records though here are more in the dataset")
            break
    lens = np.array(lens)
    total_toks = np.sum(lens)
    assert total_toks == np.sum(freqs)  # sanity
    # Zkip zero frequencies; they could be reserved words or from the other side if vocab is shared
    n_zero_types = sum(1 for f in freqs if f == 0)
    n_effective = n - n_zero_types
    probs = freqs / total_toks
    imbalance = 0.5 * np.sum(np.abs(1 / n_effective - probs))
    res = dict(n_seqs=n_seqs,
               total_toks=total_toks,
               mean_len=np.mean(lens),
               median_len=np.median(lens),
               max_len=np.max(lens),
               EMD=imbalance,
               n=n,
               zero_types=n_zero_types,
               effective_n=n_effective)
    return freqs, res


def main(args=None):
    args = args or parse_args()
    exp = TranslationExperiment(args.exp, read_only=True)

    side = args.side

    def get_data():
        batch_size = 4096  # tokens
        data = exp.get_train_data(batch_size=batch_size, steps=0, batch_first=True, fine_tune=False,
                                  shuffle=False, keep_in_mem=False, sort_by=None)
        for batch in data:
            seqs, lens = (batch.x_seqs, batch.x_len) if side == 'src' \
                else (batch.y_seqs, batch.y_len)
            for i in range(len(batch)):
                yield seqs[i][:lens[i]].tolist()

    vocab = exp.src_vocab if side == 'src' else exp.tgt_vocab
    n = len(vocab)
    freqs, stats = get_stats(get_data(), n, limit=args.max_recs)
    out = args.out

    header = []
    fmt_num = lambda x: f'{x:.4f}' if isinstance(x, float) else f'{x:,}'
    for k, v in stats.items():
        header.append(f'{k}: {fmt_num(v)}')
    header = " ".join(header)
    out.write(f'# {args.side} {header}\n')

    header = []
    # exclude zeros
    active_freqs = [f for f in freqs.tolist() if f > 0]
    active_freqs = list(sorted(active_freqs, reverse=True))
    effective_n = len(active_freqs)
    for percent in [0, 1, 5, 10, 25, 50, 75, 90, 95, 98, 99, 100]:
        sort_idx = min(effective_n - 1, int((percent / 100) * effective_n))
        freq = active_freqs[sort_idx]
        header.append(f'{percent}%: {freq}')
    header = "  ".join(header)
    out.write(f"# Freqs {header}\n")

    sort_indices = np.argsort(freqs)  # ascending frequency
    sort_indices = list(reversed(sort_indices.tolist()))  # descending frequency
    for term_id in sort_indices:
        term = vocab.decode_ids([term_id])
        freq = freqs[term_id]
        out.write(f'{term_id}\t{term}\t{freq:g}\n')
    out.flush()


def parse_args():
    import argparse
    import sys
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('exp', type=Path, help='Path to experiment')
    p.add_argument('side', choices=['src', 'tgt'])
    p.add_argument('-mr', '--max-recs', type=int, default=0,
                   help='Limit to --max-recs on training data; useful for quick testing on large datasets.'
                        'a value <= 0 implies no limit (default)')
    p.add_argument('-o', '--out', type=argparse.FileType('w'), default=sys.stdout,
                   help='Output file path')
    return p.parse_args()


if __name__ == '__main__':
    main()
