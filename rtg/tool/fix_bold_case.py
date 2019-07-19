#!/usr/bin/env python
#
# Author: Thamme Gowda [tg (at) isi (dot) edu] 
# Created: 2019-07-18


import argparse
import sys
import logging as log
from typing import List

log.basicConfig(level=log.INFO)


def fix_bold_text(seq:List[str], min_evidence: int = 3):
    flags = [tok.isupper() for tok in seq]

    flags[0] = 1 if flags[0] else 0
    for i in range(1, len(flags)):
        flags[i] = flags[i - 1] + 1 if flags[i] else 0

    fixing = max(flags) >= min_evidence
    if fixing:
        seq = [tok.lower() if tok.isupper() else tok for tok in seq]
        seq[0] = seq[0].title()
    return seq, fixing


def main(inp, out, min_evidence:int):
    seqs = (line.strip().split() for line in inp)
    modified = 0
    count = 0
    for seq in seqs:
        out_seq, changed = fix_bold_text(seq, min_evidence=min_evidence)
        modified += 1 if changed else 0
        out.write(' '.join(out_seq) + '\n')
        count += 1
    log.info(f"Modified {modified} out of {count} lines")


if __name__ == '__main__':
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('-i', '--inp', type=argparse.FileType('r'), default=sys.stdin,
                   help='Input file path. one seq per line')
    p.add_argument('-o', '--out', type=argparse.FileType('w'), default=sys.stdout,
                   help='Output file path Format: One seq per line')
    p.add_argument('-m', '--min-evidence', type=int, default=3,
                   help='Minimum number of consecutive bold tokens to trigger the fixing')
    args = vars(p.parse_args())
    main(**args)
