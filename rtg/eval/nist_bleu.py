#!/usr/bin/env python
#
# Author: Thamme Gowda [tg (at) isi (dot) edu] 
# Created: 2020-03-03

import argparse
import sys
import logging as log
from typing import List
from nltk.translate import nist_score
from functools import partial

log.basicConfig(level=log.INFO)
Sentence = List[str]


def nist_bleu(refs: List[List[Sentence]], hyps: List[Sentence], n=4):
    assert len(refs) == len(hyps), f'refs:{len(refs)} == hyps:{len(hyps)} ? '
    assert len(refs) > 0
    assert n > 0
    assert isinstance(hyps, list)
    assert isinstance(hyps[0], list)
    assert isinstance(hyps[0][0], str)

    assert isinstance(refs, list)
    assert isinstance(refs[0], list)
    assert isinstance(refs[0][0], list)
    assert isinstance(refs[0][0][0], str)

    return nist_score.corpus_nist(refs, hyps, n=n)

def ws_tokenize(line: str, lower_case=False) -> List[str]:
    if lower_case:
        line = line.lower()
    return line.strip().split()


def read_lines(stream, tokenizer):
    for line in stream:
        yield tokenizer(line)

def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--hyp', type=argparse.FileType('r'), default=sys.stdin,
                   help='Input file path')
    p.add_argument('--ref', type=argparse.FileType('r'), required=True, help='Reference')
    p.add_argument('-n', type=int, default=4, help='Max n-grams')
    p.add_argument('-f', type=int, default=3, help='Floating point precision')
    p.add_argument('-lc', dest='case_insensitive', action='store_true', default=False,
                   help='Case insensitive')
    args = vars(p.parse_args())
    return args


if __name__ == '__main__':
    args = parse_args()
    log.info(f"case_insensitive={args['case_insensitive']}; n={args['n']}")
    tokr = partial(ws_tokenize, lower_case=args['case_insensitive'])
    hyps = list(read_lines(args['hyp'], tokenizer=tokr))
    refs = [[r] for r in read_lines(args['ref'], tokenizer=tokr)]
    score = nist_bleu(refs=refs, hyps=hyps, n=args['n'])
    print(f"%.{args['f']}f" % score)
