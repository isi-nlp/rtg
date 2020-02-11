#!/usr/bin/env python
# Computes BLEU score per line
# Author: Thamme Gowda [tg (at) isi (dot) edu] 
# Created: 1/7/19

from typing import List, Union
from collections import Counter
import functools
import operator
import argparse
import sys
import logging as log
from itertools import zip_longest
import io

log.basicConfig(level=log.INFO)


def n_gram_precision(cand: List[str], ref: List[str], n: int) -> float:
    """
    Computes ngram precision
    :param cand: candidate (aka output) tokens
    :param ref: reference tokens
    :param n: n as in ngram
    :return:
    """
    assert n > 0
    # Make ngrams out of sequence
    cand_grams = [tuple(cand[i: i + n]) for i in range(len(cand) - n + 1)]
    if not cand_grams:
        # sequence is shorter than n
        return 1.0    # precision of emty is fine, recall is bad
    ref_grams = [tuple(ref[i: i + n]) for i in range(len(ref) - n + 1)]
    # Count of ngrams
    cand_grams_ct = Counter(cand_grams)
    ref_grams_ct = Counter(ref_grams)
    precise_grams_ct = sum(min(cgram_ct, ref_grams_ct.get(cgram, 0))
                           for cgram, cgram_ct in cand_grams_ct.items())
    precision = precise_grams_ct / len(cand_grams)
    return precision


def sentence_bleu(cand: Union[str, List[str]], ref: Union[str, List[str]], n: int = 4) -> float:
    """
    Computes bleu between two sequences.
    Note: implemented by following http://www.statmt.org/book/slides/08-evaluation.pdf slide 16
    :param cand: candidate (aka output) sequence
    :param ref: reference sequence
    :param n: n gram size, default=4
    :return: bleu score
    """
    assert type(cand) is type(ref)

    if isinstance(cand, str):
        cand = cand.strip().split()
        ref = ref.strip().split()
    n_precisions = [n_gram_precision(cand, ref, i) for i in range(1, n + 1)]
    n_precision = functools.reduce(operator.mul, n_precisions, 1.0)
    precision = pow(n_precision, 1.0 / n)
    brevity_penalty = min(1.0, len(cand) / len(ref))  # TODO: exponent
    bleu_score = brevity_penalty * precision
    if log.getLogger().isEnabledFor(level=log.DEBUG):
        msg = f'score={bleu_score:g} brev_penalty={brevity_penalty:g} precisions:{n_precisions}' \
              f' precision={precision:g} || cand:{cand} || ref:{ref}'
        log.debug(msg)
    return bleu_score


def nltk_sentence_bleu(cand, ref):
    from nltk.translate.bleu_score import sentence_bleu as nltk_sent_bleu
    assert type(cand) is type(ref)

    if isinstance(cand, str):
        cand = cand.strip().split()
        ref = ref.strip().split()
    return nltk_sent_bleu([ref], cand)


def main(cands, refs, n, out, no_refs=False, no_cands=False):
    for cand, ref in zip_longest(cands, refs):
        if cand is None or ref is None:
            raise Exception("Candidate and reference files have unequal lengths."
                            " Expected same line count in both files")
        cand, ref = cand.strip(), ref.strip()
        score = sentence_bleu(cand, ref, n=n)
        score2 = nltk_sentence_bleu(cand, ref)
        if score2 < 1e-12:
            score2 = 0.0
        line = f'{score:g}\t{score2:g}'
        if not no_cands:
            line += f'\t{cand}'
        if not no_refs:
            line += f'\t{ref}'
        out.write(line + '\n')


if __name__ == '__main__':
    stdin = io.TextIOWrapper(sys.stdin.buffer, encoding='utf-8', errors='ignore', newline='\n')
    stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='ignore')
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                description="Computes BLEU score per record.")
    p.add_argument('-c', '--cands', type=argparse.FileType('r'), default=stdin,
                   help='Candidate (aka output from NLG system) file')
    p.add_argument('-r', '--refs', type=argparse.FileType('r'), default=stdin,
                   help='Reference (aka human label) file')
    p.add_argument('-n', '--n', type=int, default=4,
                   help='maximum n as in ngram.')
    p.add_argument('-nr', '--no-refs', help='Do not write references to --out',
                   action='store_true')
    p.add_argument('-nc', '--no-cands', help='Do not write candidates to --out',
                   action='store_true')
    p.add_argument('-o', '--out', type=argparse.FileType('w'), default=stdout,
                   help='Output file path to store the result.')
    p.add_argument('-v', '--verbose', action='store_true', help='verbose mode')
    args = vars(p.parse_args())
    assert not(args['cands'] == stdin and args['refs'] == stdin), \
        'Only one of --refs and --cands can be read from STDIN'

    if args.pop('verbose'):
        log.getLogger().setLevel(level=log.DEBUG)
    main(**args)
