#!/usr/bin/env python
#
# Author: Thamme Gowda [tg (at) isi (dot) edu] 
# Created: 5/29/20

import logging as log
import collections as coll
from pathlib import Path
from rtg import TranslationExperiment as Experiment
import numpy as np
from scipy import stats
from functools import partial
log.basicConfig(level=log.INFO)


def get_training_frequencies(freqs_file, n_classes, has_header =True):

    log.info(f"Reading tgt side freqs from {freqs_file}")
    freqs = np.zeros(n_classes, dtype=np.int)
    with freqs_file.open() as rdr:
        term_freqs = [line.rstrip('\n') for line in rdr]
        if has_header:
            # format used by rtg.eval.datastat
            assert term_freqs[0].startswith("#")
            assert term_freqs[1].startswith("#")
            assert not term_freqs[2]
            term_freqs = term_freqs[3:]
        for line in term_freqs:
            term_idx, term, freq = line.split('\t')
            freqs[int(term_idx)] = int(freq)
    return freqs

def evaluate(sys_lines, ref_lines):
    assert len(sys_lines) == len(ref_lines)
    match_count = coll.defaultdict(int)
    sys_count = coll.defaultdict(int)
    ref_count = coll.defaultdict(int)
    for sys, ref in zip(sys_lines, ref_lines):
        sys = coll.Counter(sys)
        ref = coll.Counter(ref)
        for key in sys.keys() | ref.keys():
            sys_count[key] += sys.get(key, 0)
            ref_count[key] += ref.get(key, 0)
            match_count[key] += min(sys.get(key, 0), ref.get(key, 0))
    # all keys
    assert match_count.keys() == sys_count.keys()
    assert match_count.keys() == ref_count.keys()
    precision, recall = {}, {}
    for key, count in match_count.items():
        assert 0 <= count <= sys_count[key]
        assert 0 <= count <= ref_count[key]
        precision[key] = (count / sys_count[key]) if sys_count[key] > 0 else np.nan
        recall[key] = (count / ref_count[key]) if ref_count[key]  > 0 else np.nan
    return precision, recall


def frequency_bias(freqs, scores):
    """freqs is list[idx]=freq  scores is map[idx]=score;
    frequencies should be exhastive, scores can be only for subset of classes """
    pairs = [(freqs[idx], sc) for idx, sc in scores.items()]
    sorted_pairs= list(sorted(pairs, reverse=True))
    sorted_freqs = np.array([f for f, s in sorted_pairs])
    sorted_scores = np.array([s for f, s in sorted_pairs])
    return stats.pearsonr(sorted_freqs, sorted_scores)
    #coeff = np.corrcoef(sorted_freqs, sorted_scores)
    #return np.round(coeff, 3)

def main(args=None):
    args = args or parse_args()
    exp = Experiment(args.exp, read_only=True)
    n_classes = len(exp.tgt_vocab)
    freqs = get_training_frequencies(args.freq, n_classes=n_classes)
    tokr = partial(exp.tgt_vocab.encode_as_ids, add_bos=False, add_eos=False)
    sys = [tokr(line.strip()) for line in args.sys]
    ref = [tokr(line.strip()) for line in args.ref]
    assert len(sys) == len(ref)

    precision, recall = evaluate(sys, ref)
    print('Precsion bias: ', frequency_bias(freqs, precision))
    print('Recall bias: ',  frequency_bias(freqs, recall))

def parse_args():
    import argparse
    import sys
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    p.add_argument('exp', type=Path,  help='Path to experiment')
    p.add_argument('-s', '--sys', type=argparse.FileType('r'), default=sys.stdin,
                   help='System outputs; Multiple systems are allowed.')
    p.add_argument('-r', '--ref', type=argparse.FileType('r'), required=True,
                   help='Reference; Multiple files are allowed - one per system with the matching order')

    p.add_argument('-f', '--freq', type=Path, required=True,
                   help='File that has training frequencies on tgt side. '
                        'Get this from "python -m rtg.eval.datastat <exp> tgt -o <freqs.tsv>"')
    return p.parse_args()


if __name__ == '__main__':
    main()


