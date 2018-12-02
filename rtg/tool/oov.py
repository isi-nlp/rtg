#!/usr/bin/env python
#
# Author: Thamme Gowda [tg (at) isi (dot) edu] 
# Created: 12/1/18

import argparse
import sys
from collections import Counter


def tok_stats(lines, lowercase=False):
    lines = (l.strip() for l in lines if l.strip())
    if lowercase:
        lines = (l.lower() for l in lines)
    tokss = (l.split() for l in lines)
    stats = Counter(tok for toks in tokss for tok in toks)
    return stats


def oov_rate(train_stats, test_stats, kind='token'):
    assert kind in ('token', 'type')
    if kind == 'token':
        tot_oov_toks = sum(freq for tok, freq in test_stats.items() if tok not in train_stats)
        tot_test_toks = sum(count for tok, count in test_stats.items())
        return tot_oov_toks / tot_test_toks
    elif kind == 'type':
        tot_oov_types = sum(1 for tok, freq in test_stats.items() if tok not in train_stats)
        tot_test_types = len(test_stats)
        return tot_oov_types / tot_test_types
    else:
        raise Exception('Oh No!')


def main(train, tests):
    train_stats = tok_stats(train)
    tot_train_toks = sum(count for tok, count in train_stats.items())
    print(f'Training:{train.name}\t {len(train_stats)} types \t {tot_train_toks} tokens')
    print("# OOV Rate ::")
    for test in tests:
        name = test.name
        test_stats = tok_stats(test)
        oov_type_rate = 100 * oov_rate(train_stats, test_stats, kind='type')
        oov_tok_rate = 100 * oov_rate(train_stats, test_stats, kind='token')
        tot_test_toks = sum(count for tok, count in test_stats.items())
        tot_test_types = len(test_stats)
        print(f'{name}\t{oov_type_rate:g}% of {tot_test_types} types '
              f'\t {oov_tok_rate:g}% of {tot_test_toks} tokens', )


if __name__ == '__main__':
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('-tr', '--train', type=argparse.FileType('r'), help='Train file path',
                   required=True)
    p.add_argument('-ts', '--test', dest='tests', type=argparse.FileType('r'), default=[sys.stdin],
                   help='Test file paths', nargs='*')
    args = vars(p.parse_args())
    main(**args)
