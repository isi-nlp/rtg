#!/usr/bin/env python
# CLI interface to preparation sub task
import sys
import argparse
from argparse import ArgumentDefaultsHelpFormatter as ArgFormatter

from tgnmt import TranslationExperiment as Experiment


def parse_args():
    parser = argparse.ArgumentParser(prog="tgnmt.prep", description="prepare NMT experiment",
                                     formatter_class=ArgFormatter)
    parser.add_argument("work_dir", help="Working directory", type=str)
    parser.add_argument("-tf", '--train-file', help="Training File.", type=str, required=True)
    parser.add_argument("-vf", '--valid-file', help="Validation File.", type=str, required=True)
    parser.add_argument("-sl", '--src-len', type=int, default=200,
                        help="Truncate or filter source sentences to this length", )
    parser.add_argument("-tl", '--tgt-len', type=int, default=200,
                        help="Truncate or filter target sentences to this length")
    parser.add_argument("-tr", '--truncate', action='store_true',
                        help="Do select all training sentences and truncate them to --src-len and --tgt-len values."
                             " Default is to exclude sentences longer than --src-len and --tgt-len")
    return vars(parser.parse_args())


def main():
    args = parse_args()
    exp = Experiment(args.pop('work_dir'))
    return exp.pre_process(**args)


if __name__ == '__main__':
    main()
