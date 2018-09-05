#!/usr/bin/env python
# CLI interface to preparation sub task
import argparse
from argparse import ArgumentDefaultsHelpFormatter as ArgFormatter

from rtg.exp import TranslationExperiment, UnSupervisedMTExp

piece_types = ('unigram', 'bpe', 'char', 'word')


def parse_args():
    parser = argparse.ArgumentParser(prog="rtg.prep", description="prepare NMT experiment",
                                     formatter_class=ArgFormatter)
    parser.add_argument("work_dir", help="Working directory", type=str)
    parser.add_argument("conf_file", help="Config File", type=str)
    return parser.parse_args()


def main():
    args = parse_args()
    exp = TranslationExperiment(args.work_dir, config=args.conf_file)
    if exp.model_type == 'binmt':
        del exp
        exp = UnSupervisedMTExp(args.work_dir, config=args.conf_file)

    return exp.pre_process()


if __name__ == '__main__':
    main()
