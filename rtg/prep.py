#!/usr/bin/env python
# CLI interface to preparation sub task
import argparse
from pathlib import Path

from rtg.exp import TranslationExperiment

piece_types = ('unigram', 'bpe', 'char', 'word')


def parse_args():
    parser = argparse.ArgumentParser(prog="rtg.prep", description="prepare NMT experiment")
    parser.add_argument("work_dir", help="Working directory", type=Path)
    parser.add_argument("conf_file", type=Path, nargs='?',
                        help="Config File. By default <work_dir>/conf.yml is used")
    return parser.parse_args()


def main():
    args = parse_args()
    conf_file: Path = args.conf_file if args.conf_file else args.work_dir / 'conf.yml'
    assert conf_file.exists()
    exp = TranslationExperiment(args.work_dir, config=conf_file)
    return exp.pre_process()


if __name__ == '__main__':
    main()
