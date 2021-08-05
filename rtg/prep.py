#!/usr/bin/env python
# CLI interface to preparation sub task
import argparse
from pathlib import Path

from rtg import TranslationExperiment, log
from rtg.exp import load_conf


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
    ExpFactory = TranslationExperiment
    is_big = load_conf(conf_file).get('spark', {})
    if is_big:
        log.info("Big experiment mode enabled; checking pyspark backend")
        try:
            import pyspark
            log.info("pyspark is available")
        except:
            log.warning("unable to import pyspark. Please do 'pip install pyspark' and run again")
            raise
        from rtg.big.exp import BigTranslationExperiment
        ExpFactory = BigTranslationExperiment

    exp = ExpFactory(args.exp, config=conf_file, read_only=False)
    return exp.pre_process()

if __name__ == '__main__':
    main()
