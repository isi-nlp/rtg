#!/usr/bin/env python
#
# Author: Thamme Gowda [tg (at) isi (dot) edu] 
# Created: 3/9/19

import argparse
from rtg import log, TranslationExperiment as Experiment
from pathlib import Path
from typing import Dict, List
from rtg.module.decoder import Decoder
from rtg import RTG_PATH
from rtg.utils import IO, line_count
import subprocess


def evaluate_file(hyp: Path, ref: Path):
    script: Path = RTG_PATH / 'scripts' / 'detok-n-bleu.sh'
    for x in [hyp, ref, script]:
        assert x.exists(), f'{x}'
    cmd = f'{script} -h {hyp} -r {ref}'
    subprocess.run(cmd, shell=True, check=True)


def run_tests(exp, args=None):
    if not args:
        args = exp.config['tester']
    suit: Dict[str, List] = args['suit']
    assert suit
    log.info(f"Found {len(suit)} suit :: {suit.keys()}")

    _, step = exp.get_last_saved_model()
    dec_args: Dict = args.get('decoder', {})
    beam = dec_args.get('beam', 4)
    ensemble = dec_args.get('ensemble', 5)
    dec_args.update(dict(beam=beam, ensemble=ensemble))
    assert step > 0
    test_dir = exp.work_dir / f'test_step{step}_beam{beam}_ens{ensemble}'
    log.info(f"Test Dir = {test_dir}")
    test_dir.mkdir(parents=True, exist_ok=True)

    decoder = Decoder.new(exp, ensemble=ensemble)

    for name, (orig_src, orig_ref) in suit.items():
        orig_src, orig_ref = Path(orig_src), Path(orig_ref)
        if not orig_src.exists() or not orig_ref.exists():
            log.error(f'Doesnt exist {orig_src} or {orig_ref}')
            continue
        out_file = test_dir / f'{name}.out.tsv'
        if out_file.exists() and out_file.stat().st_size > 0:
            if line_count(out_file) == line_count(orig_src):
                log.warning(f"{out_file} exists and has desired number of lines. Skipped...")
                continue
            else:
                log.warning(f"{out_file} exists and not empty. goint to be overwritten...")
        src_link = test_dir / f'{name}.src'
        ref_link = test_dir / f'{name}.ref'
        if not src_link.exists():
            src_link.symlink_to(orig_src)
        if not ref_link.exists():
            ref_link.symlink_to(orig_ref)

        log.info(f"decoding {name}: {orig_src}")
        with IO.reader(src_link) as inp, IO.writer(out_file) as out:
            decoder.decode_file(inp, out, **dec_args)
        evaluate_file(out_file, ref_link)


def run(exp: Experiment):
    exp.pre_process()
    exp.train()
    exp = Experiment(exp.work_dir, read_only=True)
    run_tests(exp)


def parse_args():
    parser = argparse.ArgumentParser(prog="rtg.prep", description="prepare NMT experiment")
    parser.add_argument("exp", help="Working directory of experiment", type=Path)
    parser.add_argument("conf", type=Path, nargs='?',
                        help="Config File. By default <work_dir>/conf.yml is used")
    args = parser.parse_args()
    conf_file: Path = args.conf if args.conf else args.exp / 'conf.yml'
    assert conf_file.exists()
    return Experiment(args.exp, config=conf_file)


if __name__ == '__main__':
    run(parse_args())
