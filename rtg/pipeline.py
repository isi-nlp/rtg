#!/usr/bin/env python
#
# Author: Thamme Gowda [tg (at) isi (dot) edu] 
# Created: 3/9/19

import argparse
from rtg import log, TranslationExperiment as Experiment
from pathlib import Path
from typing import Dict, List, Optional
from rtg.module.decoder import Decoder
from rtg import RTG_PATH
from rtg.utils import IO, line_count
import subprocess
from dataclasses import dataclass
import torch

@dataclass
class Pipeline:
    exp: Experiment
    script: Optional[Path] = None

    def pre_checks(self):
        # Some more validation needed
        assert self.exp.work_dir.exists()
        assert self.exp.config.get('prep') is not None
        assert self.exp.config.get('trainer') is not None
        assert self.exp.config.get('tester') is not None
        assert self.exp.config['tester']['suit'] is not None
        for name, (src, ref) in self.exp.config['tester']['suit'].items():
            src, ref = Path(src), Path(ref)
            assert src.exists()
            assert ref.exists()
            assert line_count(src) == line_count(ref)

        script: Path = RTG_PATH / 'scripts' / 'detok-n-bleu.sh'
        if not script.exists():
            log.warning(f"Not found: {script}")
            script = RTG_PATH.parent / 'scripts' / 'detok-n-bleu.sh'
            log.warning(f"Not found: {script}")
        assert script.exists(), 'Unable to locate detok-n-bleu.sh script'
        self.script = script

    def evaluate_file(self, hyp: Path, ref: Path):
        for x in [hyp, ref]:
            assert x.exists()
        cmd = f'{self.script} -h {hyp} -r {ref}'
        subprocess.run(cmd, shell=True, check=True)

    def run_tests(self, exp=None, args=None):
        if exp is None:
            exp = self.exp
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
            # noinspection PyBroadException
            try:
                orig_src, orig_ref = Path(orig_src), Path(orig_ref)
                src_link = test_dir / f'{name}.src'
                ref_link = test_dir / f'{name}.ref'
                for link, orig in [(src_link, orig_src), (ref_link, orig_ref)]:
                    if not link.exists():
                        link.symlink_to(orig)
                out_file = test_dir / f'{name}.out.tsv'
                if out_file.exists() and out_file.stat().st_size > 0 \
                        and line_count(out_file) == line_count(orig_src):
                    log.warning(f"{out_file} exists and has desired number of lines. Skipped...")
                else:
                    log.warning(f"{out_file} exists and not empty. goint to be overwritten...")
                    log.info(f"decoding {name}: {orig_src}")
                    with IO.reader(src_link) as inp, IO.writer(out_file) as out:
                        decoder.decode_file(inp, out, **dec_args)
                self.evaluate_file(out_file, ref_link)
            except Exception as e:
                log.exception(f"Something went wrong with '{name}' test")
                err = test_dir / f'{name}.err'
                err.write_text(str(e))

    def run(self):
        log.update_file_handler(str(self.exp.log_file))
        self.pre_checks()   # fail early, so TG can fix and restart
        self.exp.pre_process()
        self.exp.train()
        with torch.no_grad():
            exp = Experiment(self.exp.work_dir, read_only=True)
            self.run_tests(exp)


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
    pipe = Pipeline(exp=parse_args())
    pipe.run()
