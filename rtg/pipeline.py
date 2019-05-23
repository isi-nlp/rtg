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
from dataclasses import dataclass
import torch
import random
from collections import defaultdict
from mosestokenizer import MosesDetokenizer
from sacrebleu import corpus_bleu, BLEU
import inspect


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
            src, ref = Path(src).resolve(), Path(ref).resolve()
            assert src.exists()
            assert ref.exists()
            assert line_count(src) == line_count(ref)

        script: Path = RTG_PATH / 'scripts' / 'detok-n-bleu.sh'
        if not script.exists():
            script = RTG_PATH.parent / 'scripts' / 'detok-n-bleu.sh'
        assert script.exists(), 'Unable to locate detok-n-bleu.sh script'
        self.script = script

    def detokenize(self, inp: Path, out: Path, col=0, lang='en', post_op=None):
        log.info(f"detok : {inp} --> {out}")
        tok_lines = IO.get_lines(inp, col=col,line_mapper=lambda x: x.split())
        with MosesDetokenizer(lang=lang) as detok:
            detok_lines = (detok(tok_line) for tok_line in tok_lines)
            if post_op:
                detok_lines = (post_op(line) for line in detok_lines)
            IO.write_lines(out, detok_lines)

    def evaluate_file(self, hyp: Path, ref: Path, lowercase=True) -> float:
        detok_hyp = hyp.with_name(hyp.name + '.detok')
        self.detokenize(hyp, detok_hyp)
        detok_lines = IO.get_lines(detok_hyp)
        ref_lines = IO.get_lines(ref)
        bleu: BLEU = corpus_bleu(sys_stream=detok_lines, ref_streams=ref_lines, lowercase=lowercase)
        bleu_str = f'BLEU = {bleu.score:.2f} {"/".join(f"{p:.1f}" for p in bleu.precisions)}' \
            f' (BP = {bleu.bp:.3f} ratio = {(bleu.sys_len / bleu.ref_len):.3f}' \
            f' hyp_len = {bleu.sys_len:d} ref_len={bleu.ref_len})'
        bleu_file = detok_hyp.with_suffix('.lc.bleu' if lowercase else '.oc.bleu')
        log.info(f'BLEU {hyp} : {bleu_str}')
        IO.write_lines(bleu_file, bleu_str)
        return bleu.score

    def decode_eval_file(self, decoder, src_file: Path, out_file: Path, ref_file: Path,
                         lowercase: bool, **dec_args) -> float:
        if out_file.exists() and out_file.stat().st_size > 0 \
                and line_count(out_file) == line_count(src_file):
            log.warning(f"{out_file} exists and has desired number of lines. Skipped...")
        else:
            log.info(f"decoding {src_file.name}")
            with IO.reader(src_file) as inp, IO.writer(out_file) as out:
                decoder.decode_file(inp, out, **dec_args)
        return self.evaluate_file(out_file, ref_file, lowercase=lowercase)

    def tune_decoder_params(self, exp: Experiment, tune_src: str, tune_ref: str,
                            trials: int = 20, strategy: str = 'random', lowercase=True,
                            beam_size=[1, 20], ensemble=[1, 10], lp_alpha=[0.0, 1.0]):
        _, _, _, tune_args = inspect.getargvalues(inspect.currentframe())
        del tune_args[exp]  # exclude some args

        _, step = exp.get_last_saved_model()
        tune_dir = exp.work_dir / f'tune_step{step}'
        log.info(f"Tune dir = {tune_dir}")
        tune_dir.mkdir(parents=True, exist_ok=True)
        assert strategy == 'random'  # only supported strategy for now
        tune_src, tune_ref = Path(tune_src), Path(tune_ref)
        assert tune_src.exists()
        assert tune_ref.exists()

        beam_sizes = [random.randint(beam_size[0], beam_size[1]) for _ in range(trials)]
        ensembles = [random.randint(ensemble[0], ensemble[1]) for _ in range(trials)]
        lp_alphas = [random.uniform(lp_alpha[0], lp_alpha[1]) for _ in range(trials)]

        # ensembling is somewhat costlier, so try minimize the model ensembling, by grouping them together
        grouped_ens = defaultdict(list)
        for b, e, l in zip(beam_sizes, ensembles, lp_alphas):
            grouped_ens[e].append((b, l))

        samples = []
        for e, args in grouped_ens.items():
            decoder = Decoder.new(exp, ensemble=e)
            for b_s, lp_a in args:
                name = f'tune_step{step}_beam{b_s}_ens{e}_lp{lp_a}'
                out_file = tune_dir / f'{name}.out.tsv'
                score = self.decode_eval_file(decoder, tune_src, out_file, tune_ref,
                                              beam_size=b_s, lp_alpha=lp_a, lowercase=lowercase)
                samples.append((score, dict(beam_size=b_s, lp_alpha=lp_a, ensemble=e)))
        return sorted(samples, key=lambda x: x[0], reverse=True)[0][1], tune_args

    def run_tests(self, exp=None, args=None):
        if exp is None:
            exp = self.exp
        if not args:
            args = exp.config['tester']
        suit: Dict[str, List] = args['suit']
        assert suit
        log.info(f"Found {len(suit)} suit :: {suit.keys()}")

        _, step = exp.get_last_saved_model()
        if 'decoder' not in args:
            args['decoder'] = {}
        dec_args: Dict = args['decoder']
        dec_params = dec_args
        if 'tune' in dec_args and not dec_args.get('tuned'):
            tune_args: Dict = dec_args['tune']
            prep_args = exp.config['prep']
            if 'tune_src' not in tune_args:
                tune_args['tune_src'] = prep_args['valid_src']
            if 'tune_ref' not in tune_args:
                tune_args['tune_ref'] = prep_args.get('valid_ref', prep_args['valid_tgt'])
            dec_params, tune_args_ext = self.tune_decoder_params(**tune_args)
            tune_args.update(tune_args_ext)  # Update the config file with default args
            dec_args['tuned'] = True

        beam_size = dec_params.get('beam_size', 4)
        ensemble: int = dec_params.pop('ensemble', 5),
        lp_alpha = dec_params.get('lp_alpha', 0.0)
        dec_args.update(dict(beam_size=beam_size, lp_alpha=lp_alpha, ensemble=ensemble))
        exp.persist_state()  # update the config

        assert step > 0, 'looks like no model is saved or invalid experiment dir'
        test_dir = exp.work_dir / f'test_step{step}_beam{beam_size}_ens{ensemble}_lp{lp_alpha}'
        log.info(f"Test Dir = {test_dir}")
        test_dir.mkdir(parents=True, exist_ok=True)

        decoder = Decoder.new(exp, ensemble=ensemble)
        for name, (orig_src, orig_ref) in suit.items():
            # noinspection PyBroadException
            try:
                orig_src, orig_ref = Path(orig_src).resolve(), Path(orig_ref).resolve()
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
                    if out_file.exists():
                        log.warning(f"{out_file} exists and not empty. going to be overwritten...")
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
        self.pre_checks()  # fail early, so TG can fix and restart
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
