#!/usr/bin/env python
#
# Author: Thamme Gowda [tg (at) isi (dot) edu]
# Created: 3/9/19

import argparse
from rtg import log, TranslationExperiment as Experiment
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
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
import copy
import json
import subprocess


@dataclass
class Pipeline:
    exp: Experiment

    def pre_checks(self):
        # Some more validation needed
        assert self.exp.work_dir.exists()
        conf = self.exp.config
        assert conf.get('prep') is not None
        assert conf.get('trainer') is not None
        assert conf.get('tester') is not None
        assert conf['tester']['suit'] is not None
        for name, data in conf['tester']['suit'].items():
            if isinstance(data, str):
                src, ref = data, None
            elif isinstance(data, list):
                src, ref = data[0], data[1] if len(data) > 1 else None
            else:
                src, ref = data['src'], data.get('ref')

            src = Path(src).resolve()
            assert src.exists(), f'{src} doesnt exist'
            if ref:
                ref = Path(ref).resolve()
                assert ref.exists(), f'{ref} doesnt exist'
                assert line_count(src) == line_count(ref), f'{src} and{ref} are not parallel'
        assert conf['trainer']['steps'] > 0
        if 'finetune_steps' in conf['trainer']:
            assert conf['trainer']['finetune_steps'] > conf['trainer']['steps']
            if not self.exp.finetune_file.exists():
                assert 'finetune_src' in conf['prep']
                assert 'finetune_tgt' in conf['prep']
                assert Path(conf['prep']['finetune_src']).exists()
                assert Path(conf['prep']['finetune_tgt']).exists()

    def moses_detokenize(self, inp: Path, out: Path, col=0, lang='en', post_op=None):
        log.info(f"detok : {inp} --> {out}")
        tok_lines = IO.get_lines(inp, col=col, line_mapper=lambda x: x.split())
        with MosesDetokenizer(lang=lang) as detok:
            detok_lines = (detok(tok_line) for tok_line in tok_lines)
            if post_op:
                detok_lines = (post_op(line) for line in detok_lines)
            IO.write_lines(out, detok_lines)

    @classmethod
    def shell_pipe(cls, cmd_line, inp, out):
        """

        :param cmd_line: shell commandlines
        :param inp: input file, to read records
        :param out:  output file to store records
        :return:
        """
        log.info("Shell cmd:: {cmd_line}")
        with IO.reader(inp) as rdr, IO.writer(out) as wtr:
            proc = subprocess.Popen(cmd_line, stdin=rdr, stdout=wtr, shell=True)
            proc.wait()
        log.info("Shell cmd:: Done")

    def detokenize(self, inp: Path):

        ext_detokenizer = self.exp.config.get('tester', {}).get('detokenizer')
        if ext_detokenizer:
            detok_file = inp.with_suffix('.detok')
            self.shell_pipe(cmd_line=ext_detokenizer, inp=inp, out=detok_file)
        else:
            detok_file = inp.with_suffix('.mosesdetok')
            self.moses_detokenize(inp, out=detok_file, col=0)
        return detok_file

    def evaluate_file(self, detok_hyp: Path, ref: Union[Path, List[str]], lowercase=True) -> float:
        detok_lines = IO.get_lines(detok_hyp)
        # takes multiple refs, but here we have only one
        ref_liness = [IO.get_lines(ref) if isinstance(ref, Path) else ref]
        bleu: BLEU = corpus_bleu(sys_stream=detok_lines, ref_streams=ref_liness,
                                 lowercase=lowercase)
        # this should be part of new sacrebleu  release (i sent a PR ;)
        bleu_str = f'BLEU = {bleu.score:.2f} {"/".join(f"{p:.1f}" for p in bleu.precisions)}' \
            f' (BP = {bleu.bp:.3f} ratio = {(bleu.sys_len / bleu.ref_len):.3f}' \
            f' hyp_len = {bleu.sys_len:d} ref_len={bleu.ref_len:d})'
        bleu_file = detok_hyp.with_name(detok_hyp.name + ('.lc' if lowercase else '.oc') + '.sacrebleu')
        log.info(f'BLEU {detok_hyp} : {bleu_str}')
        IO.write_lines(bleu_file, bleu_str)
        return bleu.score

    def decode_eval_file(self, decoder, src: Union[Path, List[str]], out_file: Path,
                         ref: Optional[Union[Path, List[str]]],
                         lowercase: bool = True, **dec_args) -> float:
        if out_file.exists() and out_file.stat().st_size > 0 and line_count(out_file) == (
                len(src) if isinstance(src, list) else line_count(src)):
            log.warning(f"{out_file} exists and has desired number of lines. Skipped...")
        else:
            if isinstance(src, Path):
                log.info(f"decoding {src.name}")
                src = list(IO.get_lines(src))
            if isinstance(ref, Path):
                ref = list(IO.get_lines(ref))
            with IO.writer(out_file) as out:
                decoder.decode_file(src, out, **dec_args)
        detok_hyp = self.detokenize(out_file)
        if ref:
            return self.evaluate_file(detok_hyp, ref, lowercase=lowercase)

    def tune_decoder_params(self, exp: Experiment, tune_src: str, tune_ref: str, batch_size: int,
                            trials: int = 10, lowercase=True,
                            beam_size=[1, 4, 8], ensemble=[1, 5, 10], lp_alpha=[0.0, 0.4, 0.6],
                            suggested: List[Tuple[int, int, float]] = None,
                            **fixed_args):
        _, _, _, tune_args = inspect.getargvalues(inspect.currentframe())
        tune_args.update(fixed_args)
        ex_args = ['exp', 'self', 'fixed_args', 'batch_size', 'max_len']
        if trials == 0:
            ex_args += ['beam_size', 'ensemble', 'lp_alpha']
        for x in ex_args:
            del tune_args[x]  # exclude some args

        _, step = exp.get_last_saved_model()
        tune_dir = exp.work_dir / f'tune_step{step}'
        log.info(f"Tune dir = {tune_dir}")
        tune_dir.mkdir(parents=True, exist_ok=True)
        tune_src, tune_ref = Path(tune_src), Path(tune_ref)
        assert tune_src.exists()
        assert tune_ref.exists()
        tune_src, tune_ref = list(IO.get_lines(tune_src)), list(IO.get_lines(tune_ref))
        assert len(tune_src) == len(tune_ref)

        tune_log = tune_dir / 'scores.json'  # resume the tuning
        memory: Dict[Tuple, float] = {}
        if tune_log.exists():
            data = json.load(tune_log.open())
            # JSON keys cant be tuples, so they were stringified
            memory = {eval(k): v for k, v in data.items()}

        beam_sizes, ensembles, lp_alphas = [], [], []
        if suggested:
            if isinstance(suggested[0], str):
                suggested = [eval(x) for x in suggested]
            suggested = [(x[0], x[1], round(x[2], 2)) for x in suggested]
            suggested_new = [x for x in suggested if x not in memory]
            beam_sizes += [x[0] for x in suggested_new]
            ensembles += [x[1] for x in suggested_new]
            lp_alphas += [x[2] for x in suggested_new]

        new_trials = trials - len(memory)
        if new_trials > 0:
            beam_sizes += [random.choice(beam_size) for _ in range(new_trials)]
            ensembles += [random.choice(ensemble) for _ in range(new_trials)]
            lp_alphas += [round(random.choice(lp_alpha), 2) for _ in range(new_trials)]

        # ensembling is somewhat costlier, so try minimize the model ensembling, by grouping them together
        grouped_ens = defaultdict(list)
        for b, ens, l in zip(beam_sizes, ensembles, lp_alphas):
            grouped_ens[ens].append((b, l))
        try:
            for ens, args in grouped_ens.items():
                decoder = Decoder.new(exp, ensemble=ens)
                for b_s, lp_a in args:
                    eff_batch_size = batch_size // b_s  # effective batch size
                    name = f'tune_step{step}_beam{b_s}_ens{ens}_lp{lp_a:.2f}'
                    log.info(name)
                    out_file = tune_dir / f'{name}.out.tsv'
                    score = self.decode_eval_file(decoder, tune_src, out_file, tune_ref,
                                                  batch_size=eff_batch_size, beam_size=b_s,
                                                  lp_alpha=lp_a, lowercase=lowercase, **fixed_args)
                    memory[(b_s, ens, lp_a)] = score
            best_params = sorted(memory.items(), key=lambda x: x[1], reverse=True)[0][0]
            return dict(zip(['beam_size', 'ensemble', 'lp_alpha'], best_params)), tune_args
        finally:
            # JSON keys cant be tuples, so we stringify them
            data = {str(k): v for k, v in memory.items()}
            IO.write_lines(tune_log, json.dumps(data))

    def run_tests(self, exp=None, args=None):
        exp = exp or self.exp
        args = args or exp.config['tester']
        suit: Dict[str, List] = args['suit']
        assert suit
        log.info(f"Found {len(suit)} suit :: {suit.keys()}")

        _, step = exp.get_last_saved_model()
        if 'decoder' not in args:
            args['decoder'] = {}
        dec_args: Dict = args['decoder']
        best_params = copy.deepcopy(dec_args)
        max_len = best_params.get('max_len', 50)
        batch_size = best_params.get('batch_size', 20_000)
        # TODO: this has grown to become messy (trying to make backward compatible, improve the logic here
        if 'tune' in dec_args and not dec_args['tune'].get('tuned'):
            tune_args: Dict = dec_args['tune']
            prep_args = exp.config['prep']
            if 'tune_src' not in tune_args:
                tune_args['tune_src'] = prep_args['valid_src']
            if 'tune_ref' not in tune_args:
                tune_args['tune_ref'] = prep_args.get('valid_ref', prep_args['valid_tgt'])
            best_params, tuner_args_ext = self.tune_decoder_params(
                exp=exp, max_len=max_len, batch_size=batch_size, **tune_args)
            log.info(f"tuner args = {tuner_args_ext}")
            log.info(f"Tuning complete: best_params: {best_params}")
            dec_args['tune'].update(tuner_args_ext)  # Update the config file with default args
            dec_args['tune']['tuned'] = True

        if 'tune' in best_params:
            del best_params['tune']

        log.info(f"params: {best_params}")
        beam_size = best_params.get('beam_size', 4)
        ensemble: int = best_params.pop('ensemble', 5)
        lp_alpha = best_params.get('lp_alpha', 0.0)
        eff_batch_size = batch_size // beam_size

        dec_args.update(dict(beam_size=beam_size, lp_alpha=lp_alpha, ensemble=ensemble,
                             max_len=max_len, batch_size=batch_size))
        exp.persist_state()  # update the config

        assert step > 0, 'looks like no model is saved or invalid experiment dir'
        test_dir = exp.work_dir / f'test_step{step}_beam{beam_size}_ens{ensemble}_lp{lp_alpha}'
        log.info(f"Test Dir = {test_dir}")
        test_dir.mkdir(parents=True, exist_ok=True)

        decoder = Decoder.new(exp, ensemble=ensemble)
        for name, data in suit.items():
            # noinspection PyBroadException
            src, ref = data, None
            out_file = None
            if isinstance(data, list):
                src, ref = data[:2]
            elif isinstance(data, dict):
                src, ref = data['src'], data.get('ref')
                out_file = data.get('out')
            try:
                orig_src = Path(src).resolve()
                src_link = test_dir / f'{name}.src'
                ref_link = test_dir / f'{name}.ref'
                buffer = [(src_link, orig_src)]
                if ref:
                    orig_ref = Path(ref).resolve()
                    buffer.append((ref_link, orig_ref))
                for link, orig in buffer:
                    if not link.exists():
                        link.symlink_to(orig)
                out_file = test_dir / f'{name}.out.tsv' if not out_file else out_file
                out_file.parent.mkdir(parents=True, exist_ok=True)

                self.decode_eval_file(decoder, src_link, out_file, ref_link,
                                              batch_size=eff_batch_size, beam_size=beam_size,
                                              lp_alpha=lp_alpha, max_len=max_len)
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
            self.run_tests()


def parse_args():
    parser = argparse.ArgumentParser(prog="rtg.prep", description="prepare NMT experiment")
    parser.add_argument("exp", help="Working directory of experiment", type=Path)
    parser.add_argument("conf", type=Path, nargs='?',
                        help="Config File. By default <work_dir>/conf.yml is used")
    parser.add_argument("-G", "--gpu-only", action="store_true", default=False,
                        help="Crash if no GPU is available")
    args = parser.parse_args()

    if args.gpu_only:
        assert torch.cuda.is_available(), "No GPU found... exiting"

    conf_file: Path = args.conf if args.conf else args.exp / 'conf.yml'
    assert conf_file.exists()
    return Experiment(args.exp, config=conf_file)


if __name__ == '__main__':
    pipe = Pipeline(exp=parse_args())
    pipe.run()
