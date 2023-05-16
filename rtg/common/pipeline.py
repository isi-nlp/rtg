#!/usr/bin/env python
#
# Author: Thamme Gowda [tg (at) isi (dot) edu]
# Created: 3/9/19

import argparse
import copy
import inspect
import json
import os
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
from sacrebleu import corpus_bleu, corpus_macrof

from rtg import __version__, debug_mode, log
from rtg.common import dtorch
from rtg.common.exp import BaseExperiment as Experiment
from rtg.registry import ProblemType
from rtg.utils import IO, line_count


@dataclass
class Pipeline:
    exp: Experiment

    def __post_init__(self):
        self.tests_types = {
            ProblemType.TRANSLATION: self.run_translation_tests,
            ProblemType.CLASSIFICATION: self.run_classification_tests,
        }

    def pre_checks(self):
        # Some more validation needed
        assert self.exp.work_dir.exists()
        conf = self.exp.config
        assert conf.get('prep') is not None
        assert conf.get('trainer') is not None
        assert conf.get('tester') is not None
        if not conf['tester'].get('suite') and conf['tester'].get('suit'):
            # it was mis spelled as suit https://github.com/isi-nlp/rtg/issues/9
            conf['tester']['suite'] = conf['tester']['suit']
        assert conf['tester'].get('suite') is not None
        for name, data in conf['tester']['suite'].items():
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

    def detokenize(self, inp: Path):
        post_proc = self.exp.get_post_transform(side='tgt')
        detok_file = inp.with_suffix('.detok')
        with inp.open() as lines, detok_file.open('w') as out:
            for line in lines:
                line = line.split('\t')[0]
                out.write(post_proc(line) + '\n')
        return detok_file

    def evaluate_mt_file(self, detok_hyp: Path, ref: Union[Path, List[str]], lowercase=True) -> float:
        detok_lines = list(IO.get_lines(detok_hyp))
        # takes multiple refs, but here we have only one
        if isinstance(ref, Path):
            ref = [x.strip() for x in IO.get_lines(ref)]
        assert isinstance(ref, list), f'List of strings expected, but given {type(ref)} '
        assert isinstance(ref[0], str), f'List of strings expected, but given List of {type(ref[0])} '
        refs = [ref]
        bleu = corpus_bleu(hypotheses=detok_lines, references=refs, lowercase=lowercase)
        bleu_str = bleu.format()
        bleu_file = detok_hyp.with_name(detok_hyp.name + ('.lc' if lowercase else '.oc') + '.sacrebleu')
        log.info(f'{detok_hyp}: {bleu_str}')
        IO.write_lines(bleu_file, bleu_str)
        macrof1 = corpus_macrof(hypotheses=detok_lines, references=refs, lowercase=lowercase)
        macrof1_str = macrof1.format()
        macrof1_file = detok_hyp.with_name(detok_hyp.name + ('.lc' if lowercase else '.oc') + '.macrof1')
        log.info(f'{detok_hyp}: {macrof1_str}')
        IO.write_lines(macrof1_file, macrof1_str)
        return bleu.score

    def decode_eval_file(
        self,
        decoder,
        src: Union[Path, List[str]],
        out_file: Path,
        ref: Optional[Union[Path, List[str]]],
        lowercase: bool = True,
        **dec_args,
    ) -> float:
        if (
            out_file.exists()
            and out_file.stat().st_size > 0
            and line_count(out_file) == (len(src) if isinstance(src, list) else line_count(src))
        ):
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
            return self.evaluate_mt_file(detok_hyp, ref, lowercase=lowercase)

    def tune_decoder_params(
        self,
        exp: Experiment,
        tune_src: str,
        tune_ref: str,
        batch_size: int,
        trials: int = 10,
        lowercase=True,
        beam_size=(1, 4, 8),
        ensemble=(1, 5, 10),
        lp_alpha=(0.0, 0.4, 0.6),
        suggested: List[Tuple[int, int, float]] = None,
        **fixed_args,
    ):
        _, _, _, tune_args = inspect.getargvalues(inspect.currentframe())
        from rtg.nmt.decoder import Decoder

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
                    score = self.decode_eval_file(
                        decoder,
                        tune_src,
                        out_file,
                        tune_ref,
                        batch_size=eff_batch_size,
                        beam_size=b_s,
                        lp_alpha=lp_a,
                        lowercase=lowercase,
                        **fixed_args,
                    )
                    memory[(b_s, ens, lp_a)] = score
            best_params = sorted(memory.items(), key=lambda x: x[1], reverse=True)[0][0]
            return dict(zip(['beam_size', 'ensemble', 'lp_alpha'], best_params)), tune_args
        finally:
            # JSON keys cant be tuples, so we stringify them
            data = {str(k): v for k, v in memory.items()}
            IO.write_lines(tune_log, json.dumps(data))

    def run_classification_tests(self, exp=None, args=None):
        from rtg.classifier.transformer import ClassificationExperiment

        exp: ClassificationExperiment = exp or self.exp
        assert exp.problem_type is ProblemType.CLASSIFICATION
        args = args or exp.config['tester']
        suite: Dict[str, List] = args['suite']
        assert suite
        log.info(f"Found {len(suite)} suite :: {suite.keys()}")

        eval_args = dict(
            batch_size=args.get('batch_size') or self.exp.config['trainer']['batch_size'],
            max_len=args.get('max_len', 256),
        )
        ens = args.get('ensemble', 1)
        _, step = exp.get_last_saved_model()
        model = exp.load_model(ensemble=ens)
        model = model.eval()
        test_dir = exp.work_dir / f'test_step{step}_ens{ens}'
        test_dir.mkdir(exist_ok=True, parents=True)
        for name, data in suite.items():
            src, label = data, None
            if isinstance(data, list):
                src, label = data[:2]
            try:
                src_link = test_dir / f'{name}.src'
                label_link = test_dir / f'{name}.label'
                out_file = test_dir / f'{name}.out.tsv'
                if out_file.exists() and out_file.stat().st_size > 0:
                    log.warning(f"{out_file} exists and not empty, so skipping it")
                    continue
                buffer = [(src_link, Path(src).absolute())]
                if label:
                    buffer.append((label_link, Path(label).absolute()))
                for link, orig in buffer:
                    if not link.exists():
                        orig_rel = os.path.relpath(orig, link.parent)
                        link.symlink_to(orig_rel)
                metric, top1_labels, top1_probs = exp.evaluate_classifier(
                    model, input=src_link, labels=label_link, **eval_args
                )

                log.info(metric.format(delim='\t'))

                test_dir.mkdir(parents=True, exist_ok=True)
                score_file = test_dir / f'{name}.score.tsv'
                score_file.write_text(metric.format(delim=','))

                out = '\n'.join(f'{l}\t{p:g}' for l, p in zip(top1_labels, top1_probs))

                out_file.write_text(out)

            except Exception as e:
                log.exception(f"Something went wrong with '{name}' test")
                err = test_dir / f'{name}.err'
                err.write_text(str(e))

    def run_translation_tests(self, exp=None, args=None):
        from rtg.nmt.decoder import Decoder

        exp = exp or self.exp
        args = args or exp.config['tester']
        suite: Dict[str, List] = args.get('suite')
        assert suite
        log.info(f"Found {len(suite)} suit :: {suite.keys()}")

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
                exp=exp, max_len=max_len, batch_size=batch_size, **tune_args
            )
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

        dec_args.update(
            dict(
                beam_size=beam_size,
                lp_alpha=lp_alpha,
                ensemble=ensemble,
                max_len=max_len,
                batch_size=batch_size,
            )
        )
        exp.persist_state()  # update the config

        assert step > 0, 'looks like no model is saved or invalid experiment dir'
        test_dir = exp.work_dir / f'test_step{step}_beam{beam_size}_ens{ensemble}_lp{lp_alpha}'
        log.info(f"Test Dir = {test_dir}")
        test_dir.mkdir(parents=True, exist_ok=True)

        decoder = Decoder.new(exp, ensemble=ensemble)
        for name, data in suite.items():
            # noinspection PyBroadException
            src, ref = data, None
            out_file = None
            if isinstance(data, list):
                src, ref = data[:2]
            elif isinstance(data, dict):
                src, ref = data['src'], data.get('ref')
                out_file = data.get('out')
            try:
                orig_src = Path(src).absolute()
                src_link = test_dir / f'{name}.src'
                ref_link = test_dir / f'{name}.ref'
                buffer = [(src_link, orig_src)]
                if ref:
                    orig_ref = Path(ref).absolute()
                    buffer.append((ref_link, orig_ref))
                for link, orig in buffer:
                    if not link.exists():
                        orig_rel = os.path.relpath(orig, link.parent)
                        link.symlink_to(orig_rel)
                out_file = test_dir / f'{name}.out.tsv' if not out_file else out_file
                out_file.parent.mkdir(parents=True, exist_ok=True)

                self.decode_eval_file(
                    decoder,
                    src_link,
                    out_file,
                    ref_link,
                    batch_size=eff_batch_size,
                    beam_size=beam_size,
                    lp_alpha=lp_alpha,
                    max_len=max_len,
                )
            except Exception as e:
                log.exception(f"Something went wrong with '{name}' test")
                err = test_dir / f'{name}.err'
                err.write_text(str(e))

    def run(self, run_tests=True, debug=debug_mode):
        if not self.exp.read_only:
            # if not distr.is_main:
            #    log.clear_console() # console handler
            log.update_file_handler(str(self.exp.log_file))
        self.pre_checks()  # fail early, so TG can fix and restart

        if dtorch.is_global_main:
            self.exp.pre_process()
        dtorch.barrier()
        if not self.exp.src_vocab or not self.exp.tgt_vocab:
            # if not self.exp.read_only:
            self.exp.reload()  # with updated config and vocabs from global_main
        assert self.exp.src_vocab and self.exp.tgt_vocab, "Vocabs are not loaded"
        # train on all
        if debug:
            log.warning(
                "<<<Anomaly detection enabled; this is very slow; use this only for debugging/hunting bugs>>>"
            )
            with torch.autograd.detect_anomaly():
                self.exp.train()
        else:
            self.exp.train()
        dtorch.barrier()
        if run_tests:
            if self.exp.problem_type in self.tests_types:
                if dtorch.is_global_main:
                    self.exp.reload()  # if user changed config for tests while training
                    with torch.no_grad():
                        self.tests_types[self.exp.problem_type]()
            else:
                log.warning(
                    f"{self.exp.problem_type} dont have test runner yet. "
                    f"Known runners: {self.tests_types}. Please fix me"
                )
