#!/usr/bin/env python
#
# Author: Thamme Gowda [tg (at) isi (dot) edu]
# Created: 3/9/19

import argparse
from pathlib import Path

import torch

from rtg import Pipeline, __version__, dtorch, load_conf, log, MODELS


def parse_args():
    parser = argparse.ArgumentParser(prog="rtg-pipeline", description="RTG Pipeline CLI",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-v', '--version', action='version', version=f'%(prog)s {__version__}')
    parser.add_argument("exp", metavar='EXP_DIR', help="Working directory of experiment", type=Path)
    parser.add_argument(
        "conf",
        metavar='conf.yml',
        type=Path,
        nargs='?',
        help="Config File. By default <work_dir>/conf.yml is used",
    )
    parser.add_argument(
        "-G", "--gpu-only", action="store_true", default=False, help="Crash if no GPU is available"
    )
    parser.add_argument("-fp16", "--fp16", action="store_true", default=False, help="Float 16")
    excl_parser = parser.add_mutually_exclusive_group()
    excl_parser.add_argument("-t", "--test", dest='run_tests', action="store_true", default=False,
                             help="Run test suite after training")
    excl_parser.add_argument("-nt", "--no-test", dest='run_tests', action="store_false", default=False,
                             help="Do not run test suite after training")
    

    # multi-gpu / multi-node
    parser.add_argument("--local_rank", "--local-rank", type=int, default=-1, help="Multi-GPU - Local rank")
    parser.add_argument("--master-port", type=int, default=-1, help="Master port (for multi-node SLURM jobs)")
    dtorch.setup()
    args = parser.parse_args()
    if args.fp16:
        assert torch.cuda.is_available(), "GPU required for fp16... exiting."
        dtorch.enable_fp16()

    if args.gpu_only:
        assert torch.cuda.is_available(), "No GPU found... exiting"
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            log.info(f'Cuda {i}: {torch.cuda.get_device_properties(i)}')

    conf_file: Path = args.conf if args.conf else args.exp / 'conf.yml'
    assert conf_file.exists(), f'NOT FOUND: {conf_file}'
    conf = load_conf(conf_file)
    model_type = conf['model_type']
    assert model_type in MODELS, f"Unknown model type: {model_type}. Available: {MODELS.keys()}"
    model_spec = MODELS.get(model_type)
    ExpFactory = model_spec.Experiment

    if conf.get('spark', {}):
        log.info("Big experiment mode enabled; checking pyspark backend")
        try:
            import pyspark

            log.info("pyspark is available")
        except:
            log.warning("unable to import pyspark. Please do 'pip install pyspark' and run again")
            raise
        from rtg.nmt.big.experiment import BigTranslationExperiment

        ExpFactory = BigTranslationExperiment

    read_only = not dtorch.is_global_main  # only main can modify experiment
    log.info(f"Experiment: {ExpFactory.__name__} (read_only={read_only})")
    exp = ExpFactory(args.exp, config=conf_file, read_only=read_only)
    dtorch.barrier()
    return exp, args


def main():
    exp, args = parse_args()
    pipe = Pipeline(exp=exp)
    pipe.run(run_tests=args.run_tests)


if __name__ == '__main__':
    main()
