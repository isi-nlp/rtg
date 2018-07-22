# CLI interface tp train sub task

import argparse
from argparse import ArgumentDefaultsHelpFormatter as ArgFormatter

from tgnmt import TranslationExperiment as Experiment
from tgnmt.module.rnn import RNNTrainer as RNNTrainer
from tgnmt.module.t2t import T2TTrainer
from tgnmt.utils import log_tensor_sizes


def parse_args():
    parser = argparse.ArgumentParser(prog="tgnmt.train", description="Train NMT model",
                                     formatter_class=ArgFormatter)
    parser.add_argument("work_dir", help="Working directory", type=str)
    parser.add_argument('-mt', '--mod-type', default='t2t', choices=['rnn', 't2t'],
                        help='Type of model: RNN or T2T (aka transformer)')
    parser.add_argument("-ne", "--num-epochs", help="Num epochs", type=int, default=15)
    parser.add_argument("-re", "--resume", action='store_true', dest='resume_train',
                        help="Resume Training. adds --num-epochs more epochs to the most recent model in work-dir", )
    parser.add_argument("-bs", "--batch-size", help="Batch size", type=int, default=256)
    parser.add_argument("-km", "--keep-models", type=int, default=4,
                        help="Number of models to keep. Stores one model per epoch")
    return vars(parser.parse_args())


def main():
    args = parse_args()
    exp = Experiment(args.pop('work_dir'))
    mod_type = args.pop('mod_type')
    assert exp.has_prepared(), f'Experiment dir {exp.work_dir} is not ready to train. Please run "prep" sub task'
    if exp.has_trained() and exp.model_type and exp.model_type != mod_type:
        raise Exception(f'Experiment {exp.work_dir} was previously trained with model type "{exp.model_type}". '
                        f'Please clear models or start a new experiment to train {mod_type}. Or use {exp.model_type}')
    elif exp.model_type != mod_type:
        exp.model_type = mod_type
        exp.store_config()

    trainer = {'t2t': T2TTrainer, 'rnn': RNNTrainer}[mod_type](exp)
    try:
        trainer.train(**args)
    except RuntimeError as e:
        if 'out of memory' in str(e).lower():
            log_tensor_sizes()
        raise e


if __name__ == '__main__':
    main()
