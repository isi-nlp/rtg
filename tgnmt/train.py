# CLI interface tp train sub task

import argparse
from argparse import ArgumentDefaultsHelpFormatter as ArgFormatter

from tgnmt import TranslationExperiment as Experiment
from tgnmt.module.t2t import Trainer


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
    assert exp.has_prepared(), f'Experiment dir {exp.work_dir} is not ready to train. Please run "prep" sub task'
    if args.get('mod_type') != 't2t':
        raise Exception('We are focusing on t2t at the moment!')
    trainer = Trainer(exp)
    return trainer.train(**args)


if __name__ == '__main__':
    main()
