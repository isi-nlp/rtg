#!/usr/bin/env python

import argparse
from argparse import ArgumentDefaultsHelpFormatter as ArgFormatter
from tgnmt import TranslationExperiment as Experiment
from tgnmt.seq2seq import Trainer as Seq2SeqTrainer


def parse_args():

    p = argparse.ArgumentParser(prog="tgnmt", description="Yet Another NMT", formatter_class=ArgFormatter)

    p.add_argument("work_dir", help="Working directory", type=str)
    tasks = p.add_subparsers(help='Sub tasks', dest='task')
    tasks.required = True

    prep = tasks.add_parser('prep', formatter_class=ArgFormatter)
    prep.add_argument("-tf", '--train-file', help="Training File.", type=str, required=True)
    prep.add_argument("-vf", '--valid-file', help="Validation File.", type=str, required=True)
    prep.add_argument("-sl", '--src-len', type=int, default=200,
                      help="Truncate or filter source sentences to this length", )
    prep.add_argument("-tl", '--tgt-len', type=int, default=200,
                      help="Truncate or filter target sentences to this length")
    prep.add_argument("-tr", '--truncate', action='store_true',
                      help="Do select all training sentences and truncate them to --src-len and --tgt-len values."
                           " Default is to exclude sentences longer than --src-len and --tgt-len")

    train = tasks.add_parser('train', formatter_class=ArgFormatter)
    train.add_argument("-ne", "--num-epochs", help="Num epochs", type=int, default=15)
    train.add_argument("-re", "--resume", action='store_true', dest='resume_train',
                       help="Resume Training. adds --num-epochs more epochs to the most recent model in work-dir",)
    train.add_argument("-bs", "--batch-size", help="Batch size", type=int, default=256)
    train.add_argument("-km", "--keep-models", type=int, default=4,
                       help="Number of models to keep. Stores one model per epoch")
    return p.parse_args()


def main():
    args = vars(parse_args())
    exp = Experiment(args.pop('work_dir'))
    task = args.pop('task')
    if task == 'prep':
        exp.pre_process(**args)
    elif task == 'train':
        assert exp.has_prepared(), f'Experiment dir {exp.work_dir} is not ready to train. Please run "prep" sub task'
        trainer = Seq2SeqTrainer(exp)
        trainer.train(**args)


if __name__ == '__main__':
    main()
