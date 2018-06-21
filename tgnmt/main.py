#!/usr/bin/env python

import sys
import argparse
from argparse import ArgumentDefaultsHelpFormatter as ArgFormatter
from tgnmt import TranslationExperiment as Experiment

from tgnmt.module.t2t import Trainer, EncoderDecoder
from tgnmt.module.decoder import GreedyDecoder


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
    train.add_argument('-mt', '--mod-type', default='t2t', choices=['rnn', 't2t'],
                       help='Type of model: RNN or T2T (aka transformer)')
    train.add_argument("-ne", "--num-epochs", help="Num epochs", type=int, default=15)
    train.add_argument("-re", "--resume", action='store_true', dest='resume_train',
                       help="Resume Training. adds --num-epochs more epochs to the most recent model in work-dir", )
    train.add_argument("-bs", "--batch-size", help="Batch size", type=int, default=256)
    train.add_argument("-km", "--keep-models", type=int, default=4,
                       help="Number of models to keep. Stores one model per epoch")

    decode = tasks.add_parser('decode', formatter_class=ArgFormatter)
    decode.add_argument("-if", '--input', type=argparse.FileType('r'), default=sys.stdin,
                        help='Input file path. default is STDIN')
    decode.add_argument("-of", '--output', type=argparse.FileType('w'), default=sys.stdout,
                        help='Output File path. default is STDOUT')

    return p.parse_args()


def prep(exp: Experiment, args):
    return exp.pre_process(**args)


def train(exp: Experiment, args):
    assert exp.has_prepared(), f'Experiment dir {exp.work_dir} is not ready to train. Please run "prep" sub task'
    if args.get('mod_type') != 't2t':
        raise Exception('We are focusing on t2t at the moment!')
    trainer = Trainer(exp)
    return trainer.train(**args)


def decode(exp: Experiment, args):
    assert exp.has_trained(), f'Experiment dir {exp.work_dir} is not ready to decode. Please run "train" sub task'
    mod_args = exp.get_model_args()
    last_check_pt, _ = exp.get_last_saved_model()
    decoder = GreedyDecoder(exp, EncoderDecoder.make_model, args=mod_args, check_pt_file=last_check_pt)
    return decoder.decode_file(args.pop('input'), args.pop('output'))


def main():
    args = vars(parse_args())
    exp = Experiment(args.pop('work_dir'))
    task = args.pop('task')
    {
        'prep': prep,
        'train': train,
        'decode': decode
    }[task](exp, args)


if __name__ == '__main__':
    main()
