# CLI interface to decode task
import sys
import argparse
from argparse import ArgumentDefaultsHelpFormatter as ArgFormatter

from tgnmt import TranslationExperiment as Experiment
from tgnmt.module.t2t import EncoderDecoder
from tgnmt.module.decoder import GreedyDecoder


def parse_args():
    parser = argparse.ArgumentParser(prog="tgnmt.decode", description="Decode using NMT model",
                                     formatter_class=ArgFormatter)
    parser.add_argument("work_dir", help="Working directory", type=str)

    parser.add_argument("-if", '--input', type=argparse.FileType('r'), default=sys.stdin,
                        help='Input file path. default is STDIN')
    parser.add_argument("-of", '--output', type=argparse.FileType('w'), default=sys.stdout,
                        help='Output File path. default is STDOUT')

    return vars(parser.parse_args())


def main():
    args = parse_args()
    exp = Experiment(args.pop('work_dir'))
    assert exp.has_prepared(), f'Experiment dir {exp.work_dir} is not ready to train. Please run "prep" sub task'
    assert exp.has_trained(), f'Experiment dir {exp.work_dir} is not ready to decode. Please run "train" sub task'
    mod_args = exp.get_model_args()
    last_check_pt, _ = exp.get_last_saved_model()
    decoder = GreedyDecoder(exp, EncoderDecoder.make_model, args=mod_args, check_pt_file=last_check_pt)
    return decoder.decode_file(args.pop('input'), args.pop('output'))


if __name__ == '__main__':
    main()
