# CLI interface to decode task
import argparse
import sys
from argparse import ArgumentDefaultsHelpFormatter as ArgFormatter
import torch

from rtg import TranslationExperiment as Experiment, log
from rtg.module.decoder import Decoder


def parse_args():
    parser = argparse.ArgumentParser(prog="rtg.decode", description="Decode using NMT model",
                                     formatter_class=ArgFormatter)
    parser.add_argument("exp_dir", help="Experiment directory", type=str)
    parser.add_argument("-if", '--input', default=[sys.stdin], nargs='*',
                        type=argparse.FileType('r', encoding='utf-8', errors='ignore'),
                        help='Input file path. default is STDIN')
    parser.add_argument("-of", '--output', default=[sys.stdout], nargs='*',
                        type=argparse.FileType('w', encoding='utf-8', errors='ignore'),
                        help='Output File path. default is STDOUT')
    parser.add_argument("-sc", '--skip-check', action='store_true',
                        help='Skip Checking whether the experiment dir is prepared and trained')
    parser.add_argument("-b", '--batch-size', type=int, help='batch size for 1 beam. effective_batch = batch_size/beam_size')
    parser.add_argument("-msl", '--max-src-len', type=int, help='max source len; longer seqs will be truncated')    
    args = vars(parser.parse_args())
    return args


def validate_args(cli_args, conf_args, exp: Experiment):
    if not cli_args.pop('skip_check'):  # if --skip-check is not requested
        assert exp.has_prepared(), \
            f'Experiment dir {exp.work_dir} is not ready to train. Please run "prep" sub task'
        assert exp.has_trained(), \
            f'Experiment dir {exp.work_dir} is not ready to decode.' \
                f' Please run "train" sub task or --skip-check to ignore this'
    assert len(cli_args['input']) == len(cli_args['output'])
    if cli_args.get('batch_size'):
        batch_size = cli_args['batch_size'] / conf_args.get('beam_size', 1)
        log.info(f"Batch size is {batch_size}")
        conf_args['batch_size'] = batch_size
    if cli_args.get('max_src_len'):
        conf_args['max_src_len'] = cli_args['max_src_len']


def main():
    # No grads required for decode
    torch.set_grad_enabled(False)
    cli_args = parse_args()
    exp = Experiment(cli_args.pop('exp_dir'), read_only=True)
    dec_args = exp.config.get('decoder') or exp.config['tester'].get('decoder', {})
    validate_args(cli_args, dec_args, exp)

    decoder = Decoder.new(exp, ensemble=dec_args.pop('ensemble', 1))
    for inp, out in zip(cli_args['input'], cli_args['output']):
        log.info(f"Decode :: {inp} -> {out}")
        try:
            return decoder.decode_file(inp, out, **dec_args)
        except:
            log.exception(f"Decode failed for {inp}")


if __name__ == '__main__':
    main()
