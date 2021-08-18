# CLI interface to decode task
import argparse
import sys
import io
from argparse import ArgumentDefaultsHelpFormatter as ArgFormatter
import torch
from typing import List, TextIO
import copy
from pathlib import Path
from rtg import log
from rtg.exp import load_conf
from rtg.module.decoder import Decoder
from rtg.registry import registry, MODEL
from rtg.emb.tfmcls import ClassificationExperiment
from rtg.exp import TranslationExperiment


def parse_args():

    stdin = io.TextIOWrapper(sys.stdin.buffer, encoding='utf-8', errors='ignore', newline='\n')
    stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='ignore')
    parser = argparse.ArgumentParser(prog="rtg.decode", description="Decode using NMT model",
                                     formatter_class=ArgFormatter)
    parser.add_argument("exp_dir", help="Experiment directory", type=str)
    parser.add_argument("-if", '--input', default=[stdin], nargs='*',
                        type=argparse.FileType('r', encoding='utf-8', errors='ignore'),
                        help='Input file path. default is STDIN')
    parser.add_argument("-of", '--output', default=[stdout], nargs='*',
                        type=argparse.FileType('w', encoding='utf-8', errors='ignore'),
                        help='Output File path. default is STDOUT')
    parser.add_argument("-sc", '--skip-check', action='store_true',
                        help='Skip Checking whether the experiment dir is prepared and trained')
    parser.add_argument("-b", '--batch-size', type=int,
                        help='batch size for 1 beam. effective_batch = batch_size/beam_size')
    parser.add_argument("-msl", '--max-src-len', type=int,
                        help='max source len; longer seqs will be truncated')
    parser.add_argument("-nb", '--no-buffer', action='store_true',
                        help='Processes one line per batch followed by flush output')
    args = vars(parser.parse_args())
    return args


def validate_args(cli_args, conf_args, exp):
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


def decode_mt(exp, **cli_args):
    dec_args = exp.config.get('decoder') or exp.config['tester'].get('decoder', {})
    validate_args(cli_args, dec_args, exp)
    input: List[TextIO] = cli_args.pop('input')
    output: List[TextIO] = cli_args.pop('output')
    decoder = Decoder.new(exp, ensemble=dec_args.pop('ensemble', 1))
    for inp, out in zip(input, output):
        log.info(f"Decode :: {inp} -> {out}")
        try:
            if cli_args.get('no_buffer'):
                return decoder.decode_stream(inp, out, **dec_args)
            else:
                return decoder.decode_file(inp, out, **dec_args)
        except:
            log.exception(f"Decode failed for {inp}")


def predict_cls(exp: ClassificationExperiment, **cli_args):
    conf_args = copy.copy(exp.config.get('tester', {}))
    batch_size = cli_args.get('batch_size', None) or conf_args.get('batch_size', None)
    max_len = cli_args.get('max_src_len', 0) or conf_args.get('max_len', 0)
    assert batch_size
    assert max_len > 0
    assert not cli_args.get('no_buffer'), 'Option --no-buffer is not yet supported for this model.'
    model = exp.load_model()
    for in_stream, out_stream in zip(cli_args['input'], cli_args['output']):
        input = [line.strip() for line in in_stream]
        log.info(f"going to label {len(input)} sequences; batch_size={batch_size} max_len={max_len}")
        top1_idx, top1_label, top1_prob = exp.get_predictions(
            model, input=input, batch_size=batch_size, max_len=max_len)
        for label, prob in zip(top1_label, top1_prob):
            out_stream.write(f'{label}\t{prob:g}\n')
        log.info(f"Wrote to {out_stream}")
    log.info("===All done!===")


def main(**args):
    # No grads required for decode
    torch.set_grad_enabled(False)
    cli_args = args or parse_args()
    exp_dir = Path(cli_args.pop('exp_dir'))
    conf = load_conf(exp_dir / 'conf.yml')
    assert conf.get('model_type')
    exp_factory = TranslationExperiment
    if conf['model_type'] in registry[MODEL]:
        exp_factory = registry[MODEL][conf['model_type']].Experiment
    exp = exp_factory(exp_dir, config=conf, read_only=True)
    if isinstance(exp, ClassificationExperiment):
        predict_cls(exp, **cli_args)
    else:
        assert isinstance(exp, TranslationExperiment)
        decode_mt(exp, **cli_args)


if __name__ == '__main__':
    main()
