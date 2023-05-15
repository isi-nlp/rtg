# CLI interface to decode task
import argparse
import sys
from argparse import ArgumentDefaultsHelpFormatter as ArgFormatter
import torch
from pathlib import Path

from rtg import TranslationExperiment as Experiment, log, yaml
from rtg.module.decoder import Decoder, ReloadEvent
from rtg.utils import IO


def parse_args():
    parser = argparse.ArgumentParser(prog="rtg.decode", description="Decode using NMT model",
                                     formatter_class=ArgFormatter)
    parser.add_argument("work_dir", help="Working directory", type=str)
    parser.add_argument("model_path", type=str, nargs='*',
                        help="Path to model's checkpoint. "
                             "If not specified, a best model (based on the score on validation set)"
                             " from the experiment directory will be used."
                             " If multiple paths are specified, then an ensembling is performed by"
                             " averaging the param weights")
    parser.add_argument("-if", '--input', default=sys.stdin,
                        type=argparse.FileType('r', encoding='utf-8', errors='ignore'),
                        help='Input file path. default is STDIN')
    parser.add_argument("-of", '--output', default=sys.stdout,
                        type=argparse.FileType('w', encoding='utf-8', errors='ignore'),
                        help='Output File path. default is STDOUT')
    parser.add_argument("-bs", '--beam-size', type=int, default=5,
                        help='Beam size. beam_size=1 is greedy, '
                             'In theory: higher beam is better approximation but expensive. '
                             'But in practice, higher beam doesnt always increase.')
    parser.add_argument("-bc", '--batch-size', type=int, default=1,
                        help='Number of source tokens in a batch, approximately. '
                             'tries to fit in atleast one sentence => so even if you set 0 or 1, '
                             'there will be atleast one sentence in batch. '
                             '1 sentence seems better in CPU but larger number is better on GPUs')
    parser.add_argument("-lp", '--lp-alpha', type=float, default=0.6,
                        help='Length penalty alpha. to disable set <= 0.0 '
                             'Ideally in the range [0.0, 1.0] but you are allowed to '
                             'experiment beyond > 1.0 but not less than 0.0')
    parser.add_argument("-ml", '--max-len', type=int, default=60,
                        help='Maximum output sequence length. '
                             'Example: if max_len=10 and if source_len is 50, '
                             'then decoder goes up to 50+10 time steps in search of EOS token.')
    parser.add_argument("-msl", '--max-src-len', type=int,
                        help='max source len; longer seqs will be truncated')
    parser.add_argument("-nh", '--num-hyp', type=int, default=1,
                        help='Number of hypothesis to output. This should be smaller than beam_size')
    parser.add_argument("--prepared", dest="prepared", action='store_true', default=None,
                        help='Each token is a valid integer which is an index to embedding,'
                             ' so skip indexifying again')
    parser.add_argument("-bp", '--binmt-path', type=str, default=None,
                        choices=['E1D1', 'E2D2', 'E1D2E2D1', 'E2D2E1D2', 'E1D2', 'E2D1'],
                        help='Sub module path inside BiNMT. applicable only when model is BiNMT')
    parser.add_argument("-it", '--interactive', action='store_true',
                        help='Open interactive shell with decoder')
    parser.add_argument("-sc", '--skip-check', action='store_true',
                        help='Skip Checking whether the experiment dir is prepared and trained')

    parser.add_argument("-en", '--ensemble', type=int, default=1,
                        help='Ensemble best --ensemble models by averaging them')

    parser.add_argument("-cb", '--sys-comb', type=Path,
                        help='System combine models at the softmax layer using the weights'
                             ' specified in this file. When this argument is supplied, model_path '
                             'argument is ignored.')
    args = vars(parser.parse_args())
    return args


def validate_args(args, exp: Experiment):
    if not args.pop('skip_check'):  # if --skip-check is not requested
        assert exp.has_prepared(), \
            f'Experiment dir {exp.work_dir} is not ready to train. Please run "prep" sub task'
        assert exp.has_trained(), \
            f'Experiment dir {exp.work_dir} is not ready to decode.' \
            f' Please run "train" sub task or --skip-check to ignore this'

    weights_file = exp.work_dir / 'combo-weights.yml'
    if not args.get('sys_comb') and weights_file.exists():
        log.warning("Found default combo weights, switching to combo mode")
        args['sys_comb'] = weights_file

    if args.get("sys_comb"):
        with IO.reader(args['sys_comb']) as fh:
            weights = yaml.load(fh)['weights']
            args['model_path'], args['weights'] = zip(*weights.items())
            for model in args['model_path']:
                assert Path(model).exists(), model
            assert abs(sum(args['weights']) - 1) < 1e-3, \
                f'Weights from --sys-comb file should sum to 1.0, given={args["weights"]}'


def main():
    # No grads required
    torch.set_grad_enabled(False)
    args = parse_args()
    gen_args = {}
    exp = Experiment(args.pop('work_dir'), read_only=True)
    validate_args(args, exp)

    if exp.model_type == 'binmt':
        if not args.get('path'):
            Exception('--binmt-path argument is needed for BiNMT model.')
        gen_args['path'] = args.pop('binmt_path')

    weights = args.get('weights')
    if weights:
        decoder = Decoder.combo_new(exp, model_paths=args.pop('model_path'),
                                    weights=weights)
    else:
        decoder = Decoder.new(exp, gen_args=gen_args, model_paths=args.pop('model_path', None),
                              ensemble=args.pop('ensemble', 1))
    if args.pop('interactive'):
        if weights:
            log.warning("Interactive shell not reloadable for combo mode. FIXME: TODO:")
        if args['input'] != sys.stdin or args['output'] != sys.stdout:
            log.warning('--input and --output args are not applicable in --interactive mode')
        args.pop('input')
        args.pop('output')

        while True:
            try:
                # an hacky way to unload and reload model when user tries to switch models
                decoder.decode_interactive(**args)
                break  # exit loop if there is no request for reload
            except ReloadEvent as re:
                decoder = Decoder.new(exp, gen_args=gen_args, model_paths=re.model_paths)
                args = re.state
                # go back to loop and redo interactive shell
    else:
        return decoder.decode_file(args.pop('input'), args.pop('output'), **args)


if __name__ == '__main__':
    main()
