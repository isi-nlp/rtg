# CLI interface to decode task
import argparse
import sys
from argparse import ArgumentDefaultsHelpFormatter as ArgFormatter
import torch

from rtg import TranslationExperiment as Experiment, log
from rtg.module.decoder import Decoder, ReloadEvent


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
    parser.add_argument("-ml", '--max-len', type=int, default=100,
                        help='Maximum output sequence length')
    parser.add_argument("-nh", '--num-hyp', type=int, default=1,
                        help='Number of hypothesis to output. This should be smaller than beam_size')
    parser.add_argument("--prepared", dest="prepared", action='store_true',
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

    parser.add_argument("-w", '--weight', type=float, nargs='*',
                        help='System combine models at the softmax layer using these weights. ' 
                             'This argument must be accompanied with "model_path" argument and '
                             'must contain at least two model_path.')
    args = vars(parser.parse_args())
    return args


def validate_args(args, exp: Experiment):
    if exp.model_type == 'combo':
        assert args.get('model_path') and args.get('weight'), \
            'combo mode type requires --weight and model_path arguments (at least two each)'
        assert len(args['weight']) == len(args['model_path']), \
            f"There should be one --weight per model_path. Given {len(args['weight'])} weights " \
            f"for {len(args['model_path'])} models"

    if args.get('weight'):
        assert exp.model_type == 'combo', \
            f'Only valid for combo experiments. but found {exp.model_type}'
        assert 'model_path' in args, 'model_path argument is needed'
        assert len(args['model_path']) > 1, 'at least two models must be given'
        assert len(args['model_path']) == len(args['weight']),\
            f"There should be one weight per model; {len(args['model_path'])} models" \
            f" and {len(args['weight'])} weights are specified"
        assert abs(sum(args['weight']) - 1) < 1e-3,\
            f'Weights should sum to 1.0, given={args["weight"]}'

    if not args.pop('skip_check'):  # if --skip-check is not requested
        assert exp.has_prepared(), \
            f'Experiment dir {exp.work_dir} is not ready to train. Please run "prep" sub task'
        assert exp.has_trained(), \
            f'Experiment dir {exp.work_dir} is not ready to decode.' \
            f' Please run "train" sub task or --skip-check'


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

    combo_mode = exp.model_type == 'combo'
    if combo_mode:
        decoder = Decoder.combo_new(exp, model_paths=args.pop('model_path'),
                                    weights=args['weight'])
    else:
        decoder = Decoder.new(exp, gen_args=gen_args, model_paths=args.pop('model_path', None),
                              ensemble=args.pop('ensemble', 1))
    if args.pop('interactive'):
        if combo_mode:
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
