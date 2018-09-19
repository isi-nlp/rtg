# CLI interface to decode task
import argparse
import sys
from argparse import ArgumentDefaultsHelpFormatter as ArgFormatter

from rtg import TranslationExperiment as Experiment, log
from rtg.module.decoder import Decoder, ReloadEvent


def parse_args():
    parser = argparse.ArgumentParser(prog="rtg.decode", description="Decode using NMT model",
                                     formatter_class=ArgFormatter)
    parser.add_argument("work_dir", help="Working directory", type=str)
    parser.add_argument("model_path", type=str, nargs='?',
                        help="Path to model's checkpoint. "
                             "If not specified, a best model (based on the score on validation set)"
                             " from the experiment directory will be used")
    parser.add_argument("-if", '--input', type=argparse.FileType('r'), default=sys.stdin,
                        help='Input file path. default is STDIN')
    parser.add_argument("-of", '--output', type=argparse.FileType('w'), default=sys.stdout,
                        help='Output File path. default is STDOUT')
    parser.add_argument("-bs", '--beam-size', type=int, default=5,
                        help='Beam size. beam_size=1 is greedy, higher beam is better approximation but expensive')
    parser.add_argument("-ml", '--max-len', type=int, default=100,
                        help='Maximum output sequence length')
    parser.add_argument("-nh", '--num-hyp', type=int, default=1,
                        help='Number of hypothesis to output. This should be smaller than beam_size')
    parser.add_argument("--prepared", dest="prepared", action='store_true',
                        help='Each token is a valid integer wich is an index to embedding, so skip indexifying again')
    parser.add_argument("-bp", '--binmt-path', type=str, default=None,
                        choices=['E1D1', 'E2D2', 'E1D2E2D1', 'E2D2E1D2', 'E1D2', 'E2D1'],
                        help='Sub module path inside BiNMT. applicable only when model is BiNMT')
    parser.add_argument("-it", '--interactive', action='store_true',
                        help='Open interactive shell with decoder')
    parser.add_argument("-sc", '--skip-check', action='store_true',
                        help='Skip Checking whether the experiment dir is prepared and trained')
    return vars(parser.parse_args())


def main():
    args = parse_args()
    gen_args = {}

    exp = Experiment(args.pop('work_dir'), read_only=True)
    if exp.model_type == 'binmt':
        if not args.get('path'):
            Exception('--binmt-path argument is needed for BiNMT model.')
        gen_args['path'] = args.pop('binmt_path')

    if not args.pop('skip_check'):  # if --skip-check is not requested
        assert exp.has_prepared(), f'Experiment dir {exp.work_dir} is not ready to train. Please run "prep" sub task'
        assert exp.has_trained(), f'Experiment dir {exp.work_dir} is not ready to decode. Please run "train" sub task'

    decoder = Decoder.new(exp, gen_args=gen_args, model_path=args.pop('model_path', None))
    if args.pop('interactive'):
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
                decoder = Decoder.new(exp, gen_args=gen_args, model_path=re.model_path)
                args = re.state
                # go back to loop and redo interactive shell
    else:
        return decoder.decode_file(args.pop('input'), args.pop('output'), **args)


if __name__ == '__main__':
    main()
