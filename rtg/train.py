# CLI interface tp train sub task

import argparse
from argparse import ArgumentDefaultsHelpFormatter as ArgFormatter
from rtg import TranslationExperiment as Experiment, log


def parse_args():
    parser = argparse.ArgumentParser(prog="rtg.train", description="Train NMT model",
                                     formatter_class=ArgFormatter)
    parser.add_argument("work_dir", help="Working directory", type=str)
    parser.add_argument("-rs", "--seed", help="Seed for random number generator. Set it to zero "
                                              "to not touch this part.", type=int, default=0)
    parser.add_argument("-st", "--steps", help="Total steps", type=int, default=128000)
    parser.add_argument("-cp", "--check-point", help="Store model after every --check-point steps",
                        type=int, default=1000)
    parser.add_argument("-km", "--keep-models", type=int, default=10,
                        help="Number of checkpoints to keep.")
    parser.add_argument("-bs", "--batch-size",
                        help="Mini batch size (# tokens on target side) of training and validation."
                             " A token can be subword piece.",
                        type=int, default=2048)

    parser.add_argument("-ft", "--fine-tune", action='store_true',
                        help="Use fine tune corpus instead of train corpus.")
    return vars(parser.parse_args())


def main():
    args = parse_args()
    seed = args.pop("seed")
    if seed:
        log.info(f"Seed for random number generator: {seed}")
        import random
        import torch
        random.seed(seed)
        torch.manual_seed(seed)

    exp = Experiment(args.pop('work_dir'))
    assert exp.has_prepared(), f'Experiment dir {exp.work_dir} is not ready to train. ' \
                               f'Please run "prep" sub task'
    exp.train(args)


if __name__ == '__main__':
    main()

