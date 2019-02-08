# CLI interface tp train sub task

import argparse
from argparse import ArgumentDefaultsHelpFormatter as ArgFormatter

from rtg import TranslationExperiment as Experiment, log
from rtg.module.tfmnmt import TransformerTrainer
from rtg.module.rnnmt import SteppedRNNMTTrainer
from rtg.binmt.bicycle import BiNmtTrainer
from rtg.lm.rnnlm import RnnLmTrainer
from rtg.lm.tfmlm import TfmLmTrainer
from rtg.utils import log_tensor_sizes, Optims


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
    parser.add_argument("-bs", "--batch-size", help="Mini batch size of training and validation",
                        type=int, default=256)
    parser.add_argument("-op", "--optim", type=str, default='ADAM', choices=Optims.names(),
                        help="Name of optimizer")
    parser.add_argument("-oa", "--optim-args", type=str, default='lr=0.001',
                        help="Comma separated key1=val1,key2=val2 args to optimizer."
                             " Example: lr=0.01,warmup_steps=1000 "
                             "The arguments depends on the choice of --optim")

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
    _, optim_args = exp.optim_args
    if optim_args is None:
        optim_args = {}
    if args.get('optim_args'):
        # convert key1=val1,key2=val2 format to dictionary
        pairs = [x.strip() for x in args.pop('optim_args').split(',')]
        pairs = [pair.split('=') for pair in pairs if pair]
        optim_args.update({k.strip(): float(v) for k, v in pairs})

    trainer = {
        't2t': TransformerTrainer,
        'binmt': BiNmtTrainer,
        'seq2seq': SteppedRNNMTTrainer,
        'tfmnmt': TransformerTrainer,
        'rnnmt': SteppedRNNMTTrainer,
        'rnnlm': RnnLmTrainer,
        'tfmlm': TfmLmTrainer
    }[exp.model_type](exp, optim=args.pop('optim'), **optim_args)
    try:
        trainer.train(**args)
    except RuntimeError as e:
        if 'out of memory' in str(e).lower():
            log_tensor_sizes()
        raise e


if __name__ == '__main__':
    main()

