# CLI interface tp train sub task

import argparse
from argparse import ArgumentDefaultsHelpFormatter as ArgFormatter

from rtg import TranslationExperiment as Experiment, log
from rtg.module.t2t import T2TTrainer
from rtg.binmt.model import SteppedSeq2SeqTrainer
from rtg.binmt.bicycle import BiNmtTrainer
from rtg.utils import log_tensor_sizes, Optims


def parse_args():
    parser = argparse.ArgumentParser(prog="rtg.train", description="Train NMT model",
                                     formatter_class=ArgFormatter)
    parser.add_argument("work_dir", help="Working directory", type=str)
    parser.add_argument("-rs", "--seed", help="Seed for random number generator. Set it to zero "
                                              "to not touch this part.", type=int, default=0)
    parser.add_argument("-st", "--steps", help="Total steps", type=int, default=12800)
    parser.add_argument("-re", "--resume", action='store_true', dest='resume_train',
                        help="Resume Training. adds --num-epochs more epochs to the most "
                             "recent model in work-dir", )
    parser.add_argument("-bs", "--batch-size", help="Batch size", type=int, default=256)
    parser.add_argument("-km", "--keep-models", type=int, default=10,
                        help="Number of models to keep. Stores one model per epoch")
    parser.add_argument("-op", "--optim", type=str, default='ADAM', choices=Optims.names(),
                        help="Name of optimizer")
    parser.add_argument("-oa", "--optim-args", type=str, default='lr=0.001',
                        help="Comma separated key1=val1,key2=val2 args to optimizer."
                             " Example: lr=0.001,warmup_steps=1000,step_size=1024. "
                             "The arguments depends on the choice of --optim")
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
        't2t': T2TTrainer,
        'binmt': BiNmtTrainer,
        'seq2seq': SteppedSeq2SeqTrainer,
    }[exp.model_type](exp, optim=args.pop('optim'), **optim_args)
    try:
        trainer.train(**args)
    except RuntimeError as e:
        if 'out of memory' in str(e).lower():
            log_tensor_sizes()
        raise e


if __name__ == '__main__':
    main()

