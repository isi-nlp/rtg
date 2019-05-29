#!/usr/bin/env python
#
# Author: Thamme Gowda [tg (at) isi (dot) edu] 
# Created: 2/8/19
import argparse
import sys
import torch
from rtg.module.decoder import Decoder
from rtg import TranslationExperiment as Experiment, device
from typing import TextIO
from tqdm import tqdm
import math


def log_perplexity(decoder: Decoder, test_data: TextIO):
    """
    Computes log perplexity of a language model on a given test data
    :param decoder:
    :param test_data:
    :return:

    .. math::
        P(w_i | h) <-- probability of word w_i given history h
        P(w_1, w_2, ... w_N) <-- probability of observing or generating a word sequence
        P(w_1, w_2, ... w_N) = P(w_1) x P(w_2|w_1) x P(w_3 | w_1, w_2) ... <-- chain rule

        PP_M <-- Perplexity of a Model M
        PP_M(w_1, w_2, ... w_N) <-- PP_M on a sequence w_1, w_2, ... w_N
        PP_M(w_1, w_2, ... w_N) = P(w_1, w_2, ... w_N)^{-1/N}

        log(PP_M) <-- Log perplexity of a model M
        log(PP_M) = -1/N \sum_{i=1}^{i=N} P(w_i | w_1, w_2 .. w_{i-1})


    Note: log perplexity is a practical solution to deal with floating point underflow
    """
    lines = (line.strip() for line in test_data)
    test_seqs = [decoder.out_vocab.encode_as_ids(line, add_bos=True, add_eos=True)
                 for line in lines]
    count = 0
    total = 0.0
    for seq in tqdm(test_seqs, dynamic_ncols=True):
        #  batch of 1
        # TODO: make this faster using bigger batching
        batch = torch.tensor(seq, dtype=torch.long, device=device).view(1, -1)
        for step in range(1, len(seq)):
            # assumption: BOS and EOS are included
            count += 1
            history = batch[:, :step]
            word_idx = seq[step]
            log_prob = decoder.next_word_distr(history)[0, word_idx]
            total += log_prob
    log_pp = -1/count * total
    return log_pp.item()


def parse_args():
    parser = argparse.ArgumentParser(prog='rtg.eval.perplexity',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("work_dir", help="Working/Experiment directory", type=str)
    parser.add_argument("model_path", type=str, nargs='*',
                        help="Path to model's checkpoint. "
                             "If not specified, a best model (based on the score on validation set)"
                             " from the experiment directory will be used."
                             " If multiple paths are specified, then an ensembling is performed by"
                             " averaging the param weights")
    parser.add_argument("-t", '--test', default=sys.stdin,
                        type=argparse.FileType('r', encoding='utf-8', errors='ignore'),
                        help='test file path. default is STDIN')
    parser.add_argument("-en", '--ensemble', type=int, default=1,
                        help='Ensemble best --ensemble models by averaging them')

    args = vars(parser.parse_args())
    return args


def main():
    # No grads required
    torch.set_grad_enabled(False)
    args = parse_args()
    gen_args = {}
    exp = Experiment(args.pop('work_dir'), read_only=True)

    assert exp.model_type.endswith('lm'), 'Only for Language models'
    decoder = Decoder.new(exp, gen_args=gen_args, model_paths=args.pop('model_path', None),
                          ensemble=args.pop('ensemble', 1))

    log_pp = log_perplexity(decoder, args['test'])
    print(f'Log perplexity: {log_pp:g}')
    print(f'Perplexity: {math.exp(log_pp):g}')


if __name__ == '__main__':
    main()
