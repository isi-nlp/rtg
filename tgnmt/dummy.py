#!/usr/bin/env python
# Generates dummy data for testing
import random
from tgnmt.dataprep import RESERVED_TOKS, Field
from tgnmt import TranslationExperiment
import argparse


class DataGen:

    def __init__(self, min_tok=len(RESERVED_TOKS), max_tok=16, min_len=1, max_len=10,
                 num_train_exs=1000, num_val_exs=50):
        """
        Dummy Dataset for quick testing
        :param min_tok: smallest legal token
        :param max_tok: largest legal token
        :param min_len: minimum possible sequence length
        :param max_len: maximum allowed sequence length
        :param num_train_exs: number of training examples
        :param num_val_exs: number of validation examples
        """
        assert min_tok < max_tok
        assert min_len <= max_len
        self.min_tok = min_tok
        self.max_tok = max_tok
        self.num_train_exs = num_train_exs
        self.num_val_exs = num_val_exs
        self.min_len = min_len
        self.max_len = max_len

    def make_seqs(self, num_exs):
        for _ in range(num_exs):
            seq_len = random.randint(self.min_len, self.max_len)
            seq = [random.randint(self.min_tok, self.max_tok) for _ in range(seq_len)]
            yield seq

    def make_bi_text(self, num_exs):
        seqs = (list(map(str, seq)) for seq in self.make_seqs(num_exs))
        bi_text = ((seq, seq) for seq in seqs)
        yield from bi_text

    def prepare_experiment(self, exp: TranslationExperiment):
        exp.src_vocab, exp.tgt_vocab = Field('src'), Field('tgt')
        for tok in range(self.min_tok, self.max_tok+1):
            exp.src_vocab.add_token(str(tok), -1)
            exp.tgt_vocab.add_token(str(tok), -1)

        exp.prep_file(self.make_bi_text(self.num_train_exs), exp.train_file)
        exp.prep_file(self.make_bi_text(self.num_val_exs), exp.valid_file)
        exp.persist_state()


if __name__ == '__main__':
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                description="Prepares a dummy machine translation experiment with integer sequences")
    p.add_argument("work_dir", help="experiment directory")
    p.add_argument('--max-tok', type=int, default=16, help='Largest token')
    p.add_argument('--max-len', type=int, default=10, help='Maximum sentence length')
    p.add_argument('--min-len', type=int, default=1, help='Minimum sentence length')
    p.add_argument('--num-train-exs', type=int, default=1000, help='Number of Training examples')
    p.add_argument('--num-val-exs', type=int, default=50, help='Number of Validation examples')

    args = vars(p.parse_args())
    work_dir = args.pop('work_dir')
    gen = DataGen(**args)
    exp = TranslationExperiment(work_dir)
    gen.prepare_experiment(exp)


