#!/usr/bin/env python
# Generates dummy data for testing
import random
from tgnmt.dataprep import RESERVED_TOKS, Field, Batch, Example
from tgnmt import TranslationExperiment
import numpy as np
import argparse


class DataGen:

    def __init__(self, min_tok=len(RESERVED_TOKS), max_tok=16, min_len=1, max_len=10,
                 num_train_exs=1000, num_val_exs=50):
        """
        Dummy Dataset for quick testing. This class is useful for creating a Dummy Experiment on disk.
        For in memory dummy data, see dummy.BatchIterable
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


class BatchIterable:
    # TODO: How to specify Type Hint for this as Iterable[Batch]
    """Dummy equivalent of dataprep.BatchIterable"""

    def __init__(self, vocab_size, batch_size, n_batches, seq_len=10, n_reserved_toks=Batch.eos_val+1, reverse=True,
                 batch_first=False):
        """
         "Generate random data for a src-tgt copy task."
         :param vocab_size: Vocabulary size
         :param batch_size:
         :param n_batches: number of batches to produce
         :param n_reserved_toks:  number of reserved tokens (such as pad, EOS, BOS, UNK etc)
         :param reverse: reverse the target
         :param batch_first: first dimension is batch
         :return:
         """

        self.vocab_size = vocab_size
        self.batch_size = batch_size
        self.num_batches = n_batches
        self.seq_len = seq_len
        self.n_reserved_toks = n_reserved_toks
        self.reverse = reverse
        self.batch_first = batch_first

    def make_an_ex(self):
        data = np.random.randint(self.n_reserved_toks, self.vocab_size, size=(self.seq_len,))
        tgt = self.vocab_size + (self.n_reserved_toks - 1) - data if self.reverse else data
        tgt[0] = Batch.bos_val
        data[0] = Batch.bos_val
        return Example(data, tgt)

    def __iter__(self):
        for i in range(self.num_batches):
            exs = [self.make_an_ex() for _ in range(self.batch_size)]
            yield Batch(exs, sort_dec=True, batch_first=self.batch_first)


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


