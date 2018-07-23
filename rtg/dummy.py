#!/usr/bin/env python
# Generates dummy data for testing
from rtg.dataprep import Batch, Example
import numpy as np


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
        return Example(data.tolist(), tgt.tolist())

    def __iter__(self):
        for i in range(self.num_batches):
            exs = [self.make_an_ex() for _ in range(self.batch_size)]
            yield Batch(exs, sort_dec=True, batch_first=self.batch_first)
