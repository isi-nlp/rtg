import os
from typing import List, Iterator, Tuple, Union
import torch
from tgnmt import log
from . import my_tensor as tensor, device
import math
import random

PAD_TOK = '<pad>', 0
UNK_TOK = '<unk>', 1
BOS_TOK = '<s>', 2
EOS_TOK = '</s>', 3

RESERVED_TOKS = [PAD_TOK, UNK_TOK, BOS_TOK, EOS_TOK]


RawRecord = Tuple[str, str]
TokRawRecord = Tuple[List[str], List[str]]
SeqRecord = Tuple[List[Union[int, str]], List[Union[int, str]]]
TokStream = Union[Iterator[Iterator[str]], Iterator[str]]


class Example:
    """
    An object of this class holds an example in sequence to sequence dataset
    """

    def __init__(self, x: List[int], y: List[int] = None):
        self.x = x
        self.y = y


class TSVData:

    def __init__(self, path: str, in_mem=False, shuffle=False):
        """
        :param path: path to TSV file have parallel sequences
        :param in_mem: hold data in memory instead of reading from file for subsequent pass.
         Don't use in_mem for large data_sets.
        """
        self.path = path
        self.in_mem = in_mem or shuffle
        self.shuffle = shuffle
        self.mem = list(self.read_all()) if self.in_mem else None

    @staticmethod
    def _parse(line: str):
        return [int(t) for t in line.split()]

    def read_all(self) -> Iterator[Example]:
        with open(self.path) as lines:
            recs = (line.split('\t') for line in lines)
            recs = (Example(self._parse(rec[0]), self._parse(rec[1])) for rec in recs)
            yield from recs

    def __len__(self):
        if not self.mem:
            raise RuntimeError('Length is known only when in_mem or shuffle are enabled')
        return len(self.mem)

    def __iter__(self) -> Iterator[Example]:
        if self.shuffle:
            log.info("shuffling the data...")
            random.shuffle(self.mem)
        yield from self.mem if self.mem else self.read_all()


def read_tsv(path: str):
    assert os.path.exists(path)
    with open(path, encoding='utf-8') as f:
        yield from (line.split('\t') for line in f)


def tokenize(strs: List[str]) -> List[List[str]]:
    return [s.split() for s in strs]


def subsequent_mask(size):
    """
    Mask out subsequent positions. upper diagonal elements should be zero
    :param size:
    :return: mask where positions are filled with zero for subsequent positions
    """
    # upper diagonal elements are 1s, lower diagonal and the main diagonal are zeroed
    triu = torch.triu(torch.ones(size, size, dtype=torch.int8, device=device), diagonal=1)
    # invert it
    mask = triu == 0
    mask = mask.unsqueeze(0)
    return mask


def sent_piece_train(model_type: str, vocab_size: int, model_prefix: str, files):
    """
    Train Sentence Piece Model
    :param model_type: model type
    :param vocab_size:
    :param model_prefix:
    :param files:
    :return:
    """
    model_prefix = model_prefix.replace('.model', '')
    import sentencepiece as spm
    arg = f"--input={','.join(files)} --vocab_size={vocab_size} --model_prefix={model_prefix}" \
          f" --model_type={model_type} --pad_id={PAD_TOK[1]} --bos_id={BOS_TOK[1]}" \
          f" --eos_id={EOS_TOK[1]} --unk_id={UNK_TOK[1]}"
    log.info(f"SPM: {arg}")
    spm.SentencePieceTrainer.Train(arg)


class Batch:
    """
    An object of this class holds a batch of examples
    """
    pad_value = PAD_TOK[1]
    bos_val = BOS_TOK[1]
    eos_val = EOS_TOK[1]

    def __init__(self, batch: List[Example], sort_dec=False, batch_first=True):
        """
        :param batch: List fo Examples
        :param sort_dec: True if the examples be sorted as descending order of their source sequence lengths
        :param batch_first: first dimension is batch
        """
        self.batch_first = batch_first
        if sort_dec:
            batch = sorted(batch, key=lambda _: len(_.x), reverse=True)
        self._len = len(batch)
        self.x_len = tensor([len(e.x) for e in batch])
        self.x_toks = self.x_len.sum().float().item()
        self.max_x_len = self.x_len.max()
        self.x_seqs = torch.full(size=(self._len, self.max_x_len), fill_value=self.pad_value,
                                 dtype=torch.long, device=device)
        for i, ex in enumerate(batch):
            self.x_seqs[i, :len(ex.x)] = tensor(ex.x)
        if not batch_first:      # transpose
            self.x_seqs = self.x_seqs.t()
        self.x_mask = (self.x_seqs != self.pad_value).unsqueeze(1)
        first_y = batch[0].y
        if first_y is not None:
            # Some sanity checks
            assert first_y[0] == self.bos_val, f'Output sequences must begin with BOS token {self.bos_val}'
            # assert first_y[-1] == self.eos_val, f'Output sequences must end with EOS token {self.eos_val}'

            self.y_len = tensor([len(e.y) for e in batch])  # Excluding either BOS or EOS tokens
            self.y_toks = self.y_len.sum().float().item()
            self.max_y_len = self.y_len.max().item()
            y_seqs = torch.full(size=(self._len, self.max_y_len + 1), fill_value=self.pad_value,
                                dtype=torch.long, device=device)
            for i, ex in enumerate(batch):
                y_seqs[i, :len(ex.y)] = tensor(ex.y)
            self.y_seqs_nobos = y_seqs[:, 1:]  # predictions
            self.y_seqs = y_seqs[:, :-1]
            if not batch_first:    # transpose
                self.y_seqs = self.y_seqs.t()
                self.y_seqs_nobos = self.y_seqs_nobos.t()
            self.y_mask = self.make_std_mask(self.y_seqs)

    def __len__(self):
        return self._len

    @staticmethod
    def make_std_mask(tgt, pad=pad_value):
        "Create a mask to hide padding and future words."
        tgt_mask = (tgt != pad).unsqueeze(1)
        tgt_mask = tgt_mask & subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data)
        return tgt_mask


class BatchIterable:
    # TODO: How to specify Type Hint for this as Iterable[Batch] ?

    def __init__(self, data_path: str, batch_size: int, sort_dec=True, batch_first=True, shuffle=False):
        """
        Iterator for reading training data in batches
        :param data_path: path to TSV file
        :param batch_size: number of examples per batch
        :param sort_dec: should the records within batch be sorted descending order of sequence length?
        """
        self.data = TSVData(data_path, shuffle=shuffle)
        self.batch_size = batch_size
        self.sort_dec = sort_dec
        self.batch_first = batch_first

    def read_all(self):
        batch = []
        for ex in self.data:
            batch.append(ex)
            if len(batch) >= self.batch_size:
                yield Batch(batch, sort_dec=self.sort_dec, batch_first=self.batch_first)
                batch = []
        if batch:
            log.debug(f"\nLast batch, size={len(batch)}")
            yield Batch(batch, sort_dec=self.sort_dec, batch_first=self.batch_first)

    def __iter__(self):
        yield from self.read_all()

    @property
    def num_items(self):
        return len(self.data)

    @property
    def num_batches(self):
        return math.ceil(len(self.data) / self.batch_size)
