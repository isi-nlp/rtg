import os
from typing import List, Iterator, Tuple, Union
import torch
from rtg import log
from . import my_tensor as tensor, device
import math
import random
from collections import namedtuple
from sentencepiece import SentencePieceProcessor, SentencePieceTrainer

PAD_TOK = '<pad>', 0
UNK_TOK = '<unk>', 1
BOS_TOK = '<s>', 2
EOS_TOK = '</s>', 3

RESERVED_TOKS = [PAD_TOK, UNK_TOK, BOS_TOK, EOS_TOK]


RawRecord = Tuple[str, str]
TokRawRecord = Tuple[List[str], List[str]]
SeqRecord = Tuple[List[Union[int, str]], List[Union[int, str]]]
TokStream = Union[Iterator[Iterator[str]], Iterator[str]]


class Field(SentencePieceProcessor):
    """A wrapper class for sentence piece trainer and processor"""

    def __init__(self, path):
        super(Field, self).__init__()
        assert self.load(path)

    def encode_as_ids(self, text: str, add_bos=False, add_eos=False) -> List[int]:
        ids = super(Field, self).encode_as_ids(text)
        if add_bos and ids[0] != BOS_TOK[1]:
            ids.insert(0, BOS_TOK[1])
        if add_eos and ids[-1] != EOS_TOK[1]:
            ids.append(BOS_TOK[1])
        return ids

    def decode_ids(self, ids: List[int], trunc_eos=False) -> str:
        """
        convert ids to text
        :param ids:
        :param trunc_eos: skip everything after first EOS token in sequence
        :return:
        """
        if trunc_eos:
            try:
                ids = ids[:ids.index(EOS_TOK[1])]
            except ValueError:
                pass
        return super(Field, self).decode_ids(ids)

    def tokenize(self, text: str) -> List[str]:
        return self.encode_as_pieces(text.encode())

    def detokenize(self, tokens: List[str]) -> str:
        return ''.join(tokens).replace('▁', ' ').strip()

    @staticmethod
    def train(model_type: str, vocab_size: int, model_path: str, files: Iterator[str]):
        """
        Train Sentence Piece Model
        :param model_type: sentence piece model type: {unigram, BPE, word, char}
        :param vocab_size: target vocabulary size
        :param model_path: where to store model
        :param files: input files
        :return:
        """
        model_prefix = model_path.replace('.model', '')
        arg = f"--input={','.join(files)} --vocab_size={vocab_size} --model_prefix={model_prefix}" \
              f" --model_type={model_type} --pad_id={PAD_TOK[1]} --bos_id={BOS_TOK[1]}" \
              f" --eos_id={EOS_TOK[1]} --unk_id={UNK_TOK[1]}"
        log.info(f"SPM: {arg}")
        SentencePieceTrainer.Train(arg)
        log.info("Training complete")
        if not model_path.endswith('.model'):
            model_path += '.model'
        return Field(model_path)


Example = namedtuple('Example', ['x', 'y'])
"""
An object of this class holds an example in sequence to sequence dataset
"""


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
            for ex in batch:    # check and insert BOS to output seqs
                if ex.y[0] != self.bos_val:
                    ex.y.insert(0, self.bos_val)
                if ex.y[-1] != self.eos_val:
                    ex.y.append(self.eos_val)

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
