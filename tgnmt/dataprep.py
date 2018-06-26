import os
from collections import Counter
from typing import List, Dict, Iterator, Tuple, Union
import torch
from tgnmt import log
from . import my_tensor as tensor, device

BLANK_TOK = '-BLANK-', 0
UNK_TOK = '-UNK-', 1
BOS_TOK = '-BOS-', 2
EOS_TOK = '-EOS-', 3

RESERVED_TOKS = [BLANK_TOK, UNK_TOK, BOS_TOK, EOS_TOK]

from tgnmt.bpe import SubwordTextEncoder

RawRecord = Tuple[List[str], List[str]]
SeqRecord = Tuple[List[int], List[int]]
TokStream = Union[Iterator[Iterator[str]], Iterator[str]]


class Field:
    """
    An instance of this class holds a vocabulary of a dataset.
    This class is inspired by the torchtext module's Field class.
    """

    num_reserved_toks = len(RESERVED_TOKS)

    def __init__(self, name: str, blank=False, subword_enc: SubwordTextEncoder = None):
        assert ',' not in name
        self.name: str = name
        self.subword_enc = subword_enc
        self.tok2idx: Dict[str, int] = {} if blank else {t: i for t, i in RESERVED_TOKS}
        self.idx2tok: List[str] = [] if blank else [t for t, _ in RESERVED_TOKS]
        self.freq: List[int] = [-1 for _ in self.idx2tok]
        if self.subword_enc:
            self.build_from_types(self.subword_enc.all_subtoken_strings)

    @staticmethod
    def get_tok_freq(tok_stream: TokStream, min_freq=1, nested=True):
        if nested:
            tok_stream = (tok for seq in tok_stream for tok in seq)  # flatten
        tok_freq = Counter(tok_stream).items()
        if min_freq > 1:
            tok_freq = [(t, f) for t, f in tok_freq if f >= min_freq]
        return sorted(tok_freq, key=lambda x: x[1], reverse=True)

    def build_from(self, stream: TokStream, min_freq: int = 1, max_vocab_size: int = 2 ** 24):
        tok_freq = self.get_tok_freq(stream, min_freq=min_freq)
        if len(tok_freq) > max_vocab_size:
            log.info(f'Truncating vocab size from {len(tok_freq)} to {max_vocab_size}')
            tok_freq = tok_freq[:max_vocab_size]
        for tok, freq in tok_freq:
            self.add_token(tok, inc=freq)
        return self

    def build_from_types(self, types, count=0):
        for tok in types:
            self.add_token(tok, count)
        return self

    def add_token(self, tok: str, inc: int = 1):
        """
        Adds token to vocabulary
        :param tok: token
        :param inc: frequency in the corpus
        :return:
        """
        if tok in self.tok2idx:
            self.freq[self.tok2idx[tok]] += inc
        else:
            idx = len(self.idx2tok)
            self.tok2idx[tok] = idx
            self.idx2tok.append(tok)
            self.freq.append(inc)

    def seq2idx(self, toks: List[str], add_bos=True, add_eos=True, subword_split=True) -> List[int]:
        """
        transforms a sequence of words to word indices.
         If input has tokens which doesnt exist in vocabulary, they will be replaced with UNK token's index.
        :param toks: sequence of tokens which needs to be transformed
        :param add_bos: prepend BOS token index. If input already has BOS token then this flag has no effect.
        :param add_eos: append EOS token index. If input already has EOS token then this flag has no effect.
        :param subword_split: if subword_encoder is available, split tokens into subwords
        :return: List of word indices
        """
        if subword_split and self.subword_enc:
            toks = self.subword_enc.tokens_to_subtokens(toks)
        seq = [self.tok2idx.get(tok, UNK_TOK[1]) for tok in toks]
        if add_bos and seq[0] != BOS_TOK[1]:
            seq.insert(0, BOS_TOK[1])
        if add_eos and seq[-1] != EOS_TOK[1]:
            seq.append(EOS_TOK[1])
        return seq

    def idx2seq(self, indices: List[int], trunc_eos=False) -> List[str]:
        """
        :param indices: sequence of word indices
        :param trunc_eos: True if the sequence should be truncated at the first occurrence of EOS
        :return: List of tokens
        """
        res = []
        for idx in indices:
            if trunc_eos and idx == EOS_TOK[1]:
                break
            res.append(self.idx2tok[idx] if idx < len(self.idx2tok) else '-:OutOfIndex:-')
        if self.subword_enc:
            self.subword_enc.un_split(res)
        return res

    def size(self):
        """
        :return: number of tokens, including reserved
        """
        return len(self.idx2tok)

    def dump_tsv(self, path: str):
        """
        Dumps this instance to a TSV file at given path
        :param path: path to output file
        :return:
        """
        with open(path, 'w', encoding='utf-8') as f:
            header = self.name
            header += ",subwords" if self.subword_enc else ""
            f.write(f'{header}\n')
            for i, (tok, count) in enumerate(zip(self.idx2tok, self.freq)):
                f.write(f'{i}\t{tok}\t{count}\n')

    @staticmethod
    def load_tsv(path: str):
        """
        Loads Field instance from serialized TSV data
        :param path: path to TSV file which was crated from Field.dump_tsv() method
        :return: an instance of Field
        """
        with open(path, 'r', encoding='utf-8') as f:
            parts = f.readline().strip().split(',')
            subword_mode = len(parts) > 1 and 'subwords' in parts[1:]
            field = Field(parts[0], blank=True)
            i = 0
            for line in f:
                idx, tok, count = line.split('\t')
                idx, count = int(idx), int(count)
                assert idx == i, f'expected index {i}, got={idx}'
                field.add_token(tok, inc=count)
                i += 1
            if subword_mode:
                # TODO: what happens with the reserved tokens ?
                field.subword_enc = SubwordTextEncoder(subtoks=field.idx2tok)
            return field


class Example:
    """
    An object of this class holds an example in sequence to sequence dataset
    """

    def __init__(self, x: List[int], y: List[int] = None):
        self.x = x
        self.y = y


class TSVData:

    def __init__(self, path: str, in_mem=False):
        """
        :param path: path to TSV file have parallel sequences
        :param in_mem: hold data in memory instead of reading from file for subsequent pass.
         Don't use in_mem for large data_sets.
        """
        self.path = path
        self.in_mem = in_mem
        self.mem = list(self.read_all()) if in_mem else None

    @staticmethod
    def _parse(line: str):
        return [int(t) for t in line.split()]

    def read_all(self) -> Iterator[Example]:
        with open(self.path) as lines:
            recs = (line.split('\t') for line in lines)
            recs = (Example(self._parse(rec[0]), self._parse(rec[1])) for rec in recs)
            yield from recs

    def __len__(self):
        if not self.in_mem:
            raise RuntimeError('Length is known only when in_mem')
        return len(self.mem)

    def __iter__(self) -> Iterator[Example]:
        yield from self.mem if self.in_mem else self.read_all()


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
    pad_value = BLANK_TOK[1]
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

    def __init__(self, data_path: str, batch_size: int, sort_dec=True, batch_first=True):
        """
        Iterator for reading training data in batches
        :param data_path: path to TSV file
        :param batch_size: number of examples per batch
        :param sort_dec: should the records within batch be sorted descending order of sequence length?
        """
        self.data = TSVData(data_path)
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
