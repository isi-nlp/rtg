import os
from typing import List, Iterator, Tuple, Union, Optional, Iterable, Dict, Any
import torch
from rtg import log
from . import my_tensor as tensor, device
from rtg.utils import IO, line_count
import math
import random
from collections import namedtuple
from sentencepiece import SentencePieceProcessor, SentencePieceTrainer
from pathlib import Path
import sqlite3
import pickle
from itertools import zip_longest

PAD_TOK = '<pad>', 0
UNK_TOK = '<unk>', 1
BOS_TOK = '<s>', 2
EOS_TOK = '</s>', 3
CLS_TOK = '<cls>', 4

PAD_TOK_IDX = PAD_TOK[1]
UNK_TOK_IDX = UNK_TOK[1]
BOS_TOK_IDX = BOS_TOK[1]
EOS_TOK_IDX = EOS_TOK[1]
CLS_TOK_IDX = CLS_TOK[1]

RESERVED_TOKS = [PAD_TOK, UNK_TOK, BOS_TOK, EOS_TOK, CLS_TOK]

RawRecord = Tuple[str, str]
TokRawRecord = Tuple[List[str], List[str]]
MonoSeqRecord = List[Union[int, str]]
ParallelSeqRecord = Tuple[MonoSeqRecord, MonoSeqRecord]
TokStream = Union[Iterator[Iterator[str]], Iterator[str]]


class Field(SentencePieceProcessor):
    """A wrapper class for sentence piece trainer and processor"""

    def __init__(self, path: str):
        super(Field, self).__init__()
        assert self.load(path)

    def encode_as_ids(self, text: str, add_bos=False, add_eos=False) -> List[int]:
        ids = super(Field, self).encode_as_ids(text)
        if add_bos and ids[0] != BOS_TOK[1]:
            ids.insert(0, BOS_TOK[1])
        if add_eos and ids[-1] != EOS_TOK[1]:
            ids.append(EOS_TOK[1])
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
    def train(model_type: str, vocab_size: int, model_path: str, files: Iterator[str],
              no_split_toks: Optional[List[str]] = None):
        """
        Train Sentence Piece Model
        :param model_type: sentence piece model type: {unigram, BPE, word, char}
        :param vocab_size: target vocabulary size
        :param model_path: where to store model
        :param files: input files
        :param no_split_toks: Don't split these tokens
        :return:
        """

        model_prefix = model_path.replace('.model', '')
        files = set(files)  # remove duplicates
        arg = f"--input={','.join(files)} --vocab_size={vocab_size} --model_prefix={model_prefix}" \
            f" --model_type={model_type} --pad_id={PAD_TOK[1]} --bos_id={BOS_TOK[1]}" \
            f" --eos_id={EOS_TOK[1]} --unk_id={UNK_TOK[1]} --hard_vocab_limit=false"
        # CLS token goes in the beginning because we need it get index 4
        cls_tok_str = CLS_TOK[0]
        if no_split_toks:
            no_split_toks_str = ','.join([cls_tok_str] + no_split_toks)
        else:
            no_split_toks_str = cls_tok_str
        arg += f" --user_defined_symbols={no_split_toks_str}"
        if model_type == 'bpe':  # BPE can have longer sentences, default is 2048
            arg += " --max_sentence_length=8192"
        log.info(f"SPM: {arg}")
        SentencePieceTrainer.Train(arg)
        log.info("Training complete")
        if not model_path.endswith('.model'):
            model_path += '.model'
        model = Field(model_path)
        for piece, idx in RESERVED_TOKS:
            assert model.piece_to_id(piece) == idx
        return model


Example = namedtuple('Example', ['x', 'y'])
"""
An object of this class holds an example in sequence to sequence dataset
"""


class TSVData:

    def __init__(self, path: Union[str, Path], in_mem=False, shuffle=False, longest_first=True):
        """
        :param path: path to TSV file have parallel sequences
        :param in_mem: hold data in memory instead of reading from file for subsequent pass.
         Don't use in_mem for large data_sets.
        :param shuffle: shuffle data between the reads
        :param longest_first: On the first read, get the longest sequence first by sorting by length
        """
        self.path = path
        self.in_mem = in_mem or shuffle or longest_first
        self.longest_first = longest_first
        self.shuffle = shuffle
        self.mem = list(self.read_all()) if self.in_mem else None
        self._len = len(self.mem) if self.in_mem else line_count(path)
        self.read_counter = 0

    @staticmethod
    def _parse(line: str):
        return [int(t) for t in line.split()]

    def read_all(self) -> Iterator[Example]:
        with IO.reader(self.path) as lines:
            recs = (line.split('\t') for line in lines)
            for rec in recs:
                if rec[0] and rec[0].strip():
                    yield Example(self._parse(rec[0]),
                                  self._parse(rec[1]) if len(rec) > 1 else None)

    def __len__(self):
        return self._len

    def __iter__(self) -> Iterator[Example]:
        if self.shuffle:
            if self.read_counter == 0:
                log.info("shuffling the data...")
            random.shuffle(self.mem)
        if self.longest_first:
            if self.read_counter == 0:
                log.info("Sorting the dataset by length of target sequence")
            sort_key = lambda ex: len(ex.y) if ex.y is not None else len(ex.x)
            self.mem = sorted(self.mem, key=sort_key, reverse=True)
            if self.read_counter == 0:
                log.info(f"Longest source seq length: {len(self.mem[0].x)}")

        yield from self.mem if self.mem else self.read_all()
        self.read_counter += 1

    @staticmethod
    def write_lines(lines, path):
        log.info(f"Storing data at {path}")
        with IO.writer(path) as f:
            for line in lines:
                f.write(line)
                f.write('\n')

    @staticmethod
    def write_parallel_recs(records: Iterator[ParallelSeqRecord], path: Union[str, Path]):
        seqs = ((' '.join(map(str, x)), ' '.join(map(str, y))) for x, y in records)
        lines = (f'{x}\t{y}' for x, y in seqs)
        TSVData.write_lines(lines, path)

    @staticmethod
    def write_mono_recs(records: Iterator[MonoSeqRecord], path: Union[str, Path]):
        lines = (' '.join(map(str, rec)) for rec in records)
        TSVData.write_lines(lines, path)

    @staticmethod
    def read_raw_parallel_lines(src_path: Union[str, Path], tgt_path: Union[str, Path]) \
            -> Iterator[RawRecord]:
        with IO.reader(src_path) as src_lines, IO.reader(tgt_path) as tgt_lines:
            # if you get an exception here --> files have un equal number of lines
            recs = ((src.strip(), tgt.strip()) for src, tgt in zip_longest(src_lines, tgt_lines))
            recs = ((src, tgt) for src, tgt in recs if src and tgt)
            yield from recs

    @staticmethod
    def read_raw_parallel_recs(src_path: Union[str, Path], tgt_path: Union[str, Path],
                               truncate: bool, src_len: int, tgt_len: int, tokenizer) \
            -> Iterator[ParallelSeqRecord]:
        recs = TSVData.read_raw_parallel_lines(src_path, tgt_path)
        recs = ((tokenizer(x), tokenizer(y)) for x, y in recs)
        if truncate:
            recs = ((src[:src_len], tgt[:tgt_len]) for src, tgt in recs)
        else:  # Filter out longer sentences
            recs = ((src, tgt) for src, tgt in recs if len(src) <= src_len and len(tgt) <= tgt_len)
        return recs

    @staticmethod
    def read_raw_mono_recs(path: Union[str, Path], truncate: bool, max_len: int, tokenizer):
        with IO.reader(path) as lines:
            recs = (tokenizer(line.strip()) for line in lines if line.strip())
            if truncate:
                recs = (rec[:max_len] for rec in recs)
            else:  # Filter out longer sentences
                recs = (rec for rec in recs if 0 < len(rec) <= max_len)
            yield from recs


class SqliteFile:
    TABLE_STATEMENT = """CREATE TABLE IF NOT EXISTS data (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        x BLOB NOT NULL,
        y BLOB,
        x_len INTEGER,
        y_len INTEGER)"""

    INSERT_STMT = "INSERT INTO data (x, y, x_len, y_len) VALUES (?, ?, ?, ?)"
    READ_RANDOM = "SELECT * from data ORDER BY RANDOM()"
    READ_X_LEN_DESC_RANDOM = "SELECT * from data ORDER BY x_len DESC, RANDOM()"
    READ_Y_LEN_DESC_RANDOM = "SELECT * from data ORDER BY y_len DESC, RANDOM()"
    COUNT_ROWS = "SELECT COUNT(*) as COUNT from data"

    def __init__(self, path: Path, shuffle=True, longest_first=False, sort_side='tgt'):
        self.path = path
        assert path.exists()
        assert sort_side in {'src', 'tgt'}
        self.sort_side = sort_side
        self.shuffle = shuffle
        self.longest_first = longest_first
        self.db = sqlite3.connect(str(path))

        def dict_factory(cursor, row):  # map tuples to dictionary with column names
            d = {}
            for idx, col in enumerate(cursor.description):
                key = col[0]
                val = row[idx]
                if key in ('x', 'y') and val is not None:
                    val = pickle.loads(val)  # unmarshall
                d[key] = val
            return d

        self.db.row_factory = dict_factory

    def __len__(self):
        return self.db.execute(self.COUNT_ROWS).fetchone()['COUNT']

    def read_all(self) -> Iterator[Dict[str, Any]]:
        assert self.shuffle, 'only shuffled read is supported as of now'
        # because thats the only usecase for now

        if self.longest_first:
            select_qry = self.READ_X_LEN_DESC_RANDOM if self.sort_side == 'src' \
                else self.READ_X_LEN_DESC_RANDOM
        else:
            select_qry = self.READ_RANDOM
        return self.db.execute(select_qry)

    def __iter__(self) -> Iterator[Example]:
        for d in self.read_all():
            yield Example(d['x'], d.get('y'))

    @classmethod
    def write(cls, path, records: Iterator[ParallelSeqRecord]):
        if path.exists():
            log.warning(f"Overwriting {path} with new records")
            os.remove(str(path))
        log.info(f'Creating {path}')

        conn = sqlite3.connect(str(path))
        cur = conn.cursor()
        cur.execute(cls.TABLE_STATEMENT)
        count = 0
        for x_seq, y_seq in records:
            # marshall variable length sequences to a json array
            values = (pickle.dumps(x_seq),
                      None if y_seq is None else pickle.dumps(y_seq),
                      len(x_seq), len(y_seq) if y_seq is not None else -1)
            cur.execute(cls.INSERT_STMT, values)
            count += 1
        cur.close()
        conn.commit()
        log.info(f"stored {count} rows in {path}")


def read_tsv(path: str):
    assert os.path.exists(path)
    with IO.reader(path) as f:
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


def padded_sequence_mask(lengths, max_len=None):
    """
    :param lengths: a sequence of lenghts
    :param max_len: pad upto this length
    :return:
    """
    max_len = max_len if max_len else lengths.max()
    batch_size = lengths.size(0)
    # create a row [0, 1, ... s] and duplicate this row batch_size times --> [B, S]
    seq_range_expand = torch.arange(0, max_len, dtype=torch.long,
                                    device=device).expand(batch_size, max_len)
    # make lengths vectors to [B x 1] and duplicate columns to [B, S]
    seq_length_expand = lengths.unsqueeze(1).expand_as(seq_range_expand)
    return seq_range_expand < seq_length_expand  # 0 if padding, 1 otherwise


class Batch:
    """
    An object of this class holds a batch of examples
    """
    pad_value = PAD_TOK[1]
    bos_val = BOS_TOK[1]
    eos_val = EOS_TOK[1]

    _x_attrs = ['x_len', 'x_seqs']
    _y_attrs = ['y_len', 'y_seqs']

    @classmethod
    def bos_eos_check(cls, batch: List[Example], side: str, bos: bool, eos: bool):
        """
        ensures and inserts (if needed) EOS and BOS tokens
        :param batch:
        :param side: which side? choices: {'x', 'y'}
        :param bos: True if should have BOS, False if should not have BOS
        :param eos: True if should have EOS, False if should not have EOS
        :return: None, all modifications are inplace of batch
        """
        assert side in ('x', 'y')
        for ex in batch:
            seq: List = ex.x if side == 'x' else ex.y
            if bos:
                if not seq[0] == cls.bos_val:
                    seq.insert(0, cls.bos_val)
            else:  # should not have BOS
                assert seq[0] != cls.bos_val
            if eos:
                if not seq[-1] == cls.eos_val:
                    seq.append(cls.eos_val)
            else:  # Should not have EOS
                assert seq[-1] != cls.eos_val

    def __init__(self, batch: List[Example], sort_dec=False, batch_first=True,
                 add_eos_x=True, add_eos_y=True, add_bos_x=False, add_bos_y=False):
        """
        :param batch: List fo Examples
        :param sort_dec: True if the examples be sorted as descending order of their source sequence lengths
        :Param Batch_First: first dimension is batch
        """
        self.eos_x = add_eos_x
        self.eos_y = add_eos_y
        self.bos_x = add_bos_x
        self.bos_y = add_bos_y
        self.batch_first = batch_first

        self.bos_eos_check(batch, 'x', add_bos_x, add_eos_x)
        if sort_dec:
            batch = sorted(batch, key=lambda _: len(_.x), reverse=True)
        self._len = len(batch)
        self.x_len = tensor([len(e.x) for e in batch])
        self.x_toks = self.x_len.sum().float().item()
        self.max_x_len = self.x_len.max()

        # create x_seqs on CPU RAM and move to GPU at once
        self.x_seqs = torch.full(size=(self._len, self.max_x_len), fill_value=self.pad_value,
                                 dtype=torch.long)
        for i, ex in enumerate(batch):
            self.x_seqs[i, :len(ex.x)] = torch.tensor(ex.x, dtype=torch.long)
        self.x_seqs = self.x_seqs.to(device)
        if not batch_first:  # transpose
            self.x_seqs = self.x_seqs.t()

        first_y = batch[0].y
        self.has_y = first_y is not None
        if self.has_y:
            self.bos_eos_check(batch, 'y', add_bos_y, add_eos_y)
            self.y_len = tensor([len(e.y) for e in batch])
            self.y_toks = self.y_len.sum().float().item()
            self.max_y_len = self.y_len.max().item()
            y_seqs = torch.full(size=(self._len, self.max_y_len), fill_value=self.pad_value,
                                dtype=torch.long)
            for i, ex in enumerate(batch):
                y_seqs[i, :len(ex.y)] = torch.tensor(ex.y, dtype=torch.long)
            self.y_seqs = y_seqs.to(device)
            if not batch_first:  # transpose
                self.y_seqs = self.y_seqs.t()

    def __len__(self):
        return self._len

    def to(self, device):
        """Move this batch to given device"""
        for name in self._x_attrs + (self._y_attrs if self.has_y else []):
            setattr(self, name, getattr(self, name).to(device))
        return self

    @staticmethod
    def make_target_mask(tgt, pad=pad_value):
        "Create a mask to hide padding and future words."
        tgt_mask = (tgt != pad).unsqueeze(1)
        tgt_mask = tgt_mask & subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data)
        return tgt_mask


class BatchIterable(Iterable[Batch]):

    def __init__(self, data_path: Union[str, Path], batch_size: int,
                 sort_desc: bool =False, batch_first: bool = True, shuffle: bool = False):
        """
        Iterator for reading training data in batches
        :param data_path: path to TSV file
        :param batch_size: number of tokens on the target size per batch

        :param sort_desc: should the batch be sorted by src sequence len (useful for RNN api)
        """
        self.sort_desc = sort_desc
        if not isinstance(data_path, Path):
            data_path = Path(data_path)
        if data_path.name.endswith(".db"):
            self.data = SqliteFile(data_path, shuffle=shuffle, longest_first=True)
        else:
            self.data = TSVData(data_path, shuffle=shuffle, longest_first=True)
        self.batch_size = batch_size
        self.batch_first = batch_first
        log.info(f'Batch Size = {batch_size} toks')

    def read_all(self):
        batch = []
        max_len = 0
        for ex in self.data:
            if min(len(ex.x), len(ex.y)) == 0:
                log.warn("Skipping a record,  either source or target is empty")
                continue

            this_len = max(len(ex.x), len(ex.y))
            if (len(batch) + 1) * max(max_len, this_len) <= self.batch_size:
                batch.append(ex) # this one can go in
                max_len = max(max_len, this_len)
            else:
                if this_len > self.batch_size:
                    raise Exception(f'Unable to make a batch of {self.batch_size} toks'
                                    f' with a seq of x_len:{len(ex.x)} y_len:{len(ex.y)}')
                # yield the current batch
                yield Batch(batch, sort_dec=self.sort_desc, batch_first=self.batch_first)
                batch = [ex] #  new batch
                max_len = this_len
        if batch:
            log.debug(f"\nLast batch, size={len(batch)}")
            yield Batch(batch, sort_dec=self.sort_desc, batch_first=self.batch_first)

    def __iter__(self) -> Iterator[Batch]:
        yield from self.read_all()

    @property
    def num_items(self) -> int:
        return len(self.data)

    @property
    def num_batches(self) -> int:
        return int(math.ceil(len(self.data) / self.batch_size))


class LoopingIterable(Iterable[Batch]):
    """
    An iterable that keeps looping until a specified number of step count is reached
    """

    def __init__(self, iterable: Iterable, batches: int):
        self.itr = iterable
        self.total = batches
        self.count = 0

    def __iter__(self) -> Iterator[Batch]:
        while self.count < self.total:
            for batch in self.itr:
                yield batch
                self.count += 1
                if self.count >= self.total:
                    break
