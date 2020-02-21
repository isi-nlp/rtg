import os
from typing import List, Iterator, Tuple, Union, Optional, Iterable, Dict, Any
import torch
from rtg import log, my_tensor as tensor, device
from rtg.utils import IO, line_count, get_my_args
import math
import random
from collections import namedtuple
from sentencepiece import SentencePieceProcessor, SentencePieceTrainer
from pathlib import Path
import sqlite3
import pickle
from itertools import zip_longest
from abc import ABCMeta, abstractmethod

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


class Field(metaclass=ABCMeta):

    @abstractmethod
    def encode_as_ids(self, text, add_bos, add_eos):
        pass

    @abstractmethod
    def decode_ids(self, ids, trunc_eos):
        """
        convert ids to text
        :param ids:
        :param trunc_eos: skip everything after first EOS token in sequence
        :return:
        """
        pass

    @abstractmethod
    def tokenize(self, text):
        pass

    @abstractmethod
    def detokenize(self, tokens):
        pass

    @staticmethod
    @abstractmethod
    def train(model_type: str, vocab_size: int, model_path: str, files: Iterator[str],
              no_split_toks: Optional[List[str]] = None,
              char_coverage: float = -1.0):
        """
        Train Sentence Piece Model
        :param model_type: sentence piece model type: {unigram, BPE, word, char}
        :param vocab_size: target vocabulary size
        :param model_path: where to store model
        :param files: input files
        :param no_split_toks: Don't split these tokens
        :return:
        """
        pass


class SPField(SentencePieceProcessor, Field):
    """A wrapper class for sentence piece trainer and processor"""

    def __init__(self, path: str):
        super().__init__()
        assert self.load(path)

    def encode_as_ids(self, text: str, add_bos=False, add_eos=False) -> List[int]:
        ids = super(SPField, self).encode_as_ids(text)
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
        return super(SPField, self).decode_ids(ids)

    def tokenize(self, text: str) -> List[str]:
        return self.encode_as_pieces(text.encode())

    def detokenize(self, tokens: List[str]) -> str:
        return ''.join(tokens).replace('▁', ' ').strip()

    @staticmethod
    def train(model_type: str, vocab_size: int, model_path: str, files: Iterator[str],
              no_split_toks: Optional[List[str]] = None, char_coverage: float = 0):
        """
        Train Sentence Piece Model
        :param model_type: sentence piece model type: {unigram, BPE, word, char}
        :param vocab_size: target vocabulary size
        :param model_path: where to store model
        :param files: input files
        :param no_split_toks: Don't split these tokens
        :param char_coverage: character coverage (0, 1]. value <= 0 => default coverage 0.9995%
        :return:
        """
        model_prefix = model_path.replace('.model', '')
        files = ','.join(files)  # remove duplicates
        arg = f"--input={files} --vocab_size={vocab_size} --model_prefix={model_prefix}" \
            f" --model_type={model_type} --pad_id={PAD_TOK[1]} --bos_id={BOS_TOK[1]}" \
            f" --eos_id={EOS_TOK[1]} --unk_id={UNK_TOK[1]} --hard_vocab_limit=false"
        if char_coverage > 0:
            assert 0 < char_coverage <= 1
            arg += f" --character_coverage={char_coverage}"
        # CLS token goes in the beginning because we need it get index 4
        cls_tok_str = CLS_TOK[0]
        if no_split_toks:
            no_split_toks_str = ','.join([cls_tok_str] + no_split_toks)
        else:
            no_split_toks_str = cls_tok_str
        arg += f" --user_defined_symbols={no_split_toks_str}"
        if model_type == 'bpe':  # BPE can have longer sentences, default is 2048
            arg += " --max_sentence_length=8192"
        if model_type == 'word':
            arg += ' --use_all_vocab'
        log.info(f"SPM: {arg}")
        SentencePieceTrainer.Train(arg)
        log.info("Training complete")
        if not model_path.endswith('.model'):
            model_path += '.model'
        model = SPField(model_path)
        for piece, idx in RESERVED_TOKS:
            assert model.piece_to_id(piece) == idx
        return model


class NLField(Field):
    # from nlcodec lib

    def __init__(self, path: Union[str, Path]):
        # this is experimental
        from nlcodec import load_scheme, EncoderScheme, Type
        self.codec: EncoderScheme = load_scheme(path)
        self.vocab: List[Type] = self.codec.table
        log.info(f'Loaded {len(self.codec)} types from {path}')
        for tok, idx in RESERVED_TOKS:  # reserved are reserved
            # Todo swap it with nlcodec.Reserved
            assert self.vocab[idx].name == tok

    def encode_as_ids(self, text: str, add_bos=False, add_eos=False) -> List[int]:
        ids = self.codec.encode(text)
        if add_bos and ids[0] != BOS_TOK[1]:
            ids.insert(0, BOS_TOK[1])
        if add_eos and ids[-1] != EOS_TOK[1]:
            ids.append(EOS_TOK[1])
        return ids

    def decode_ids(self, ids: List[int], trunc_eos=False, remove_pads=True) -> str:
        if trunc_eos:
            try:
                ids = ids[:ids.index(EOS_TOK[1])]
            except ValueError:
                pass
        if remove_pads:
            ids = [i for i in ids if i != PAD_TOK_IDX]
        return self.codec.decode(ids)

    def tokenize(self, text: str) -> List[str]:
        return self.codec.encode_str(text)

    def detokenize(self, tokens: List[str]) -> str:
        return self.codec.decode_str(tokens)

    def __len__(self):
        return len(self.vocab)

    @classmethod
    def train(cls, model_type: str, vocab_size: int, model_path: str, files: List[str],
              no_split_toks: Optional[List[str]] = None, char_coverage: float = 0):
        """

        :param model_type: word, char, bpe
        :param vocab_size: vocabulary size
        :param model_path: where to store vocabulary model
        :param files: text for creating vcabulary
        :param no_split_toks:
        :param char_coverage: character coverage (0, 1]. value <= 0 => default coverage
        :return:
        """
        from nlcodec import learn_vocab
        inp = IO.get_liness(*files)
        assert not no_split_toks, 'not supported in nlcodec yet'
        kwargs = dict(char_coverage=char_coverage) if char_coverage > 0 else {}
        learn_vocab(inp=inp, level=model_type, model=model_path, vocab_size=vocab_size, **kwargs)
        return cls(model_path)


Example = namedtuple('Example', ['x', 'y'])
"""
An object of this class holds an example in sequence to sequence dataset
"""


class TSVData:

    def __init__(self, path: Union[str, Path], in_mem=False, shuffle=False, longest_first=True,
                 max_src_len: int = 512, max_tgt_len: int = 512, truncate: bool = False):
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
        self.truncate = truncate
        self.max_src_len, self.max_tgt_len = max_src_len, max_tgt_len
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
                x = self._parse(rec[0].strip())
                y = self._parse(rec[1].strip()) if len(rec) > 1 else None
                if self.truncate:  # truncate long recs
                    x = x[:self.max_src_len]
                    y = y if y is None else y[:self.max_tgt_len]
                elif len(x) > self.max_src_len or (0 if y is None else len(y)) > self.max_tgt_len:
                    continue  # skip long recs
                if not x or (y is not None and len(y) == 0):  # empty on one side
                    log.warning(f"Ignoring an empty record  x:{len(x)}    y:{len(y)}")
                    continue
                yield Example(x, y)

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
                               truncate: bool, src_len: int, tgt_len: int, src_tokenizer,
                               tgt_tokenizer) \
            -> Iterator[ParallelSeqRecord]:
        recs = TSVData.read_raw_parallel_lines(src_path, tgt_path)
        recs = ((src_tokenizer(x), tgt_tokenizer(y)) for x, y in recs)
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
    COUNT_ROWS = "SELECT COUNT(*) as COUNT from data"

    @classmethod
    def make_query(cls, sort_by: str, len_rand: int):
        assert len_rand >= 1
        template = "SELECT * from data ORDER BY %s + (RANDOM() %% %d) %s"
        known_queries = dict(y_len_asc=template % ('y_len', len_rand, 'ASC'),
                             y_len_desc=template % ('y_len', len_rand, 'DESC'),
                             x_len_asc=template % ('x_len', len_rand, 'ASC'),
                             x_len_desc=template % ('x_len', len_rand, 'DESC'),
                             random=cls.READ_RANDOM,
                             eq_len_rand_batch=template % ('y_len', len_rand, 'DESC'))
        assert sort_by in known_queries, ('sort_by must be one of ' + str(known_queries.keys()))
        return known_queries[sort_by]

    def __init__(self, path: Path, sort_by='random', len_rand=2,
                 max_src_len: int = 512, max_tgt_len: int = 512, truncate: bool = False):

        log.info(f"{type(self)} Args: {get_my_args()}")
        self.path = path
        assert path.exists()
        self.select_qry = self.make_query(sort_by, len_rand=len_rand)
        self.max_src_len, self.max_tgt_len = max_src_len, max_tgt_len
        self.truncate = truncate
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
        return self.db.execute(self.select_qry)

    def __iter__(self) -> Iterator[Example]:
        for d in self.read_all():
            x, y = d['x'], d.get('y')
            if not x or not y:
                log.warning(f"Ignoring an empty record   x:{len(x)}    y:{len(y)}")
                continue
            if len(x) > self.max_src_len or len(y) > self.max_tgt_len:
                if self.truncate:
                    x, y = x[:self.max_src_len], y[:self.max_tgt_len]
                else:  # skip this record
                    continue
            yield Example(x, y)

    def get_all(self, cols, sort):
        assert cols and sort
        qry = f"SELECT {', '.join(cols)} FROM data ORDER BY {sort}"
        return self.db.execute(qry)

    def get_all_ids(self, ids):
        ids_str = ",".join(map(str, ids))
        qry = f"SELECT * FROM  data WHERE id IN ({ids_str})"
        return self.db.execute(qry)

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

    # This should have been called as Dataset
    def __init__(self, data_path: Union[str, Path], batch_size: int,
                 sort_desc: bool = False, batch_first: bool = True, shuffle: bool = False,
                 sort_by: str = None, **kwargs):
        """
        Iterator for reading training data in batches
        :param data_path: path to TSV file
        :param batch_size: number of tokens on the target size per batch

        :param sort_desc: should the batch be sorted by src sequence len (useful for RNN api)
        """
        self.sort_desc = sort_desc
        self.batch_size = batch_size
        self.batch_first = batch_first
        self.sort_by = sort_by
        self.data_path = data_path
        if not isinstance(data_path, Path):
            data_path = Path(data_path)
        assert data_path.exists(), f'Invalid State: Training data doesnt exist;' \
            f' Please remove _PREPARED and rerun.'
        if data_path.name.endswith(".db"):
            self.data = SqliteFile(data_path, sort_by=sort_by, **kwargs)
        else:
            if sort_by:
                raise Exception(f'sort_by={sort_by} not supported for TSV data')
            self.data = TSVData(data_path, shuffle=shuffle, longest_first=False, **kwargs)
        log.info(f'Batch Size = {batch_size} toks, sort_by={sort_by}')

    def read_all(self):
        batch = []
        max_len = 0
        for ex in self.data:
            if min(len(ex.x), len(ex.y)) == 0:
                log.warn("Skipping a record,  either source or target is empty")
                continue

            this_len = max(len(ex.x), len(ex.y))
            if (len(batch) + 1) * max(max_len, this_len) <= self.batch_size:
                batch.append(ex)  # this one can go in
                max_len = max(max_len, this_len)
            else:
                if this_len > self.batch_size:
                    raise Exception(f'Unable to make a batch of {self.batch_size} toks'
                                    f' with a seq of x_len:{len(ex.x)} y_len:{len(ex.y)}')
                # yield the current batch
                yield Batch(batch, sort_dec=self.sort_desc, batch_first=self.batch_first)
                batch = [ex]  # new batch
                max_len = this_len
        if batch:
            log.debug(f"\nLast batch, size={len(batch)}")
            yield Batch(batch, sort_dec=self.sort_desc, batch_first=self.batch_first)

    def _make_eq_len_batch_ids(self):
        rows = self.data.get_all(cols=['id', 'x_len', 'y_len'], sort='y_len desc, random() desc')
        stats = [(r['id'], r['x_len'], r['y_len']) for r in rows]
        batches = []
        batch = []
        max_len = 0
        for id, x_len, y_len in stats:
            if min(x_len, y_len) == 0:
                log.warn("Skipping a record, either source or target is empty")
                continue

            this_len = max(x_len, y_len)
            if (len(batch) + 1) * max(max_len, this_len) <= self.batch_size:
                batch.append(id)  # this one can go in
                max_len = max(max_len, this_len)
            else:
                if this_len > self.batch_size:
                    raise Exception(f'Unable to make a batch of {self.batch_size} toks'
                                    f' with a seq of x_len:{x_len} y_len:{y_len}')
                batches.append(batch)
                batch = [id]  # new batch
                max_len = this_len
        if batch:
            batches.append(batch)
        return batches

    def make_eq_len_ran_batches(self):
        # every pass introduces some randomness
        batches = self._make_eq_len_batch_ids()
        log.info(f"length sorted random batches = {len(batches)}. Shuffling🔀...")
        if not batches:
            raise Exception(f'Found no training data. Please check config and {self.data_path}')
        random.shuffle(batches)

        for batch_ids in batches:
            batch = list(self.data.get_all_ids(batch_ids))
            batch = [Example(r['x'], r.get('y')) for r in batch]
            yield Batch(batch, sort_dec=self.sort_desc, batch_first=self.batch_first)

    def __iter__(self) -> Iterator[Batch]:
        if self.sort_by == 'eq_len_rand_batch':
            yield from self.make_eq_len_ran_batches()
        else:
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
