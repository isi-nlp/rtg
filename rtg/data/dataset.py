import math
import os
import pickle
import random
import sqlite3
from itertools import zip_longest
from pathlib import Path
from typing import List, Iterator, Tuple, Union, Iterable, Dict, Any, Optional
import torch
from tqdm import tqdm
import numpy as np

from rtg import log, device, cpu_device
from rtg.data.codec import Field
from rtg.utils import IO, line_count, get_my_args, max_RSS, maybe_compress


Array = np.ndarray
RawRecord = Tuple[str, str]
TokRawRecord = Tuple[List[str], List[str]]
MonoSeqRecord = List[Union[int, str]]
ParallelSeqRecord = Tuple[MonoSeqRecord, MonoSeqRecord]
TokStream = Union[Iterator[Iterator[str]], Iterator[str]]


class IdExample:
    __slots__ = 'x', 'y', 'id', 'x_raw', 'y_raw', 'x_len', 'y_len'

    def __init__(self, x, y, id):
        self.x: Array = x
        self.y: Array = y
        self.id = id
        self.x_raw: Optional[str] = None
        self.y_raw: Optional[str] = None

    def val_exists_at(self, side, pos: int, exist: bool, val:int):
        assert side == 'x' or side == 'y'
        assert pos == 0 or pos == -1
        seq = self.x if side == 'x' else self.y
        if exist:
            if seq[pos] != val:
                if pos == 0:
                    seq = np.append(np.int32(val), seq)
                else: # pos = -1
                    seq = np.append(seq, np.int32(val))
                # update
                if side == 'x':
                    self.x = seq
                else:
                    self.y = seq
        else:  # should not have val at pos
            assert seq[pos] != val


    def __getitem__(self, key):
        if key == 'x_len':
            return len(self.x)
        elif key == 'y_len':
            return len(self.y)
        else:
            return getattr(self, key)



class NLDbExample(IdExample):
    """
    # NLDd has (id, x, y) where as here (x, y, id) ; I think NLDb is doing correctly
    """
    __slots__ = 'x', 'y', 'id'
    def __init__(self, id, x, y):
        super().__init__(x, y, id)


class TSVData(Iterable[IdExample]):

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

    def read_all(self) -> Iterator[IdExample]:
        with IO.reader(self.path) as lines:
            recs = (line.split('\t') for line in lines)
            for idx, rec in enumerate(recs):
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
                yield IdExample(x, y, id=idx)

    def __len__(self):
        return self._len

    def __iter__(self) -> Iterator[IdExample]:
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


class TokenizerTask:
    """Works with Parallel data"""

    def __init__(self, tokenizers: List, lengths: List[int], truncate: bool):
        assert len(tokenizers) == len(lengths)
        self.tokenizers = tokenizers
        self.lengths = lengths
        self.truncate = truncate

    def __call__(self, record):
        record = [tokr(col) for col, tokr in zip(record, self.tokenizers)]
        if self.truncate:
            record = [col[:max_len] for col, max_len in zip(record, self.lengths)]
        else:
            # filter out long sequences if any of the column is long
            if any(len(col) > max_len for col, max_len in zip(record, self.lengths)):
                record = None
        return record

class InMemoryData:

    def __init__(self, stream: Iterator[IdExample]):
        self.data = []
        self.ids: Dict[Any, int] = {}
        log.info("Loading data to memory")

        with tqdm(stream, mininterval=1, unit='recs') as data_bar:
            for idx, rec in enumerate(data_bar):
                assert isinstance(rec, IdExample)
                assert rec.id not in self.ids, f'Record with id {id} is a duplicate record'
                self.ids[rec.id] = len(self.data)
                self.data.append(rec)

                if idx % 1000 == 0:
                    mem = max_RSS()[1]
                    data_bar.set_postfix(mem=mem, refresh=False)
        log.info(f"Total={len(self.data)} records; Total memory used={max_RSS()[1]}")
        assert len(self.data) == len(self.ids)

    def get_all(self, cols, sort):
        assert cols
        data = self.data
        if sort:
            sort_col, sort_order = sort.split()
            assert sort_col in {'x_len', 'y_len'}
            assert sort_order in {'asc', 'desc'}
            reverse = sort_order == 'desc'
            data = sorted(data, key=lambda x: x[sort_col], reverse=reverse)
        recs = ({cn: ex[cn] for cn in cols} for ex in data)
        return recs

    def get_all_ids(self, ids):
        idxs = (self.ids[id] for id in ids)
        examples = (self.data[ix] for ix in idxs)
        return examples

    def __len__(self):
        return len(self.data)

    def __iter__(self) -> Iterator[IdExample]:
        yield from self.data


class SqliteFile(Iterable[IdExample]):
    """
    Change log::
    VERSION 0: (unset)
        x_seq and y_seq were list of integers, picked using pickle.dumps
        very inefficient
    VERSION 1:
        x_seq and y_seq were np.array(, dtyp=np.int32).tobytes()

    """
    CUR_VERSION = 1

    TABLE_STATEMENT = f"""CREATE TABLE IF NOT EXISTS data (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        x BLOB NOT NULL,
        y BLOB,
        x_len INTEGER,
        y_len INTEGER);"""
    INDEX_X_LEN = "CREATE INDEX IF NOT EXISTS  idx_x_len ON data (x_len);"
    INDEX_Y_LEN = "CREATE INDEX IF NOT EXISTS  idx_y_len ON data (y_len);"

    INSERT_STMT = "INSERT INTO data (x, y, x_len, y_len) VALUES (?, ?, ?, ?)"
    READ_RANDOM = "SELECT * from data ORDER BY RANDOM()"
    COUNT_ROWS = "SELECT COUNT(*) as COUNT from data"

    @classmethod
    def make_query(cls, sort_by: str, len_rand: int):
        assert len_rand >= 1
        select_no_sort = 'SELECT * from data'
        template = f"{select_no_sort} ORDER BY %s + (RANDOM() %% %d) %s"
        known_queries = dict(y_len_asc=template % ('y_len', len_rand, 'ASC'),
                             y_len_desc=template % ('y_len', len_rand, 'DESC'),
                             x_len_asc=template % ('x_len', len_rand, 'ASC'),
                             x_len_desc=template % ('x_len', len_rand, 'DESC'),
                             random=cls.READ_RANDOM,
                             eq_len_rand_batch=template % ('y_len', len_rand, 'DESC'))
        known_queries[None] = known_queries['none'] = select_no_sort
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
        self.db_version = self.db.execute('PRAGMA user_version;').fetchone()[0]

        def dict_factory(cursor, row):  # map tuples to dictionary with column names
            d = {}
            for idx, col in enumerate(cursor.description):
                key = col[0]
                val = row[idx]
                if key in ('x', 'y') and val is not None:
                    if self.db_version < 1:
                        val = pickle.loads(val)  # unmarshall
                        val = np.array(val, dtype=np.int32)
                    else: # version 1 and above
                        val = np.frombuffer(val, dtype=np.int32)
                d[key] = val
            return d

        self.db.row_factory = dict_factory

    def __len__(self):
        return self.db.execute(self.COUNT_ROWS).fetchone()['COUNT']

    def read_all(self) -> Iterator[Dict[str, Any]]:
        return self.db.execute(self.select_qry)

    def __iter__(self) -> Iterator[IdExample]:
        for d in self.read_all():
            id, x, y = d['id'], d['x'], d.get('y')
            if x is None or y is None or len(x) == 0 or len(y) == 0:
                log.warning(f"Ignoring an empty record   x:{len(x)}    y:{len(y)}")
                continue
            if len(x) > self.max_src_len or len(y) > self.max_tgt_len:
                if self.truncate:
                    x, y = x[:self.max_src_len], y[:self.max_tgt_len]
                else:  # skip this record
                    continue
            yield IdExample(x=x, y=y, id=id)

    def get_all(self, cols, sort=None):
        assert cols
        qry = f"SELECT {', '.join(cols)} FROM data"
        if sort:
            qry += f' ORDER BY {sort}'
        return self.db.execute(qry)

    def get_all_ids(self, ids):
        ids_str = ",".join(map(str, ids))
        qry = f"SELECT * FROM  data WHERE id IN ({ids_str})"
        recs = (IdExample(x=rec['x'], y=rec.get('y'), id=rec['id']) for rec in self.db.execute(qry))
        return recs

    @classmethod
    def write(cls, path, records: Iterator[ParallelSeqRecord]):
        if path.exists():
            log.warning(f"Overwriting {path} with new records")
            os.remove(str(path))
        maybe_tmp = IO.maybe_tmpfs(path)
        log.info(f'Creating {maybe_tmp}')
        conn = sqlite3.connect(str(maybe_tmp))
        cur = conn.cursor()
        cur.execute(cls.TABLE_STATEMENT)
        cur.execute(cls.INDEX_X_LEN)
        cur.execute(cls.INDEX_Y_LEN)
        cur.execute(f"PRAGMA user_version = {cls.CUR_VERSION};")

        count = 0
        for x_seq, y_seq in records:
            # use numpy. its a lot efficient
            if not isinstance(x_seq, np.ndarray):
                x_seq = np.array(x_seq, dtype=np.int32)
            if y_seq is not None and not isinstance(y_seq, np.ndarray):
                y_seq = np.array(y_seq, dtype=np.int32)
            values = (x_seq.tobytes(),
                      None if y_seq is None else y_seq.tobytes(),
                      len(x_seq), len(y_seq) if y_seq is not None else -1)
            cur.execute(cls.INSERT_STMT, values)
            count += 1
        cur.close()
        conn.commit()
        if maybe_tmp != path:
            # bring the file back to original location where it should be
            IO.copy_file(maybe_tmp, path)
        log.info(f"stored {count} rows in {path}")


def read_tsv(path: str):
    assert os.path.exists(path)
    with IO.reader(path) as f:
        yield from (line.split('\t') for line in f)


def tokenize(strs: List[str]) -> List[List[str]]:
    return [s.split() for s in strs]


def subsequent_mask(size, device=device):
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


def padded_sequence_mask(lengths, max_len=None, device=device):
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
    _x_attrs = ['x_len', 'x_seqs']
    _y_attrs = ['y_len', 'y_seqs', 'ys']
    _all_attrs = _x_attrs + _y_attrs

    def __init__(self, batch: List[IdExample], sort_dec=False, batch_first=True,
                 add_eos_x=True, add_eos_y=True, add_bos_x=False, add_bos_y=False,
                 y_is_cls=False, field: Field = None, device=cpu_device):
        """

        :param batch: List fo Examples
        :param sort_dec: sort by descending of lengths
        :param batch_first: batch dim first, (false => time dim first)
        :param add_eos_x: append eos to x seqs; false => make sure no eos at x's end
        :param add_eos_y: append eos to y seqs; false => make sure no eos at y's end
        :param add_bos_x: prepend bos to x seqs; false => make sure no bos at x's front
        :param add_bos_y: prepend bos to y seqs; false => make sure no bos at y's front
        :param y_is_cls: y is a class. ignore eos, bos things on y seqs
        :param field:
        :param device:
        """
        """
        :param batch: 
        :param sort_dec: True if the examples be sorted as descending order of their source sequence lengths
        :Param Batch_First: first dimension is batch
        
        """
        assert field
        self.bos_val: int = field.bos_idx
        self.eos_val: int = field.eos_idx
        self.pad_val: int = field.pad_idx
        self.eos_x = add_eos_x
        self.eos_y = add_eos_y
        self.bos_x = add_bos_x
        self.bos_y = add_bos_y
        self.y_is_cls = y_is_cls
        self.batch_first = batch_first

        self.bos_eos_check(batch, 'x', add_bos_x, add_eos_x)
        if sort_dec:
            batch = sorted(batch, key=lambda _: len(_.x), reverse=True)
        self._len = len(batch)
        self.x_len = torch.tensor([len(e.x) for e in batch], device=device)
        self.x_toks = self.x_len.sum().float().item()
        self.max_x_len = self.x_len.max()

        # create x_seqs on CPU RAM and move to GPU at once
        self.x_seqs = torch.full(size=(self._len, self.max_x_len), fill_value=self.pad_val,
                                 dtype=torch.long, device=device)
        for i, ex in enumerate(batch):
            self.x_seqs[i, :len(ex.x)] = torch.tensor(ex.x, dtype=torch.long, device=device)
        self.x_seqs = self.x_seqs.to(device)
        if not batch_first:  # transpose
            self.x_seqs = self.x_seqs.t()
        self.x_raw = None
        if batch[0].x_raw:
            self.x_raw = [ex.x_raw for ex in batch]

        first_y = batch[0].y
        self.has_y = first_y is not None
        if self.has_y:
            if y_is_cls:
                ys = torch.full(size=(self._len,), fill_value=self.pad_val,
                                    dtype=torch.long, device=device)
                for i, ex in enumerate(batch):
                    y = ex.y
                    if hasattr(y, '__len__'):
                        assert len(y) == 1
                        y = y[0]
                    ys[i] = y
                self.ys = ys.to(device)
            else:
                self.bos_eos_check(batch, 'y', add_bos_y, add_eos_y)
                self.y_len = torch.tensor([len(e.y) for e in batch], device=device)
                self.y_toks = self.y_len.sum().float().item()
                self.max_y_len = self.y_len.max().item()
                y_seqs = torch.full(size=(self._len, self.max_y_len), fill_value=self.pad_val,
                                    dtype=torch.long, device=device)
                for i, ex in enumerate(batch):
                    y_seqs[i, :len(ex.y)] = torch.tensor(ex.y, dtype=torch.long)
                self.y_seqs = y_seqs.to(device)
                if not batch_first:  # transpose
                    self.y_seqs = self.y_seqs.t()
                self.y_raw = None
                if batch[0].y_raw:
                    self.y_raw = [ex.y_raw for ex in batch]

    def bos_eos_check(self, batch: List[IdExample], side: str, bos: bool, eos: bool):
        """
        ensures and inserts (if needed) EOS and BOS tokens
        :param batch:
        :param side: which side? choices: {'x', 'y'}
        :param bos: True if should have BOS, False if should not have BOS
        :param eos: True if should have EOS, False if should not have EOS
        :return: None, all modifications are inplace of batch
        """
        for ex in batch:
            ex.val_exists_at(side, pos=0, exist=bos, val=self.bos_val)
            ex.val_exists_at(side, pos=-1, exist=eos, val=self.eos_val)

    def __len__(self):
        return self._len

    def to(self, device):
        """Move this batch to given device"""
        for name in self._all_attrs:
            if hasattr(self, name):
                setattr(self, name, getattr(self, name).to(device))
        return self

    def make_autoreg_mask(self, tgt):
        "Create a mask to hide padding and future words for autoregressive generation."
        return self.make_autogres_mask_(tgt, self.pad_val)

    @staticmethod
    def make_autogres_mask_(tgt, pad_val: int):
        "Create a mask to hide padding and future words."
        tgt_mask = (tgt != pad_val).unsqueeze(1)
        tgt_mask = tgt_mask & subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data)
        return tgt_mask


class BatchIterable(Iterable[Batch]):

    # This should have been called as Dataset
    def __init__(self, data_path: Union[str, Path], batch_size:Union[int, Tuple[int,int]], field: Field,
                 sort_desc: bool = False, batch_first: bool = True, shuffle: bool = False,
                 sort_by: str = None, keep_in_mem=False, raw_path: Tuple[Path]=None,
                 device=cpu_device, y_is_cls=False, **kwargs):
        """
        Iterator for reading training data in batches
        :param data_path: path to TSV file
        :param batch_size: number of tokens on the target size per batch
        :param raw_path: (src, tgt) paths for loading the sentences (optional); use it for validation
               required: keep_mem=true, shuffle=False, sort_by=None
        :param keep_in_mem: keep the dataset in-memory
        :param sort_desc: should the batch be sorted by src sequence len (useful for RNN api)
        """
        self.field = field
        self.sort_desc = sort_desc
        
        if isinstance(batch_size, int):
            self.max_toks, self.max_sents = batch_size, batch_size
        else:
            self.max_toks, self.max_sents = batch_size
        self.batch_first = batch_first
        self.sort_by = sort_by
        self.data_path = data_path
        self.keep_in_mem = keep_in_mem
        self.y_is_cls = y_is_cls
        self.device = device
        if not isinstance(data_path, Path):
            data_path = Path(data_path)

        assert data_path.exists(), f'Invalid State: Training data doesnt exist;' \
            f' Please remove _PREPARED and rerun.'
        self.data_path = data_path

        if any([data_path.name.endswith(suf) for suf in ('.nldb', '.nldb.tmp')]):
            assert sort_by == 'random', f'sort_by={sort_by} is not supported for nldb. Try "random"'
            from nlcodec.db import MultipartDb
            self.data = MultipartDb.load(data_path, shuffle=shuffle, rec_type=NLDbExample)
            self.n_batches = -1
        elif any([data_path.name.endswith(suf) for suf in ('.db', '.db.tmp')]):
            self.data = SqliteFile(data_path, sort_by=sort_by, **kwargs)
            self.n_batches = len(self._make_eq_len_batch_ids())
        else:
            if sort_by:
                raise Exception(f'sort_by={sort_by} not supported for TSV data')
            self.data = TSVData(data_path, shuffle=shuffle, longest_first=False, **kwargs)
            self.n_batches = len(self.data)

        if raw_path:  # for logging and validation BLEU
            # Only narrower use case is supported
            assert not shuffle
            assert not sort_by
            assert keep_in_mem
            assert len(raw_path) == 2, 'both src and tgt should be given'
        if self.keep_in_mem:
            in_mem_file = data_path.with_suffix(".memdb.pkl")
            if in_mem_file.exists():
                log.info(f"Loading from {in_mem_file}")
                with in_mem_file.open('rb') as rdr:
                    self.data = pickle.load(rdr)
            else:
                self.data = InMemoryData(self.data)
                if raw_path:         # raw data for logging
                    src_raw, tgt_raw = raw_path[0], raw_path[1]
                    log.info(f"Reading raw from src:{src_raw} tgt:{tgt_raw}")
                    raw_data = list(TSVData.read_raw_parallel_lines(src_raw, tgt_raw))
                    if len(raw_data) == len(self.data):
                        for idx, (src, tgt) in enumerate(raw_data):
                            self.data.data[idx].x_raw = src
                            self.data.data[idx].y_raw = tgt
                    else:
                        log.warning(f'Raw={len(raw_data)}, but bin={len(self.data)} segs '
                            f'Try setting prep.truncate=true to truncate instead of skip of recs.')
                        log.warning("This disables BLEU logging on validation")
                log.info(f"saving in-memory to {in_mem_file}")
                with in_mem_file.open('wb') as wrt:
                    pickle.dump(self.data, wrt)
        log.info(f'Batch Size = {batch_size} toks, sort_by={sort_by}')

    def read_all(self):
        batch = []
        max_len = 0
        for ex in self.data:
            if min(len(ex.x), len(ex.y)) == 0:
                log.warn("Skipping a record,  either source or target is empty")
                continue

            this_len = max(len(ex.x), len(ex.y))
            if len(batch) < self.max_sents and (len(batch) + 1) * max(max_len, this_len) <= self.max_toks:
                batch.append(ex)  # this one can go in
                max_len = max(max_len, this_len)
            else:
                if this_len > self.max_toks:
                    raise Exception(f'Unable to make a batch of {self.max_toks} toks'
                                    f' with a seq of x_len:{len(ex.x)} y_len:{len(ex.y)}')
                # yield the current batch
                yield Batch(batch, sort_dec=self.sort_desc, batch_first=self.batch_first,
                            field=self.field, device=self.device, y_is_cls=self.y_is_cls)
                batch = [ex]  # new batch
                max_len = this_len
        if batch:
            log.debug(f"\nLast batch, size={len(batch)}")
            yield Batch(batch, sort_dec=self.sort_desc, batch_first=self.batch_first,
                        field=self.field, device=self.device, y_is_cls=self.y_is_cls)

    def _make_eq_len_batch_ids(self):
        sort = 'y_len desc'
        if isinstance(self.data, SqliteFile):  # only sqlite supports multiple sorts as of now
            sort += ', random() desc'
        rows = self.data.get_all(cols=['id', 'x_len', 'y_len'], sort=sort)
        batches = []
        batch = []
        max_len = 0

        for row in rows:
            id, x_len, y_len = row['id'], row['x_len'], row['y_len']
            if min(x_len, y_len) == 0:
                log.warn("Skipping a record, either source or target is empty")
                continue

            this_len = max(x_len, y_len)
            if len(batch) < self.max_sents and (len(batch) + 1) * max(max_len, this_len) <= self.max_toks:
                batch.append(id)  # this one can go in
                max_len = max(max_len, this_len)
            else:
                if this_len > self.max_toks:
                    raise Exception(f'Unable to make a batch of {self.max_toks} toks'
                                    f' with a seq of x_len:{x_len} y_len:{y_len}')
                batches.append(maybe_compress(batch))
                batch = [id]  # new batch
                max_len = this_len
        if batch:
            batches.append(maybe_compress(batch))
        return batches

    def make_eq_len_ran_batches(self):
        # every pass introduces some randomness
        batches = self._make_eq_len_batch_ids()
        self.n_batches = len(batches)
        log.info(f"length sorted random batches = {len(batches)}. Shuffling🔀...")
        if not batches:
            raise Exception(f'Found no training data. Please check config and {self.data_path}')
        random.shuffle(batches)

        for batch_ids in batches:
            batch = list(self.data.get_all_ids(batch_ids))
            # batch = [Example(r['x'], r.get('y')) for r in batch]
            yield Batch(batch, sort_dec=self.sort_desc, batch_first=self.batch_first,
                        field=self.field, device=self.device, y_is_cls=self.y_is_cls)

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
        return self.n_batches


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


class GenerativeBatchIterable(Iterable[Batch]):

    def __init__(self, file_creator: callable, batches: int, batch_size: int, field: Field,
                 dynamic_epoch: bool = False, batch_first: bool = True, shuffle: bool = True,
                 sort_by: str = 'eq_len_rand_batch', **kwargs):
        self.file_creator = file_creator
        self.total = batches
        self.batch_size = batch_size
        self.field = field
        self.dynamic_epoch = dynamic_epoch
        self.batch_first = batch_first
        self.shuffle = shuffle
        self.sort_by = sort_by
        self.kwargs = kwargs
        self.count = 0
        self.feed = True

    def generate_data(self) -> Iterable[Batch]:
        file_name = self.file_creator()
        data = BatchIterable(
            file_name, batch_size=self.batch_size, field=self.field, sort_by=self.sort_by,
            batch_first=self.batch_first, shuffle=self.shuffle, **self.kwargs
        )
        return data

    def __iter__(self) -> Iterator[Batch]:
        data = self.generate_data()

        while self.feed:
            for batch in data:
                yield batch
                self.count += 1
                if self.count >= self.total:
                    self.feed = False
                    break

            if self.dynamic_epoch:
                data = self.generate_data()
