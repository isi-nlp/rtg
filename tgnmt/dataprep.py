import os
import glob
import torch
from . import my_tensor as tensor, device
import json
import numpy as np
from tgnmt import log
from collections import Counter
from typing import List, Dict, Iterator, Tuple, Optional

RawRecord = Tuple[List[str], List[str]]
SeqRecord = Tuple[List[int], List[int]]


BLANK_TOK = '-BLANK-', 0
UNK_TOK = '-UNK-', 1
BOS_TOK = '-BOS-', 2
EOS_TOK = '-EOS-', 3

RESERVED_TOKS = [BLANK_TOK, UNK_TOK, BOS_TOK, EOS_TOK]


class Field:
    def __init__(self, name: str, blank=False):
        self.name: str = name
        self.tok2idx: Dict[str, int] = {} if blank else {t: i for t, i in RESERVED_TOKS}
        self.idx2tok: List[str] = [] if blank else [t for t, _ in RESERVED_TOKS]
        self.freq: List[int] = [-1 for _ in self.idx2tok]

    def build_from(self, stream: Iterator[List[str]], min_freq: int = 1, max_vocab_size: int = 2 ** 24):
        tok_stream = (tok for seq in stream for tok in seq)  # flatten
        tok_freq = Counter(tok_stream).items()
        if min_freq > 1:
            tok_freq = [(t, f) for t, f in tok_freq if f >= min_freq]
        tok_freq = sorted(tok_freq, key=lambda x: x[1], reverse=True)
        if len(tok_freq) > max_vocab_size:
            log.info(f'Truncating vocab size from {len(tok_freq)} to {max_vocab_size}')
            tok_freq = tok_freq[:max_vocab_size]
        for tok, freq in tok_freq:
            self.add_token(tok, inc=freq)

    def add_token(self, tok: str, inc: int = 1):
        if tok in self.tok2idx:
            self.freq[self.tok2idx[tok]] += inc
        else:
            idx = len(self.idx2tok)
            self.tok2idx[tok] = idx
            self.idx2tok.append(tok)
            self.freq.append(inc)

    def seq2idx(self, toks: List[str], add_bos=False, add_eos=False) -> List[int]:
        seq = [self.tok2idx.get(tok, UNK_TOK[1]) for tok in toks]
        if add_bos:
            seq.insert(0, BOS_TOK[1])
        if add_eos:
            seq.append(EOS_TOK[1])
        return seq

    def idx2seq(self, indices: List[int]) -> List[str]:
        return [self.idx2tok[idx] for idx in indices]

    def size(self):
        return len(self.idx2tok)

    def dump_tsv(self, path: str):
        with open(path, 'w', encoding='utf-8') as f:
            f.write(f'{self.name}\n')
            for i, (tok, count) in enumerate(zip(self.idx2tok, self.freq)):
                f.write(f'{i}\t{tok}\t{count}\n')

    @staticmethod
    def load_tsv(path: str):
        with open(path, 'r', encoding='utf-8') as f:
            name = f.readline()
            field = Field(name, blank=True)
            i = 0
            for line in f:
                idx, tok, count = line.split('\t')
                idx, count = int(idx), int(count)
                assert idx == i, f'expected index {i}, got={idx}'
                field.add_token(tok, inc=count)
                i += 1
            return field


class Example:

    def __init__(self, x: List[int], y: List[int]=None):
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


class TranslationExperiment:
    def __init__(self, work_dir: str):
        log.info(f"Initializing an experiment. Directory = {work_dir}")
        self.work_dir = work_dir
        self.data_dir = os.path.join(work_dir, 'data')
        self.model_dir = os.path.join(self.work_dir, 'models')
        for _dir in [self.model_dir, self.data_dir]:
            if not os.path.exists(_dir):
                os.makedirs(_dir)
        self.args_file = os.path.join(self.model_dir, 'args.json')
        self.src_field_file = os.path.join(self.data_dir, 'src-field.tsv')
        self.src_field = Field.load_tsv(self.src_field_file) if os.path.exists(self.src_field_file) else None

        self.tgt_field_file = os.path.join(self.data_dir, 'tgt-field.tsv')
        self.tgt_field = Field.load_tsv(self.tgt_field_file) if os.path.exists(self.tgt_field_file) else None
        self.train_file = os.path.join(self.data_dir, 'train.tsv')
        self.valid_file = os.path.join(self.data_dir, 'valid.tsv')

    def has_prepared(self):
        return all([self.src_field, self.tgt_field, os.path.exists(self.train_file)])

    def has_trained(self):
        return self.get_last_saved_model()[0] is not None

    def prep_file(self, records: Iterator[RawRecord], path: str):
        seqs = ((self.src_field.seq2idx(sseq), self.tgt_field.seq2idx(tseq)) for sseq, tseq in records)
        seqs = ((' '.join(map(str, x)), ' '.join(map(str, y))) for x, y in seqs)
        lines = (f'{x}\t{y}\n' for x, y in seqs)

        log.info(f"Storing data at {path}")
        with open(path, 'w') as f:
            for line in lines:
                f.write(line)

    @staticmethod
    def read_raw_data(path: str, truncate: bool, src_len: int, tgt_len: int) -> Iterator[RawRecord]:
        """Raw dataset is a simple TSV with source \t target sequence of words"""
        recs = map(tokenize, read_tsv(path))
        recs = ((src, tgt) for src, tgt in recs if len(src) > 0 and len(tgt) > 0)
        if truncate:
            recs = ((src[:src_len], tgt[:tgt_len]) for src, tgt in recs)
        else:   # Filter out longer sentences
            recs = ((src, tgt) for src, tgt in recs if len(src) <= src_len and len(tgt) <= tgt_len)
        return recs

    def pre_process(self, train_file: str, valid_file: str, src_len: int, tgt_len: int, truncate=False):

        log.info(f'Training file:: {train_file}')
        # in memory  --  var len lists and sparse dictionary -- should be okay for now
        train_recs = list(self.read_raw_data(train_file, truncate, src_len, tgt_len))
        log.info(f'Found {len(train_recs)} records')
        self.src_field, self.tgt_field = Field('src'), Field('tgt')
        log.info("Building source vocabulary")
        self.src_field.build_from(x for x, _ in train_recs)
        log.info("Building target vocabulary")
        self.tgt_field.build_from(y for _, y in train_recs)
        log.info(f"Vocab sizes, source: {self.src_field.size()}, target:{self.tgt_field.size()}")
        self.src_field.dump_tsv(self.src_field_file)
        self.tgt_field.dump_tsv(self.tgt_field_file)

        self.prep_file(train_recs, self.train_file)
        val_recs = self.read_raw_data(valid_file, truncate, src_len, tgt_len)
        self.prep_file(val_recs, self.valid_file)
        args = {'src_vocab': self.src_field.size(), 'tgt_vocab': self.tgt_field.size()}
        self.store_model_args(args)

    def store_model(self, epoch: int, model, score: float, keep: int):
        """
        :param epoch: epoch number of model
        :param model: model object itself
        :param score: score of model
        :param keep: number of recent models to keep, older models will be deleted
        :return:
        """
        """saves model to a given path"""
        name = f'model_{epoch:03d}_{score:.4f}.pkl'
        path = os.path.join(self.model_dir, name)
        log.info(f"Saving epoch {epoch} to {path}")
        torch.save(model, path)
        for old_model in self.list_models()[keep:]:
            log.info(f"Deleting old {old_model} . Keep={keep}")
            os.remove(old_model)

    def list_models(self) -> List[str]:
        """
        Lists models in descending order of modification time
        :return: list of model paths
        """
        pat = f'{self.model_dir}/model_*.pkl'
        paths = glob.glob(pat)
        return sorted(paths, key=lambda p: os.path.getmtime(p), reverse=True)     # sort by descending time

    def get_last_saved_model(self) -> Tuple[Optional[str], int]:
        models = self.list_models()
        if models:
            _, epoch, score = models[0].replace('.pkl', '').split('_')
            return models[0], int(epoch)
        else:
            return None, -1

    def get_model_args(self) -> Optional[Dict]:
        """
        Gets args from file
        :return: args if exists or None otherwise
        """
        if not os.path.exists(self.args_file):
            return None
        with open(self.args_file, encoding='utf-8') as f:
            return json.load(f)

    def store_model_args(self, args):
        """
        Stores args to args file
        :param args: args to be stored
        :return:
        """
        with open(self.args_file, 'w', encoding='utf-8') as f:
            return json.dump(args, f, ensure_ascii=False)


def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subseq_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return (torch.from_numpy(subseq_mask) == 0).to(device)


class Batch:

    pad_value = BLANK_TOK[1]

    def __init__(self, batch: List[Example], sort_dec=True, batch_first=True):
        self.sort_dec = sort_dec
        self.batch_first = batch_first
        if sort_dec:
            batch = sorted(batch, key=lambda _: len(_.x), reverse=True)
        self._len = len(batch)
        self.max_x_len = len(batch[0].x) if sort_dec else max(len(ex.x) for ex in batch)
        self.x_seqs = torch.full((len(batch), self.max_x_len), fill_value=self.pad_value, dtype=torch.long,
                                 device=device)
        for i, ex in enumerate(batch):
            self.x_seqs[i, :len(ex.x)] = tensor(ex.x, dtype=torch.long)
        self.x_len = tensor([len(ex.x) for ex in batch], dtype=torch.long)
        if not batch_first:
            self.x_seqs = self.x_seqs.t()   # transpose
        self.x_mask = (self.x_seqs != self.pad_value).unsqueeze(-2)
        self.num_x_toks = self.x_len.sum().item()
        if batch[0].y:      # also has y_seq
            self.max_y_len = max(len(ex.y) for ex in batch)
            self.y_seqs = torch.full((len(batch), self.max_y_len), fill_value=self.pad_value, dtype=torch.long,
                                     device=device)
            self.y_len = tensor([len(ex.y) for ex in batch], dtype=torch.long)
            for i, ex in enumerate(batch):
                self.y_seqs[i, :len(ex.y)] = tensor(ex.y, dtype=torch.long)
            if not self.batch_first:
                self.y_seqs = self.y_seqs.t()  # transpose
            self.y_mask = self.make_std_mask(self.y_seqs)
            self.num_y_toks = self.y_len.sum().item()

    @staticmethod
    def make_std_mask(tgt, pad=pad_value):
        "Create a mask to hide padding AND future words."
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data)
        return tgt_mask

    def __len__(self):
        return self._len


class BatchIterable:

    def __init__(self, data_path: str, batch_size: int, sort_dec=True, in_mem_recs=False, in_mem_batch=False,
                 batch_first=True):
        """
        Iterator for reading training data in batches
        :param data_path: path to TSV file
        :param batch_size: number of examples per batch
        :param sort_dec: should the records within batch be sorted descending order of sequence length?
        :param in_mem_recs: should the raw records be held in memory?
        :param in_mem_batch: Should the batches be held in memory? Use True only if data set is extremely small
        :param batch_first: Shuld the first dimension be batch instead of sequence length
        """
        self.data = TSVData(data_path, in_mem=in_mem_recs)
        self.batch_size = batch_size
        self.sort_dec = sort_dec
        self.batch_first = batch_first
        self.mem = list(self.read_all()) if in_mem_batch else None

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
        yield from self.mem if self.mem is not None else self.read_all()

