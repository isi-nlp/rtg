import glob
import json
import os
from datetime import datetime
from itertools import chain
from typing import Optional, Dict, Any, Iterator, List, Tuple

import torch

from tgnmt import Field, log
from tgnmt.bpe import SubwordTextEncoder
from tgnmt.dataprep import RawRecord, tokenize, read_tsv, TokStream


class TranslationExperiment:

    def __init__(self, work_dir: str, read_only=False):
        log.info(f"Initializing an experiment. Directory = {work_dir}")
        self.read_only = read_only
        self.work_dir = work_dir
        self.data_dir = os.path.join(work_dir, 'data')
        self.model_dir = os.path.join(self.work_dir, 'models')
        self.config_file = os.path.join(self.work_dir, 'config.json')
        self.src_vocab_file = os.path.join(self.data_dir, 'src-vocab.tsv')
        self.tgt_vocab_file = os.path.join(self.data_dir, 'tgt-vocab.tsv')
        self.shared_vocab_file = os.path.join(self.data_dir, 'vocab.tsv')
        self.train_file = os.path.join(self.data_dir, 'train.tsv')
        self.valid_file = os.path.join(self.data_dir, 'valid.tsv')

        if read_only:
            for _dir in [self.work_dir, self.data_dir, self.model_dir]:
                assert os.path.isdir(_dir), f'{os.path.realpath(_dir)} doesnt exist'
        else:
            for _dir in [self.model_dir, self.data_dir]:
                if not os.path.exists(_dir):
                    os.makedirs(_dir)

        self.src_vocab = Field.load_tsv(self.src_vocab_file) if os.path.exists(self.src_vocab_file) else None
        self.tgt_vocab = Field.load_tsv(self.tgt_vocab_file) if os.path.exists(self.tgt_vocab_file) else None
        self.shared_vocab = Field.load_tsv(self.shared_vocab_file) if os.path.exists(self.shared_vocab_file) else None
        self.config = self.load_config()

    def load_config(self) -> Optional[Dict[str, Any]]:
        if os.path.exists(self.config_file):
            with open(self.config_file, encoding='utf-8') as f:
                return json.load(f)
        return None

    def store_config(self):
        with open(self.config_file, 'w', encoding='utf-8') as fp:
            return json.dump(self.config, fp, ensure_ascii=False)

    def has_prepared(self):
        vocab_found = self.shared_vocab is not None or all([self.src_vocab, self.tgt_vocab])
        return vocab_found and os.path.exists(self.train_file)

    def has_trained(self):
        return self.get_last_saved_model()[0] is not None

    def get_vocab(self, side: str) -> Field:
        if self.shared_vocab:
            return self.shared_vocab
        assert side in ('src', 'tgt')
        return self.src_vocab if side == 'src' else self.tgt_vocab

    def prep_file(self, records: Iterator[RawRecord], path: str):

        seqs = ((self.get_vocab('src').seq2idx(sseq),
                 self.get_vocab('tgt').seq2idx(tseq, add_bos=True)) for sseq, tseq in records)
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
        else:  # Filter out longer sentences
            recs = ((src, tgt) for src, tgt in recs if len(src) <= src_len and len(tgt) <= tgt_len)
        return recs

    @staticmethod
    def _make_subword_vocb(name: str, tok_stream: TokStream, max_toks: int, min_count: int):
        tok_freq = Field.get_tok_freq(tok_stream)
        shared_enc = SubwordTextEncoder.build_to_target_size(target_size=max_toks, token_counts=dict(tok_freq),
                                                             min_val=min_count, max_val=5 * min_count)
        subtoks = shared_enc.all_subtoken_strings
        vocab = Field(name, subword_enc=shared_enc).build_from_types(subtoks)
        log.info(f"Built subword vocabulary of {vocab.size()} ")
        return vocab

    def pre_process(self, train_file: str, valid_file: str, src_len: int, tgt_len: int, truncate=False, min_count=1,
                    src_types: Optional[int] = None, tgt_types: Optional[int] = None, bpe=False, shared=False):

        log.info(f'Training file:: {train_file}')
        # in memory  --  var len lists and sparse dictionary -- should be okay for now
        train_recs = list(self.read_raw_data(train_file, truncate, src_len, tgt_len))
        log.info(f'Found {len(train_recs)} records')
        src_recs = [x for x, _ in train_recs]
        tgt_recs = [y for _, y in train_recs]
        if shared:
            shared_input = chain(src_recs, tgt_recs)
            if bpe:
                log.info(f'Going to do shared subword vocabulary of {tgt_types} types')
                self.shared_vocab = self._make_subword_vocb('shared', shared_input, tgt_types, min_count=min_count)
            else:
                log.info(f'Going to do shared vocabulary of {tgt_types} types')
                self.shared_vocab = Field('shared').build_from(shared_input,
                                                               min_freq=min_count, max_vocab_size=tgt_types)
            log.info(f"Shared vocab size: {self.shared_vocab.size()}")
        else:
            log.info("Building source and target vocabulary")
            if bpe:
                self.src_vocab = self._make_subword_vocb('src', src_recs, max_toks=src_types, min_count=min_count)
                self.tgt_vocab = self._make_subword_vocb('tgt', tgt_recs, max_toks=tgt_types, min_count=min_count)
            else:
                self.src_vocab, self.tgt_vocab = Field('src'), Field('tgt')
                self.src_vocab.build_from(src_recs, max_vocab_size=src_types)
                self.tgt_vocab.build_from(tgt_recs, max_vocab_size=tgt_types)
            log.info(f"Vocab sizes, source: {self.src_vocab.size()}, target:{self.tgt_vocab.size()}")
        self.prep_file(train_recs, self.train_file)
        val_recs = self.read_raw_data(valid_file, truncate, src_len, tgt_len)
        self.prep_file(val_recs, self.valid_file)
        # update state on disk
        self.persist_state()

    def persist_state(self):
        """Writes state of current experiment to the disk"""
        assert not self.read_only
        if self.config is None:
            self.config = {}

        for vocab, f_path in [(self.src_vocab, self.src_vocab_file),
                              (self.tgt_vocab, self.tgt_vocab_file),
                              (self.shared_vocab, self.shared_vocab_file)]:
            if vocab is not None:
                vocab.dump_tsv(f_path)

        args = self.config.get('model_args', {})
        self.config['model_args'] = args
        shared_vocab_size = self.shared_vocab.size() if self.shared_vocab else None
        args['src_vocab'] = shared_vocab_size if shared_vocab_size else self.src_vocab.size()
        args['tgt_vocab'] = shared_vocab_size if shared_vocab_size else self.tgt_vocab.size()
        self.config['shared'] = self.shared_vocab is not None
        self.config['updated_att'] = datetime.now().isoformat()
        self.store_config()

    def store_model(self, epoch: int, model, score: float, keep: int):
        """
        saves model to a given path
        :param epoch: epoch number of model
        :param model: model object itself
        :param score: score of model
        :param keep: number of recent models to keep, older models will be deleted
        :return:
        """
        assert not self.read_only
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
        return sorted(paths, key=lambda p: os.path.getmtime(p), reverse=True)  # sort by descending time

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
        return self.config.get('model_args')
