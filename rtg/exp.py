import glob
import yaml
import os
from datetime import datetime
from typing import Optional, Dict, Iterator, List, Tuple
import torch

from rtg import log, load_conf
from rtg.dataprep import RawRecord, SeqRecord, Field


class TranslationExperiment:

    def __init__(self, work_dir: str, read_only=False, config=None):
        log.info(f"Initializing an experiment. Directory = {work_dir}")
        self.read_only = read_only
        self.work_dir = work_dir
        self.data_dir = os.path.join(work_dir, 'data')
        self.model_dir = os.path.join(self.work_dir, 'models')
        self._config_file = os.path.join(self.work_dir, 'conf.yml')
        self._shared_field_file = os.path.join(self.data_dir, 'sentpiece.shared.model')
        self.train_file = os.path.join(self.data_dir, 'train.tsv')
        self.valid_file = os.path.join(self.data_dir, 'valid.tsv')

        if not read_only:
            for _dir in [self.model_dir, self.data_dir]:
                if not os.path.exists(_dir):
                    os.makedirs(_dir)
        if type(config) is str:
            config = load_conf(config)
        self.config = config if config else load_conf(self._config_file)
        self.shared_field = Field(self._shared_field_file) if os.path.exists(self._shared_field_file) else None

    def store_config(self):
        with open(self._config_file, 'w', encoding='utf-8') as fp:
            return yaml.dump(self.config, fp)

    @property
    def model_type(self) -> Optional[str]:
        return self.config.get('model_type')

    @model_type.setter
    def model_type(self, mod_type: str):
        self.config['model_type'] = mod_type

    def has_prepared(self):
        return self.shared_field and os.path.exists(self.train_file) and os.path.exists(self.valid_file)

    def has_trained(self):
        return self.get_last_saved_model()[0] is not None

    @staticmethod
    def write_tsv(records: Iterator[SeqRecord], path: str):
        seqs = ((' '.join(map(str, x)), ' '.join(map(str, y))) for x, y in records)
        lines = (f'{x}\t{y}\n' for x, y in seqs)
        log.info(f"Storing data at {path}")
        with open(path, 'w', encoding='utf-8') as f:
            for line in lines:
                f.write(line)

    @staticmethod
    def read_raw_lines(src_path: str, tgt_path: str) -> Iterator[RawRecord]:
        with open(src_path) as src_lines, open(tgt_path) as tgt_lines:
            recs = ((src.strip(), tgt.strip()) for src, tgt in zip(src_lines, tgt_lines))
            recs = ((src, tgt) for src, tgt in recs if src and tgt)
            yield from recs

    def read_raw_data(self, src_path: str, tgt_path: str, truncate: bool, src_len: int, tgt_len: int, tokenizer)\
            -> Iterator[SeqRecord]:
        recs = self.read_raw_lines(src_path, tgt_path)
        recs = ((tokenizer(x), tokenizer(y)) for x, y in recs)
        if truncate:
            recs = ((src[:src_len], tgt[:tgt_len]) for src, tgt in recs)
        else:  # Filter out longer sentences
            recs = ((src, tgt) for src, tgt in recs if len(src) <= src_len and len(tgt) <= tgt_len)
        return recs

    def pre_process(self, args=None):
        args = args if args else self.config['prep']

        files = [args['train_src'], args['train_tgt']]
        for val in [args.get('mono_src'), args.get('mono_tgt')]:
            if val:
                files.extend(val)
        self.shared_field = Field.train(self.config['pieces'], self.config['vocab_size'], self._shared_field_file, files)

        # create Piece IDs
        train_recs = self.read_raw_data(args['train_src'], args['train_tgt'], args['truncate'],
                                        args['src_len'], args['tgt_len'], tokenizer=self.src_vocab.encode_as_ids)
        self.write_tsv(train_recs, self.train_file)
        val_recs = self.read_raw_data(args['valid_src'], args['valid_tgt'], args['truncate'],
                                      args['src_len'], args['tgt_len'], tokenizer=self.tgt_vocab.encode_as_ids)
        self.write_tsv(val_recs, self.valid_file)

        # Redo again as Pieces
        train_recs = self.read_raw_data(args['train_src'], args['train_tgt'], args['truncate'],
                                        args['src_len'], args['tgt_len'], tokenizer=self.src_vocab.tokenize)
        self.write_tsv(train_recs, self.train_file.replace('.tsv', '.pieces.tsv'))
        val_recs = self.read_raw_data(args['valid_src'], args['valid_tgt'], args['truncate'],
                                      args['src_len'], args['tgt_len'], tokenizer=self.tgt_vocab.tokenize)
        self.write_tsv(val_recs, self.valid_file.replace('.tsv', '.pieces.tsv'))

        # update state on disk
        self.persist_state()

    def persist_state(self):
        """Writes state of current experiment to the disk"""
        assert not self.read_only
        if 'model_args' not in self.config:
            self.config['model_args'] = {}
        args = self.config['model_args']
        args['src_vocab'] = args['tgt_vocab'] = len(self.shared_field)
        self.config['updated_at'] = datetime.now().isoformat()
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

    @property
    def model_args(self) -> Optional[Dict]:
        """
        Gets args from file
        :return: args if exists or None otherwise
        """
        return self.config.get('model_args')

    @model_args.setter
    def model_args(self, model_args):
        """
        set model args
        """
        self.config['model_args'] = model_args

    @property
    def src_vocab(self):
        return self.shared_field

    @property
    def tgt_vocab(self):
        return self.shared_field
