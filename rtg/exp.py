import copy
import os
import random
from datetime import datetime, timedelta
from pathlib import Path
from functools import partial
from typing import Optional, Dict, List, Tuple, Union, Any
import time

import numpy as np
import torch
import hashlib
import portalocker

from rtg import log, yaml, device
from rtg.data.dataset import (TSVData, BatchIterable, LoopingIterable, SqliteFile, GenerativeBatchIterable)
from rtg.data.codec import Field, SPField, NLField, PretrainMatchField
from rtg.utils import IO, line_count


seeded = False


def load_conf(inp: Union[str, Path]):
    with IO.reader(inp) as fh:
        return yaml.load(fh)


class BaseExperiment:

    def __init__(self, work_dir: Union[str, Path], read_only=False,
                 config: Union[str, Path, Optional[Dict[str, Any]]] = None):
        if type(work_dir) is str:
            work_dir = Path(work_dir)

        log.info(f"Initializing an experiment. Directory = {work_dir}")
        self.read_only = read_only
        self.work_dir = work_dir
        self.log_dir = work_dir / 'logs'
        self.log_file = self.log_dir / 'rtg.log'
        self.data_dir = work_dir / 'data'
        self.model_dir = work_dir / 'models'
        self._config_file = work_dir / 'conf.yml'
        if isinstance(config, str) or isinstance(config, Path):
            config = load_conf(config)
        self.config = config if config else load_conf(self._config_file)
        self.codec_name = self.config.get('prep', {}).get('codec_lib', 'sentpiece')  # with default
        codec_libs = {'sentpiece': SPField,
                      'nlcodec': NLField,
                      'pretrainmatch': PretrainMatchField}
        self.codec_supports_multiproc = self.codec_name in {'nlcodec'}
        assert self.codec_name in codec_libs, f'{self.codec_name} is not in {codec_libs.keys()}'
        log.info(f"codec lib = {self.codec_name}")
        self.Field = codec_libs[self.codec_name]

        self._shared_field_file = self.data_dir / f'{self.codec_name}.shared.model'
        self._prepared_flag = self.work_dir / '_PREPARED'
        self._trained_flag = self.work_dir / '_TRAINED'

        self.train_file = self.data_dir / 'train.tsv.gz'
        self.train_db = self.data_dir / 'train.db'
        self.train_db_tmp = self.data_dir / 'train.db.tmp'
        self.finetune_file = self.data_dir / 'finetune.db'
        self.valid_file = self.data_dir / 'valid.tsv.gz'
        self.combo_file = self.data_dir / 'combo.tsv.gz'
        # a set of samples to watch the progress qualitatively
        self.samples_file = self.data_dir / 'samples.tsv.gz'

        if not read_only:
            for _dir in [self.model_dir, self.data_dir, self.log_dir]:
                if not _dir.exists():
                    _dir.mkdir(parents=True)

        assert self.config, 'Looks like the config is emtpy or invalid'
        self.maybe_seed()

        self.shared_field = self.Field(str(self._shared_field_file)) \
            if self._shared_field_file.exists() else None

    @property
    def problem_type(self):
        raise NotImplementedError

    def maybe_seed(self):
        global seeded
        if not seeded and 'seed' in self.config:
            seed = self.config['seed']
            log.info(f"Manual seeding the RNG with {seed}")
            random.seed(seed)
            torch.manual_seed(seed)
            np.random.seed(seed)
            if torch.cuda.is_available():
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
            seeded = True
        else:
            log.info("No manual seed! Letting the RNGs do their stuff")

    def store_config(self):
        yaml.dump(self.config, stream=self._config_file)

    @property
    def model_type(self) -> Optional[str]:
        return self.config.get('model_type')

    @model_type.setter
    def model_type(self, mod_type: str):
        self.config['model_type'] = mod_type

    def has_prepared(self):
        return self._prepared_flag.exists()

    def has_trained(self):
        return self._trained_flag.exists()

    def store_model(self, optimizer_step: int, model, train_score: float, val_score: float, keep: int,
                    prefix='model', keeper_sort='step'):
        """
        saves model to a given path
        :param optimizer_step: optimizer step of the model
        :param model: model object itself
        :param train_score: score of model on training split
        :param val_score: score of model on validation split
        :param keep: number of good models to keep, bad models will be deleted
        :param prefix: prefix to store model. default is "model"
        :param keeper_sort: criteria for choosing the old or bad models for deletion.
            Choices: {'total_score', 'step'}
        :return:
        """
        # TODO: improve this by skipping the model save if the model is not good enough to be saved
        if self.read_only:
            log.warning("Ignoring the store request; experiment is readonly")
            return
        name = f'{prefix}_{optimizer_step:03d}_{train_score:.6f}_{val_score:.6f}.pkl'
        path = self.model_dir / name
        log.info(f"Saving optimizer step {optimizer_step} to {path}")
        torch.save(model, str(path))

        del_models = []
        if keeper_sort == 'total_score':
            del_models = self.list_models(sort='total_score', desc=False)[keep:]
        elif keeper_sort == 'step':
            del_models = self.list_models(sort='step', desc=True)[keep:]
        else:
            Exception(f'Sort criteria{keeper_sort} not understood')
        for d_model in del_models:
            log.info(f"Deleting model {d_model} . Keep={keep}, sort={keeper_sort}")
            os.remove(str(d_model))

        with IO.writer(os.path.join(self.model_dir, 'scores.tsv'), append=True) as f:
            cols = [str(optimizer_step), datetime.now().isoformat(), name, f'{train_score:g}',
                    f'{val_score:g}']
            f.write('\t'.join(cols) + '\n')

    @staticmethod
    def _path_to_validn_score(path):
        parts = str(path.name).replace('.pkl', '').split('_')
        valid_score = float(parts[-1])
        return valid_score

    @staticmethod
    def _path_to_total_score(path):
        parts = str(path.name).replace('.pkl', '').split('_')
        tot_score = float(parts[-2]) + float(parts[-1])
        return tot_score

    @staticmethod
    def _path_to_step_no(path):
        parts = str(path.name).replace('.pkl', '').split('_')
        step_no = int(parts[-3])
        return step_no

    def list_models(self, sort: str = 'step', desc: bool = True) -> List[Path]:
        """
        Lists models in descending order of modification time
        :param sort: how to sort models ?
          - valid_score: sort based on score on validation set
          - total_score: sort based on validation_score + training_score
          - mtime: sort by modification time
          - step (default): sort by step number
        :param desc: True to sort in reverse (default); False to sort in ascending
        :return: list of model paths
        """
        paths = list(self.model_dir.glob('model_*.pkl'))
        if not paths:
            paths = list(self.model_dir.glob('embeddings_*.gz'))
        sorters = {
            'valid_score': self._path_to_validn_score,
            'total_score': self._path_to_total_score,
            'mtime': lambda p: p.stat().st_mtime,
            'step': self._path_to_step_no
        }
        if sort not in sorters:
            raise Exception(f'Sort {sort} not supported. valid options: {sorters.keys()}')
        return sorted(paths, key=sorters[sort], reverse=desc)

    def _get_first_model(self, sort: str, desc: bool) -> Tuple[Optional[Path], int]:
        """
        Gets the first model that matches the given sort criteria
        :param sort: sort mechanism
        :param desc: True for descending, False for ascending
        :return: Tuple[Optional[Path], step_num:int]
        """
        models = self.list_models(sort=sort, desc=desc)
        if models:
            name = models[0].name.replace('.pkl', '').replace('.txt.gz', '')
            step, train_score, valid_score = name.split('_')[-3:]
            return models[0], int(step)
        else:
            return None, 0

    def get_best_known_model(self) -> Tuple[Optional[Path], int]:
        """Gets best Known model (best on lowest scores on training and validation sets)
        """
        return self._get_first_model(sort='total_score', desc=False)

    def get_last_saved_model(self) -> Tuple[Optional[Path], int]:
        return self._get_first_model(sort='step', desc=True)

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
    def optim_args(self) -> Tuple[Optional[str], Dict]:
        """
        Gets optimizer args from file
        :return: optimizer args if exists or None otherwise
        """
        opt_conf = self.config.get('optim')
        if opt_conf:
            return opt_conf.get('name'), opt_conf.get('args')
        else:
            return None, {}

    @optim_args.setter
    def optim_args(self, optim_args: Tuple[str, Dict]):
        """
        set optimizer args
        """
        name, args = optim_args
        self.config['optim'].update({'name': name, 'args': args})

    @property
    def shared_vocab(self) -> Field:
        return self.shared_field

    @staticmethod
    def get_first_found_file(paths: List[Path]):
        """returns the first file that is not None, and actually exists on disc;
        If no file is valid, it returns None"""
        for p in paths:
            if p and p.exists():
                return p
        return None

    def pre_process(self, args=None, force=False):
        if self.has_prepared() and not force:
            log.warning("Already prepared")
            return
        args = args if args else self.config['prep']
        if 'parent' in self.config:
            self.inherit_parent()

        if 'same_data' in args:
            data = Path(args['same_data']) / 'data'
            assert data.exists()
            log.info(f"Reusing prepared data dir from {data}")
            if self.data_dir.exists():
                if self.data_dir.is_symlink():
                    self.data_dir.unlink()
                else:
                    self.data_dir.rename('data.bak')
            self.data_dir.symlink_to(data.resolve(), target_is_directory=True)
            self.reload()
            self._prepared_flag.touch()

    def inherit_parent(self):
        raise NotImplemented()

    def train(self, args=None):
        raise NotImplementedError()

    def reload(self):
        exp = type(self)(self.work_dir, read_only=self.read_only)
        self.__dict__ = exp.__dict__

    @classmethod
    def _checkpt_to_model_state(cls, checkpt_path: Union[str, Path]):
        state = torch.load(checkpt_path, map_location=device)
        if 'model_state' in state:
            state = state['model_state']
        return state

    @classmethod
    def average_states(cls, model_paths: List[Path]):
        assert model_paths, 'at least one model checkpoint should be given. Check your directory'
        for i, mp in enumerate(model_paths):
            next_state = cls._checkpt_to_model_state(mp)
            if i < 1:
                state_dict = next_state
                key_set = set(state_dict.keys())
            else:
                # noinspection PyUnboundLocalVariable
                assert key_set == set(next_state.keys())
                for key in key_set:     # Running average
                    state_dict[key] = (i*state_dict[key] + next_state[key]) / (i + 1)
        return state_dict

    def maybe_ensemble_state(self, model_paths: Optional[List[str]], ensemble: int = 1):
        if model_paths and len(model_paths) == 1:
            log.info(f" Restoring state from requested model {model_paths[0]}")
            return self._checkpt_to_model_state(model_paths[0])
        elif not model_paths and ensemble <= 1:
            model_path, _ = self.get_best_known_model()
            log.info(f" Restoring state from best known model: {model_path}")
            return self._checkpt_to_model_state(model_path)
        else:
            if not model_paths:
                # Average last n models
                model_paths = self.list_models(sort='step', desc=True)[:ensemble]
            digest = hashlib.md5(";".join(str(p) for p in model_paths).encode('utf-8')).hexdigest()
            cache_file = self.model_dir / f'avg_state{len(model_paths)}_{digest}.pkl'
            lock_file = cache_file.with_suffix('.lock')
            MAX_TIMEOUT = 1 * 60 * 60  # 1 hour
            with portalocker.Lock(lock_file, 'w', timeout=MAX_TIMEOUT) as fh:
                # check if downloaded by  other parallel process
                if lock_file.exists() and cache_file.exists():
                    log.info(f"Cache exists: reading from {cache_file}")
                    state = self._checkpt_to_model_state(cache_file)
                else:
                    log.info(f"Averaging {len(model_paths)} model states :: {model_paths}")
                    state = self.average_states(model_paths)
                    if len(model_paths) > 1:
                        log.info(f"Caching the averaged state at {cache_file}")
                        torch.save(state, str(cache_file))
            return state
        
    def load_model(self, model_paths=None, ensemble=1):
        from rtg.registry import factories
        factory = factories[self.model_type]
        model = factory(exp=self, **self.model_args)[0]
        state = self.maybe_ensemble_state(model_paths=model_paths, ensemble=ensemble)
        errors = model.load_state_dict(state)
        log.info(f"{errors}")
        return model

    def load_model_with_state(self, checkpt_state):
        from rtg.registry import factories
        chkpt = checkpt_state
        state = chkpt['model_state']
        model_type = chkpt['model_type']
        model_args = chkpt['model_args']
        # Dummy experiment wrapper
        factory = factories[model_type]
        model = factory(exp=self, **model_args)[0]
        errors = model.load_state_dict(state)
        log.info(f"{errors}")
        log.info(f"Successfully restored the model state of : {model_type}")
        return model


class TranslationExperiment(BaseExperiment):

    def __init__(self, work_dir: Union[str, Path], read_only=False,
                 config: Union[str, Path, Optional[Dict[str, Any]]] = None):
        super().__init__(work_dir, read_only=read_only, config=config)
        self._src_field_file = self.data_dir / f'{self.codec_name}.src.model'
        self._tgt_field_file = self.data_dir / f'{self.codec_name}.tgt.model'

        self.emb_src_file = self.data_dir / 'emb_src.pt'
        self.emb_tgt_file = self.data_dir / 'emb_tgt.pt'
        self.ext_emb_src_file = self.data_dir / 'ext_emb_src.pt'  # external Embeddings
        self.ext_emb_tgt_file = self.data_dir / 'ext_emb_tgt.pt'  # external Embeddings

        self.reload_vocabs()

        # Either shared field  OR  individual  src and tgt fields
        assert not (self.shared_field and self.src_field)
        assert not (self.shared_field and self.tgt_field)
        # both are set or both are unset
        #assert (self.src_field is None) == (self.tgt_field is None)

        self._unsupervised = self.model_type in {'binmt', 'rnnlm', 'tfmlm'}
        if self._unsupervised:
            self.mono_train_src = self.data_dir / 'mono.train.src.gz'
            self.mono_train_tgt = self.data_dir / 'mono.train.tgt.gz'
            self.mono_valid_src = self.data_dir / 'mono.valid.src.gz'
            self.mono_valid_tgt = self.data_dir / 'mono.valid.tgt.gz'

        self.parent_model_state = self.data_dir / 'parent_model_state.pt'

    @property
    def problem_type(self):
        from rtg.registry import ProblemType
        return ProblemType.TRANSLATION

    def reload_vocabs(self):
        self.src_field, self.tgt_field, self.shared_field = [
            self.Field(str(f)) if f.exists() else None for f in (
                self._src_field_file, self._tgt_field_file, self._shared_field_file)]

    def check_line_count(self, name, file1, file2):
        count1 = line_count(file1)
        count2 = line_count(file2)
        if count1 == count2:
            log.info(f"Found {count1:,} parallel lines for {name}")
        else:
            log.error(f"Found line mismatch in {name} ")
            raise Exception(f'{file1} has {count1:,} lines but {file2} has {count2:,} lines')

    def pre_process_parallel(self, args: Dict[str, Any]):
        # check if files are parallel
        self.check_line_count('validation', args['valid_src'], args['valid_tgt'])
        if 'spark' in self.config:
            log.warning(f"Spark backend detected: line count on training data is skipped")
        else:
            log.warning(f"Going to count lines. If this is a big dataset, it will take long time")
            self.check_line_count('training', args['train_src'], args['train_tgt'])

        xt_args = dict(no_split_toks=args.get('no_split_toks'),
                       char_coverage=args.get('char_coverage', 0))
        if args.get('shared_vocab'):  # shared vocab
            corpus = [args[key] for key in ['train_src', 'train_tgt', 'mono_src', 'mono_tgt']
                      if args.get(key)]
            self.shared_field = self._make_vocab("shared", self._shared_field_file, args['pieces'],
                                                 args['max_types'], corpus=corpus, **xt_args)
        else:  # separate vocabularies
            src_corpus = [args[key] for key in ['train_src', 'mono_src'] if args.get(key)]
            self.src_field = self._make_vocab("src", self._src_field_file, args['pieces'],
                                              args['max_src_types'], corpus=src_corpus, **xt_args)

            # target vocabulary
            tgt_corpus = [args[key] for key in ['train_tgt', 'mono_tgt'] if args.get(key)]
            self.tgt_field = self._make_vocab("src", self._tgt_field_file, args['pieces'],
                                              args['max_tgt_types'], corpus=tgt_corpus, **xt_args)

        train_file = self.train_db

        self._pre_process_parallel('train_src', 'train_tgt', out_file=train_file, args=args,
                                   line_check=False)
        self._pre_process_parallel('valid_src', 'valid_tgt', out_file=self.valid_file, args=args,
                                   line_check=False)

        if args.get("finetune_src") or args.get("finetune_tgt"):
            self._pre_process_parallel('finetune_src', 'finetune_tgt', self.finetune_file)

        # get samples from validation set
        n_samples = args.get('num_samples', 5)
        space_tokr = lambda line: line.strip().split()
        val_raw_recs = TSVData.read_raw_parallel_recs(
            args['valid_src'], args['valid_tgt'], args['truncate'], args['src_len'],
            args['tgt_len'], src_tokenizer=space_tokr, tgt_tokenizer=space_tokr)
        val_raw_recs = list(val_raw_recs)
        random.shuffle(val_raw_recs)
        samples = val_raw_recs[:n_samples]
        TSVData.write_parallel_recs(samples, self.samples_file)

    def _make_vocab(self, name: str, vocab_file: Path, model_type: str, vocab_size: int,
                    corpus: List, no_split_toks: List[str] = None, char_coverage=0,
                    min_co_ev=None) -> Field:
        """
        Construct vocabulary file
        :param name: name : src, tgt or shared -- for the sake of logging
        :param vocab_file: where to save the vocab file
        :param model_type: sentence piece model type
        :param vocab_size: max types in vocab
        :param corpus: as the name says, list of files from which the vocab should be learned
        :param no_split_toks: tokens that needs to be preserved from splitting, or added
        :return:
        """
        if vocab_file.exists():
            log.info(f"{vocab_file} exists. Skipping the {name} vocab creation")
            return self.Field(str(vocab_file))
        flat_uniq_corpus = set()  # remove dupes, flat the nested list or sets
        for i in corpus:
            if isinstance(i, set) or isinstance(i, list):
                flat_uniq_corpus.update(i)
            else:
                flat_uniq_corpus.add(i)

        flat_uniq_corpus = list(flat_uniq_corpus)
        log.info(f"Going to build {name} vocab from files")
        xt_args = {}
        if min_co_ev:
            xt_args["min_co_ev"] = min_co_ev
        return self.Field.train(model_type, vocab_size, str(vocab_file), flat_uniq_corpus,
                                no_split_toks=no_split_toks, char_coverage=char_coverage, **xt_args)

    def pre_process_mono(self, args):
        xt_args = dict(no_split_toks=args.get('no_split_toks'),
                       char_coverage=args.get('char_coverage', 0))

        mono_files = [args[key] for key in ['mono_train_src', 'mono_train_tgt'] if key in args]
        assert mono_files, "At least one of 'mono_train_src', 'mono_train_tgt' should be set"
        log.info(f"Found mono files: {mono_files}")
        if args.get('shared_vocab'):
            self.shared_field = self._make_vocab("shared", self._shared_field_file, args['pieces'],
                                                 args['max_types'], corpus=mono_files, **xt_args)
        else:  # separate vocabularies
            if 'mono_train_src' in args:
                self.src_field = self._make_vocab("src", self._src_field_file,
                                                  args['pieces'], args['max_src_types'],
                                                  corpus=[args['mono_train_src']], **xt_args)
            else:
                log.warning("Skipping source vocab creation since mono_train_src is not given")

            # target vocabulary
            if 'mono_train_tgt' in args:
                self.tgt_field = self._make_vocab("src", self._tgt_field_file,
                                                  args['pieces'], args['max_tgt_types'],
                                                  corpus=[args['mono_train_tgt']], **xt_args)
            else:
                log.warning("Skipping target vocab creation since mono_train_tgt is not given")

        def _prep_file(file_key, out_file, do_truncate, max_len, field: Field):
            if file_key not in args:
                log.warning(f'Skipped: {file_key} because it is not found in config')
                return

            raw_file = args[file_key]

            recs = TSVData.read_raw_mono_recs(raw_file, do_truncate, max_len, field.encode_as_ids)
            # TODO: use SQLite storage
            TSVData.write_mono_recs(recs, out_file)
            if args.get('text_files'):
                recs = TSVData.read_raw_mono_recs(raw_file, do_truncate, max_len, field.tokenize)
                TSVData.write_mono_recs(recs, str(out_file).replace('.tsv', '.pieces.tsv'))

        _prep_file('mono_train_src', self.mono_train_src, args['truncate'], args['src_len'],
                   self.src_vocab)
        _prep_file('mono_train_tgt', self.mono_train_tgt, args['truncate'], args['tgt_len'],
                   self.tgt_vocab)

        _prep_file('mono_valid_src', self.mono_valid_src, args['truncate'], args['src_len'],
                   self.src_vocab)
        _prep_file('mono_valid_tgt', self.mono_valid_tgt, args['truncate'], args['tgt_len'],
                   self.tgt_vocab)

    def _pre_process_parallel(self, src_key: str, tgt_key: str, out_file: Path,
                              args: Optional[Dict[str, Any]] = None, line_check=True,
                              split_ratio: float = 0.):
        """
        Pre process records of a parallel corpus
        :param args: all arguments for 'prep' task
        :param src_key: key that contains source sequences
        :param tgt_key: key that contains target sequences
        :param out_file: path to store processed TSV data (compresses if name ends with .gz)
        :return:
        """
        args = args if args else self.config['prep']
        log.info(f"Going to prep files {src_key} and {tgt_key}")
        assert src_key in args, f'{src_key} not found in experiment config or args'
        assert tgt_key in args, f'{tgt_key} not found in experiment config or args'
        if line_check:
            assert line_count(args[src_key]) == line_count(args[tgt_key]), \
                f'{args[src_key]} and {args[tgt_key]} must have same number of lines'
        # create Piece IDs
        s_time = time.time()
        reader_func = TSVData.read_raw_parallel_recs
        parallel_recs = reader_func(
            args[src_key], args[tgt_key], args['truncate'], args['src_len'], args['tgt_len'],
            src_tokenizer=partial(self.src_vocab.encode_as_ids, split_ratio=split_ratio),
            tgt_tokenizer=partial(self.tgt_vocab.encode_as_ids, split_ratio=split_ratio))
        if any([out_file.name.endswith(suf) for suf in ('.nldb', '.nldb.tmp')]):
            from nlcodec.db import MultipartDb
            MultipartDb.create(path=out_file, recs=parallel_recs, field_names=('x', 'y'))
        elif any([out_file.name.endswith(suf) for suf in ('.db', '.db.tmp')]):
            SqliteFile.write(out_file, records=parallel_recs)
        else:
            TSVData.write_parallel_recs(parallel_recs, out_file)
        e_time = time.time()
        log.info(f"Time taken to process: {timedelta(seconds=(e_time - s_time))}")
        if args.get('text_files'):
            # Redo again as plain text files
            parallel_recs = reader_func(
                args[src_key], args[tgt_key], args['truncate'], args['src_len'], args['tgt_len'],
                src_tokenizer=self.src_vocab.tokenize, tgt_tokenizer=self.tgt_vocab.tokenize)

            text_file_name = str(out_file).replace('.db', '.tsv.gz').replace('.tsv', '.pieces.tsv')
            TSVData.write_parallel_recs(parallel_recs, text_file_name)

    def maybe_pre_process_embeds(self, do_clean=False):

        def _read_vocab(path: Path) -> List[str]:
            with IO.reader(path) as rdr:
                vocab = [line.strip().split()[0] for line in rdr]
                if do_clean:
                    # sentence piece starts with '▁' character
                    vocab = [word[1:] if word[0] == '▁' else word for word in vocab]
                return vocab

        def _map_and_store(inp: Path, vocab_file: Path):
            id_to_str = _read_vocab(vocab_file)
            str_to_id = {tok: idx for idx, tok in enumerate(id_to_str)}
            assert len(id_to_str) == len(id_to_str)
            vocab_size = len(id_to_str)

            matched_set, ignored_set, duplicate_set = set(), set(), set()

            with inp.open(encoding='utf-8') as in_fh:
                header = in_fh.readline()
                parts = header.strip().split()
                if len(parts) == 2:
                    tot, dim = int(parts[0]), int(parts[1])
                    matrix = torch.zeros(vocab_size, dim)
                else:
                    assert len(parts) > 2
                    word, vec = parts[0], [float(x) for x in parts[1:]]
                    dim = len(vec)
                    matrix = torch.zeros(vocab_size, dim)
                    if word in str_to_id:
                        matrix[str_to_id[word]] = torch.tensor(vec, dtype=torch.float)
                        matched_set.add(word)
                    else:
                        ignored_set.add(word)

                for line in in_fh:
                    parts = line.strip().split()
                    word = parts[0]
                    if word in str_to_id:
                        if word in matched_set:
                            duplicate_set.add(word)
                        # Note: this overwrites duplicate words
                        vec = [float(x) for x in parts[1:]]
                        matrix[str_to_id[word]] = torch.tensor(vec, dtype=torch.float)
                        matched_set.add(word)
                    else:
                        ignored_set.add(word)
            pre_trained = matched_set | ignored_set
            vocab_set = set(id_to_str)
            oovs = vocab_set - matched_set
            stats = {
                'pre_trained': len(pre_trained),
                'vocab': len(vocab_set),
                'matched': len(matched_set),
                'ignored': len(ignored_set),
                'oov': len(oovs)
            }
            stats.update({
                'oov_rate': stats['oov'] / stats['vocab'],
                'match_rate': stats['matched'] / stats['vocab'],
                'useless_rate': stats['ignored'] / stats['pre_trained'],
                'useful_rate': stats['matched'] / stats['pre_trained']
            })
            return matrix, stats

        def _write_emb_matrix(matrix, path: str):
            torch.save(matrix, path)

        def _write_dict(dict, path: Path):
            with IO.writer(path) as out:
                for key, val in dict.items():
                    out.write(f"{key}\t{val}\n")

        args = self.config['prep']
        mapping = {
            'pre_emb_src': self.emb_src_file,
            'pre_emb_tgt': self.emb_tgt_file,
            'ext_emb_src': self.ext_emb_src_file,
            'ext_emb_tgt': self.ext_emb_tgt_file,
        }
        if not any(x in args for x in mapping):
            log.info("No pre trained embeddings are found in config; skipping it")
            return

        for key, outp in mapping.items():
            if key in args:
                inp = Path(args[key])
                assert inp.exists()
                voc_file = self.data_dir / f'sentpiece.shared.vocab'
                if not voc_file.exists():
                    field_name = key.split('_')[-1]  # emb_src --> src ; emb_tgt --> tgt
                    voc_file = self.data_dir / f'sentpiece.{field_name}.vocab'
                    assert voc_file.exists()

                log.info(f"Processing {key}: {inp}")
                emb_matrix, report = _map_and_store(inp, voc_file)
                _write_dict(report, Path(str(outp) + '.report.txt'))
                _write_emb_matrix(emb_matrix, str(outp))

    def shrink_vocabs(self):
        assert self.codec_name == 'nlcodec', 'Only nlcodec supports shrinking of vocabs'
        args = self.config['prep']

        if self.shared_vocab:
            corpus = [args[key] for key in ['train_src', 'train_tgt', 'mono_src', 'mono_tgt']
                      if args.get(key)]
            remap_src = self.shared_vocab.shrink_vocab(files=corpus, min_freq=1,
                                                    save_at=self._shared_field_file)
            remap_tgt = remap_src
        else:
            corpus_src = [args[key] for key in ['train_src', 'mono_src'] if args.get(key)]
            remap_src = self.src_vocab.shrink_vocab(files=corpus_src, min_freq=1,
                                                     save_at=self._src_field_file)
            corpus_tgt = [args[key] for key in ['train_tgt', 'mono_tgt'] if args.get(key)]
            remap_tgt = self.tgt_vocab.shrink_vocab(files=corpus_tgt, min_freq=1,
                                                     save_at=self._tgt_field_file)
        self.reload_vocabs()
        self.model_args['src_vocab'] = len(self.src_vocab)
        self.model_args['tgt_vocab'] = len(self.tgt_vocab)
        return remap_src, remap_tgt

    def inherit_parent(self):
        parent = self.config['parent']
        parent_exp = type(self)(parent['experiment'], read_only=True)
        log.info(f"Parent experiment: {parent_exp.work_dir}")
        parent_exp.has_prepared()
        vocab_sepc = parent.get('vocab')
        if vocab_sepc:
            log.info(f"Parent vocabs inheritance spec: {vocab_sepc}")
            codec_lib = parent_exp.config['prep'].get('codec_lib')
            if codec_lib:
                self.config['prep']['codec_lib'] = codec_lib

            def _locate_field_file(exp: TranslationExperiment, name, check_exists=False) -> Path:
                switch = {'src': exp._src_field_file,
                          'tgt': exp._tgt_field_file,
                          'shared': exp._shared_field_file}
                assert name in switch, f'{name} not allowed; valid options= {switch.keys()}'
                file = switch[name]
                if check_exists:
                    assert file.exists(), f'{file} doesnot exist; for {name} of {exp.work_dir}'
                return file

            for to_field, from_field in vocab_sepc.items():
                from_field_file = _locate_field_file(parent_exp, from_field, check_exists=True)
                to_field_file = _locate_field_file(self, to_field, check_exists=False)
                IO.copy_file(from_field_file, to_field_file)
            self.reload_vocabs()
        else:
            log.info("No vocabularies are inherited from parent")
        model_sepc = parent.get('model')
        if model_sepc:
            log.info("Parent model inheritance spec")
            if model_sepc.get('args'):
                self.model_args = parent_exp.model_args
            ensemble = model_sepc.get('ensemble', 1)
            model_paths = parent_exp.list_models(sort='step', desc=True)[:ensemble]
            log.info(f"Averaging {len(model_paths)} checkpoints of parent model: \n{model_paths}")
            avg_state = self.average_states(model_paths=model_paths)
            log.info(f"Saving parent model's state to {self.parent_model_state}")
            torch.save(avg_state, self.parent_model_state)

        shrink_spec = parent.get('shrink')
        if shrink_spec:
            remap_src, remap_tgt = self.shrink_vocabs()
            def map_rows(mapping: List[int], source: torch.Tensor, name=''):
                assert max(mapping) < len(source)
                target = torch.zeros((len(mapping), *source.shape[1:]),
                                     dtype=source.dtype, device=source.device)
                for new_idx, old_idx in enumerate(mapping):
                    target[new_idx] = source[old_idx]
                log.info(f"Mapped {name} {source.shape} --> {target.shape} ")
                return target

            """ src_embed.0.lut.weight [N x d]
                tgt_embed.0.lut.weight [N x d]
                generator.proj.weight [N x d]
                generator.proj.bias [N] """
            if remap_src:
                key = 'src_embed.0.lut.weight'
                avg_state[key] = map_rows(remap_src, avg_state[key], name=key)
            if remap_tgt:
                map_keys = ['tgt_embed.0.lut.weight', 'generator.proj.weight', 'generator.proj.bias']
                for key in map_keys:
                    if key not in avg_state:
                        log.warning(f'{key} not found in avg_state of parent model. Mapping skipped')
                        continue
                    avg_state[key] = map_rows(remap_tgt, avg_state[key], name=key)
            if self.parent_model_state.exists():
                self.parent_model_state.rename(self.parent_model_state.with_suffix('.orig'))
            torch.save(avg_state, self.parent_model_state)
            self.persist_state()  # this will fix src_vocab and tgt_vocab of model_args conf


    def pre_process(self, args=None, force=False):
        args = args or self.config['prep']
        super(TranslationExperiment, self).pre_process(args, )
        if self.has_prepared() and not force:
            log.warning("Already prepared")
            return

        if self._unsupervised:
            self.pre_process_mono(args)
        else:
            self.pre_process_parallel(args)

        self.maybe_pre_process_embeds()
        # update state on disk
        self.persist_state()
        self._prepared_flag.touch()

    def persist_state(self):
        """Writes state of current experiment to the disk"""
        assert not self.read_only
        if 'model_args' not in self.config:
            self.config['model_args'] = {}
        args = self.config['model_args']
        if self.model_type in {'rnnlm', 'tfmlm', 'wv_cbow'}:
            # Language models
            # TODO: improve the design of this thing
            args['vocab_size'] = max(len(self.src_vocab) if self.src_vocab else 0,
                                     len(self.tgt_vocab) if self.tgt_vocab else 0)
        else:
            # Translation models
            args['src_vocab'] = len(self.src_vocab) if self.src_vocab else 0
            args['tgt_vocab'] = len(self.tgt_vocab) if self.tgt_vocab else 0

        self.config['updated_at'] = datetime.now().isoformat()
        self.store_config()

    def train(self, args=None):
        run_args = copy.deepcopy(self.config.get('trainer', {}))
        if args:
            run_args.update(args)
        if 'init_args' in run_args:
            del run_args['init_args']
        train_steps = run_args['steps']
        finetune_steps = run_args.pop('finetune_steps', None)
        finetune_batch_size = run_args.pop('finetune_batch_size', run_args.get('batch_size'))
        if finetune_steps:
            assert type(finetune_steps) is int
            assert finetune_steps > train_steps, f'finetune_steps={finetune_steps} should be' \
                                                 f' greater than steps={train_steps}'

        _, last_step = self.get_last_saved_model()
        if self._trained_flag.exists():
            # noinspection PyBroadException
            try:
                last_step = max(last_step, yaml.load(self._trained_flag.read_text())['steps'])
            except Exception as _:
                pass

        if last_step >= train_steps and (finetune_steps is None or last_step >= finetune_steps):
            log.warning(
                f"Already trained upto {last_step}; Requested: train={train_steps}, finetune={finetune_steps} Skipped")
            return

        from rtg.registry import trainers, factories
        name, optim_args = self.optim_args
        trainer = trainers[self.model_type](self, optim=name,
                                            model_factory=factories[self.model_type], **optim_args)
        if last_step < train_steps:  # regular training
            stopped = trainer.train(fine_tune=False, **run_args)
            if not self.read_only:
                status = dict(steps=train_steps, early_stopped=stopped, finetune=False)
                try:
                    status['earlier'] = yaml.load(self._trained_flag.read_text())
                except Exception as _:
                    pass
                yaml.dump(status, stream=self._trained_flag)
        if finetune_steps:  # Fine tuning
            log.info(f"Fine tuning upto {finetune_steps}, batch_size={finetune_batch_size}")
            assert finetune_batch_size
            run_args['steps'] = finetune_steps
            run_args['batch_size'] = finetune_batch_size

            stopped = trainer.train(fine_tune=True, **run_args)
            status = dict(steps=finetune_steps, early_stopped=stopped, finetune=True)
            try:
                status['earlier'] = yaml.load(self._trained_flag.read_text())
            except Exception as _:
                pass
            yaml.dump(status, stream=self._trained_flag)

    @property
    def src_vocab(self) -> Field:
        return self.shared_field if self.shared_field is not None else self.src_field

    @property
    def tgt_vocab(self) -> Field:
        return self.shared_field if self.shared_field is not None else self.tgt_field

    def _get_batch_args(self):
        prep_args = self.config.get('prep', {})
        return {ok: prep_args[ik] for ik, ok in
                [('src_len', 'max_src_len'), ('tgt_len', 'max_tgt_len'), ('truncate', 'truncate')]
                if ik in prep_args}

    def get_train_data(self, batch_size:  Union[int, Tuple[int,int]], steps: int = 0, sort_by='eq_len_rand_batch',
                       batch_first=True, shuffle=False, fine_tune=False, keep_in_mem=False,
                       split_ratio: float = 0., dynamic_epoch=False, y_is_cls=False):

        data_path = self.train_db if self.train_db.exists() else self.train_file
        if fine_tune:
            if not self.finetune_file.exists():
                # user may have added fine tune file later
                self._pre_process_parallel('finetune_src', 'finetune_tgt', self.finetune_file)
            log.info("Using Fine tuning corpus instead of training corpus")
            data_path = self.finetune_file

        if split_ratio > 0:
            data_path = IO.maybe_tmpfs(data_path)
            train_file = data_path.with_suffix('.db.tmp')
            assert not y_is_cls, 'Not supported feature'
            file_creator = partial(self.file_creator, train_file=train_file, split_ratio=split_ratio)
            train_data = GenerativeBatchIterable(
                file_creator=file_creator, batches=steps, batch_size=batch_size, field=self.tgt_vocab,
                dynamic_epoch=dynamic_epoch, batch_first=batch_first, shuffle=shuffle, sort_by=sort_by,
                **self._get_batch_args())
        else:
            data = BatchIterable(
                data_path=data_path, batch_size=batch_size, field=self.tgt_vocab, sort_by=sort_by,
                batch_first=batch_first, shuffle=shuffle, y_is_cls=y_is_cls, **self._get_batch_args())
            train_data = LoopingIterable(data, steps)

        return train_data


    def file_creator(self, train_file, split_ratio, *args, **kwargs):
        self._pre_process_parallel(*args, src_key='train_src', tgt_key='train_tgt',
                                   out_file=train_file, split_ratio=split_ratio, **kwargs)
        return train_file

    def get_val_data(self, batch_size: Union[int, Tuple[int,int]], sort_desc=False, batch_first=True,
                     shuffle=False, y_is_cls=False):
        raw_path = None
        prep = self.config.get('prep', {})
        if 'valid_src' in prep and 'valid_tgt' in prep:
            raw_path = prep['valid_src'], prep['valid_tgt']

        return BatchIterable(self.valid_file, batch_size=batch_size, sort_desc=sort_desc,
                             batch_first=batch_first, shuffle=shuffle, field=self.tgt_vocab,
                             keep_in_mem=True, raw_path=raw_path, y_is_cls=y_is_cls,
                             **self._get_batch_args())

    def get_combo_data(self, batch_size: int, steps: int = 0, sort_desc=False, batch_first=True,
                       shuffle=False):
        if not self.combo_file.exists():
            # user may have added fine tune file later
            self._pre_process_parallel('combo_src', 'combo_tgt', self.combo_file)
        combo_file = IO.maybe_tmpfs(self.combo_file)
        data = BatchIterable(
            combo_file, batch_size=batch_size, sort_desc=sort_desc, field=self.tgt_vocab,
            batch_first=batch_first, shuffle=shuffle, **self._get_batch_args()
        )
        if steps > 0:
            data = LoopingIterable(data, steps)
        return data


    def copy_vocabs(self, other):
        """
        Copies vocabulary files from self to other
        :param other: other experiment
        :return:
        """
        other: TranslationExperiment = other
        if not other.data_dir.exists():
            other.data_dir.mkdir(parents=True)
        for source, destination in [(self._src_field_file, other._src_field_file),
                                    (self._tgt_field_file, other._tgt_field_file),
                                    (self._shared_field_file, other._shared_field_file)]:
            if source.exists():
                IO.copy_file(source.resolve(), destination.resolve())
                src_txt_file = source.with_name(source.name.replace('.model', '.vocab'))
                if src_txt_file.exists():
                    dst_txt_file = destination.with_name(
                        destination.name.replace('.model', '.vocab'))
                    IO.copy_file(src_txt_file, dst_txt_file)

    def get_mono_data(self, split: str, side: str, batch_size: int, sort_desc: bool = False,
                      batch_first: bool = False, shuffle: bool = False, num_batches: int = 0):
        """
        reads monolingual data
        :param split: name of the split. choices = {train, valid}
        :param side: which side ? choices = {src, tgt}
        :param batch_size: what should be batch size. example =64
        :param sort_desc: should the seqs in batch be sorted descending order of length ?
        :param batch_first: should the first dimension be batch instead of time step ?
        :param shuffle: should the seqs be shuffled before reading (and for each re-reading
            if num_batches is too large)
        :param num_batches: how many batches to read?
        :return: iterator of batches
        """
        assert side in ('src', 'tgt')
        assert split in ('train', 'valid')
        inp_file = {
            ('train', 'src'): self.mono_train_src,
            ('train', 'tgt'): self.mono_train_tgt,
            ('valid', 'src'): self.mono_valid_src,
            ('valid', 'tgt'): self.mono_valid_tgt,
        }[(split, side)]
        assert inp_file.exists()
        # read this file
        field = self.tgt_vocab if side == 'tgt' else self.src_field
        data = BatchIterable(inp_file, batch_size=batch_size, sort_desc=sort_desc,
                             batch_first=batch_first, shuffle=shuffle, field=field,
                             **self._get_batch_args())

        if num_batches > 0:
            data = LoopingIterable(data, num_batches)
        return data
