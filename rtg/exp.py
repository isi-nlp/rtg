import yaml
import os
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Iterator, List, Tuple, Union, Any
import torch
import random
from rtg import log
from rtg.dataprep import (RawRecord, ParallelSeqRecord, MonoSeqRecord,
                          Field, BatchIterable, LoopingIterable)
from rtg.utils import IO, line_count
from itertools import zip_longest


def load_conf(inp: Union[str, Path]):
    with IO.reader(inp) as fh:
        return yaml.load(fh)


class TranslationExperiment:

    def __init__(self, work_dir: Union[str, Path], read_only=False,
                 config: Union[str, Path, Optional[Dict[str, Any]]] = None):
        if type(work_dir) is str:
            work_dir = Path(work_dir)

        log.info(f"Initializing an experiment. Directory = {work_dir}")
        self.read_only = read_only
        self.work_dir = work_dir
        self.data_dir = work_dir / 'data'
        self.model_dir = work_dir / 'models'
        self._config_file = work_dir / 'conf.yml'
        self._shared_field_file = self.data_dir / 'sentpiece.shared.model'
        self._src_field_file = self.data_dir / 'sentpiece.src.model'
        self._tgt_field_file = self.data_dir / 'sentpiece.tgt.model'
        self._prepared_flag = self.work_dir / '_PREPARED'
        self._trained_flag = self.work_dir / '_TRAINED'

        self.train_file = self.data_dir / 'train.tsv.gz'
        self.finetune_file = self.data_dir / 'finetune.tsv.gz'
        self.valid_file = self.data_dir / 'valid.tsv.gz'
        self.combo_file = self.data_dir / 'combo.tsv.gz'
        # a set of samples to watch the progress qualitatively
        self.samples_file = self.data_dir / 'samples.tsv.gz'

        self.emb_src_file = self.data_dir / 'emb_src.pt'
        self.emb_tgt_file = self.data_dir / 'emb_tgt.pt'
        self.aln_emb_src_file = self.data_dir / 'aln_emb_src.pt'  # src embs aligned to tgt

        if not read_only:
            for _dir in [self.model_dir, self.data_dir]:
                if not _dir.exists():
                    _dir.mkdir(parents=True)
        if isinstance(config, str) or isinstance(config, Path):
            config = load_conf(config)
        self.config = config if config else load_conf(self._config_file)

        self.shared_field, self.src_field, self.tgt_field = [
            Field(str(f)) if f.exists() else None
            for f in (self._shared_field_file, self._src_field_file, self._tgt_field_file)]

        # Either shared field  OR  individual  src and tgt fields
        assert not (self.shared_field and self.src_field)
        assert not (self.shared_field and self.tgt_field)
        # both are set or both are unset
        assert (self.src_field is None) == (self.tgt_field is None)

        self._unsupervised = self.model_type in {'binmt', 'rnnlm', 'tfmlm'}
        if self._unsupervised:
            self.mono_train_src = self.data_dir / 'mono.train.src.gz'
            self.mono_train_tgt = self.data_dir / 'mono.train.tgt.gz'
            self.mono_valid_src = self.data_dir / 'mono.valid.src.gz'
            self.mono_valid_tgt = self.data_dir / 'mono.valid.tgt.gz'

    def store_config(self):
        with IO.writer(self._config_file) as fp:
            return yaml.dump(self.config, fp, default_flow_style=False)

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

    @staticmethod
    def write_tsv(records: Iterator[ParallelSeqRecord], path: Union[str, Path]):
        seqs = ((' '.join(map(str, x)), ' '.join(map(str, y))) for x, y in records)
        lines = (f'{x}\t{y}\n' for x, y in seqs)
        log.info(f"Storing data at {path}")
        with IO.writer(path) as f:
            for line in lines:
                f.write(line)

    @staticmethod
    def write_mono_lines(records: Iterator[MonoSeqRecord], path: Union[str, Path]):
        lines = (' '.join(map(str, rec)) + '\n' for rec in records)
        log.info(f"Storing data at {path}")
        with IO.writer(path) as f:
            for line in lines:
                f.write(line)

    @staticmethod
    def read_raw_lines(src_path: Union[str, Path], tgt_path: Union[str, Path]) \
            -> Iterator[RawRecord]:
        with IO.reader(src_path) as src_lines, IO.reader(tgt_path) as tgt_lines:
            # if you get an exception here --> files have un equal number of lines
            recs = ((src.strip(), tgt.strip()) for src, tgt in zip_longest(src_lines, tgt_lines))
            recs = ((src, tgt) for src, tgt in recs if src and tgt)
            yield from recs

    def read_raw_data(self, src_path: Union[str, Path], tgt_path: Union[str, Path],
                      truncate: bool, src_len: int, tgt_len: int, tokenizer) \
            -> Iterator[ParallelSeqRecord]:
        recs = self.read_raw_lines(src_path, tgt_path)
        recs = ((tokenizer(x), tokenizer(y)) for x, y in recs)
        if truncate:
            recs = ((src[:src_len], tgt[:tgt_len]) for src, tgt in recs)
        else:  # Filter out longer sentences
            recs = ((src, tgt) for src, tgt in recs if len(src) <= src_len and len(tgt) <= tgt_len)
        return recs

    @staticmethod
    def read_mono_raw_data(path: Union[str, Path], truncate: bool, max_len: int, tokenizer):
        with IO.reader(path) as lines:
            recs = (tokenizer(line.strip()) for line in lines if line.strip())
            if truncate:
                recs = (rec[:max_len] for rec in recs)
            else:  # Filter out longer sentences
                recs = (rec for rec in recs if 0 < len(rec) <= max_len)
            yield from recs

    def pre_process_parallel(self, args: Dict[str, Any]):
        assert args['shared_vocab']  # TODO support individual vocab types

        # check if files are parallel
        assert line_count(args['train_src']) == line_count(args['train_tgt'])
        assert line_count(args['valid_src']) == line_count(args['valid_tgt'])

        if self._shared_field_file.exists():
            log.info(f"{self._shared_field_file} exists. Skipping shared vocab creation")
        else:
            files = [args['mono_train_src'], args['mono_train_tgt']]
            # monolingual files for vocab creation
            files += [args[key] for key in ['mono_src', 'mono_tgt'] if key in args]
            no_split_toks = args.get('no_split_toks')
            self.shared_field = Field.train(args['pieces'], args['max_types'],
                                            str(self._shared_field_file), files,
                                            no_split_toks=no_split_toks)

        self._pre_process_parallel('train_src', 'train_tgt', out_file=self.train_file, args=args,
                                   line_check=False)
        self._pre_process_parallel('valid_src', 'valid_tgt', out_file=self.valid_file, args=args,
                                   line_check=False)

        if args.get("finetune_src") or args.get("finetune_tgt"):
            self._pre_process_parallel('finetune_src', 'finetune_tgt', self.finetune_file)

        # get samples from validation set
        n_samples = args.get('num_samples', 5)
        val_raw_recs = self.read_raw_data(args['valid_src'], args['valid_tgt'], args['truncate'],
                                          args['src_len'], args['tgt_len'],
                                          tokenizer=lambda line: line.strip().split())
        val_raw_recs = list(val_raw_recs)
        random.shuffle(val_raw_recs)
        samples = val_raw_recs[:n_samples]
        self.write_tsv(samples, self.samples_file)

    def pre_process_mono(self, args):

        mono_files = [args[key] for key in ['mono_train_src', 'mono_train_tgt'] if key in args]
        assert mono_files, "At least one of 'mono_train_src', 'mono_train_tgt' should be set"
        log.info(f"Found mono files: {mono_files}")
        no_split_toks = args.get('no_split_toks')
        if args.get('shared_vocab'):
            if self._shared_field_file.exists():
                log.info(f"{self._shared_field_file} exists. Skipping shared vocab creation")
            else:
                log.info("Going to build shared vocab from mono files")
                self.shared_field = Field.train(args['pieces'],
                                                args['max_types'],
                                                str(self._shared_field_file), mono_files,
                                                no_split_toks=no_split_toks)
        else:  # separate vocabularies
            # Source vocabulary
            if self._src_field_file.exists():
                log.info(f"{self._src_field_file} exists. Skipping source vocab creation... ")
            else:
                if 'mono_train_src' in args:
                    log.info("Going to build source vocab from mono_train_src")
                    self.src_field = Field.train(args['pieces'], args['max_src_types'],
                                                 str(self._src_field_file),
                                                 [args['mono_train_src']],
                                                 no_split_toks=no_split_toks)
                else:
                    log.warning("Skipping source vocab creation since mono_train_src is not given")

            # target vocabulary
            if self._tgt_field_file.exists():
                log.info(f"{self._tgt_field_file} exists. Skipping target vocab creation")
            else:
                if 'mono_train_tgt' in args:
                    log.info("Going to build target vocab from mono_train_tgt")
                    self.tgt_field = Field.train(args['pieces'], args['max_tgt_types'],
                                                 str(self._tgt_field_file),
                                                 [args['mono_train_tgt']],
                                                 no_split_toks=no_split_toks)
                else:
                    log.warning("Skipping target vocab creation since mono_train_src is not given")

        def _prep_file(file_key, out_file, do_truncate, max_len, field: Field):
            if file_key not in args:
                log.warning(f'Skipped: {file_key} because it is not found in config')
                return

            raw_file = args[file_key]
            recs = self.read_mono_raw_data(raw_file, do_truncate, max_len, field.encode_as_ids)
            self.write_mono_lines(recs, out_file)
            if args.get('text_files'):
                recs = self.read_mono_raw_data(raw_file, do_truncate, max_len, field.tokenize)
                self.write_mono_lines(recs, str(out_file).replace('.tsv', '.pieces.tsv'))

        _prep_file('mono_train_src', self.mono_train_src, args['truncate'], args['src_len'],
                   self.src_vocab)
        _prep_file('mono_train_tgt', self.mono_train_tgt, args['truncate'], args['tgt_len'],
                   self.tgt_vocab)

        _prep_file('mono_valid_src', self.mono_valid_src, args['truncate'], args['src_len'],
                   self.src_vocab)
        _prep_file('mono_valid_tgt', self.mono_valid_tgt, args['truncate'], args['tgt_len'],
                   self.tgt_vocab)

    def _pre_process_parallel(self, src_key: str, tgt_key: str, out_file: Path,
                              args: Optional[Dict[str, Any]] = None, line_check=True):
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
        parallel_recs = self.read_raw_data(args[src_key], args[tgt_key],
                                           args['truncate'], args['src_len'], args['tgt_len'],
                                           tokenizer=self.src_vocab.encode_as_ids)
        self.write_tsv(parallel_recs, out_file)

        if args.get('text_files'):
            # Redo again as plain text files
            parallel_recs = self.read_raw_data(args[src_key], args[tgt_key],
                                               args['truncate'], args['src_len'], args['tgt_len'],
                                               tokenizer=self.src_vocab.tokenize)
            self.write_tsv(parallel_recs, str(out_file).replace('.tsv', '.pieces.tsv'))

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
            'aln_emb_src': self.aln_emb_src_file,
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

    def pre_process(self, args=None):
        args = args if args else self.config['prep']
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
        if self.model_type in {'rnnlm', 'tfmlm'}:
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

    def store_model(self, epoch: int, model, train_score: float, val_score: float, keep: int,
                    prefix='model'):
        """
        saves model to a given path
        :param epoch: epoch number of model
        :param model: model object itself
        :param train_score: score of model on training split
        :param val_score: score of model on validation split
        :param keep: number of good models to keep, bad models will be deleted
        :param prefix: prefix to store model. default is "model"
        :return:
        """
        # TODO: improve this by skipping the model save if the model is not good enough to be saved
        if self.read_only:
            log.warning("Ignoring the store request; experiment is readonly")
            return
        name = f'{prefix}_{epoch:03d}_{train_score:.6f}_{val_score:.6f}.pkl'
        path = self.model_dir / name
        log.info(f"Saving epoch {epoch} to {path}")
        torch.save(model, str(path))

        for bad_model in self.list_models(sort='total_score', desc=False)[keep:]:
            log.info(f"Deleting bad model {bad_model} . Keep={keep}")
            os.remove(str(bad_model))

        with IO.writer(os.path.join(self.model_dir, 'scores.tsv'), append=True) as f:
            cols = [str(epoch), datetime.now().isoformat(), name, f'{train_score:g}',
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
        paths = self.model_dir.glob('model_*.pkl')
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
            step, train_score, valid_score = models[0].name.replace('.pkl', '').split('_')[-3:]
            return models[0], int(step)
        else:
            return None, -1

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
        self.config['optim'] = {'name': name, 'args': args}

    @property
    def src_vocab(self) -> Field:
        return self.shared_field if self.shared_field is not None else self.src_field

    @property
    def tgt_vocab(self) -> Field:
        return self.shared_field if self.shared_field is not None else self.tgt_field

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

    @property
    def pre_trained_src_emb(self):
        return torch.load(self.emb_src_file) if self.emb_src_file.exists() else None

    @property
    def pre_trained_tgt_emb(self):
        return torch.load(self.emb_tgt_file) if self.emb_tgt_file.exists() else None

    def get_train_data(self, batch_size: int, steps: int = 0, sort_dec=True, batch_first=True,
                       shuffle=False, fine_tune=False):
        inp_file = self.train_file
        if fine_tune:
            if not self.finetune_file.exists():
                # user may have added fine tune file later
                self._pre_process_parallel('finetune_src', 'finetune_tgt', self.finetune_file)
            log.info("Using Fine tuning corpus instead of training corpus")
            inp_file = self.finetune_file

        train_data = BatchIterable(inp_file, batch_size=batch_size, sort_dec=sort_dec,
                                   batch_first=batch_first, shuffle=shuffle)
        if steps > 0:
            train_data = LoopingIterable(train_data, steps)
        return train_data

    def get_val_data(self, batch_size: int, sort_dec=True, batch_first=True,
                     shuffle=False):
        return BatchIterable(self.valid_file, batch_size=batch_size, sort_dec=sort_dec,
                             batch_first=batch_first, shuffle=shuffle)

    def get_combo_data(self, batch_size: int, steps: int = 0, sort_dec=True, batch_first=True,
                       shuffle=False):
        if not self.combo_file.exists():
            # user may have added fine tune file later
            self._pre_process_parallel('combo_src', 'combo_tgt', self.combo_file)
        data = BatchIterable(self.combo_file, batch_size=batch_size, sort_dec=sort_dec,
                             batch_first=batch_first, shuffle=shuffle)
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
        for source, destination in [(self._src_field_file, other._src_field_file),
                                    (self._tgt_field_file, other._tgt_field_file),
                                    (self._shared_field_file, other._shared_field_file)]:
            if source.exists():
                log.info(f"{source} --> {destination}")
                destination.write_bytes(source.read_bytes())
                src_txt_file = source.with_name(source.name.replace('.model', '.vocab'))
                if src_txt_file.exists():
                    dst_txt_file = destination.with_name(
                        destination.name.replace('.model', '.vocab'))
                    dst_txt_file.write_bytes(src_txt_file.read_bytes())

    def get_mono_data(self, split: str, side: str, batch_size: int, sort_dec: bool = False,
                      batch_first: bool = False, shuffle: bool = False, num_batches: int = 0):
        """
        reads monolingual data
        :param split: name of the split. choices = {train, valid}
        :param side: which side ? choices = {src, tgt}
        :param batch_size: what should be batch size. example =64
        :param sort_dec: should the seqs in batch be sorted descending order of length ?
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
        data = BatchIterable(inp_file, batch_size=batch_size, sort_dec=sort_dec,
                             batch_first=batch_first, shuffle=shuffle)
        if num_batches > 0:
            data = LoopingIterable(data, num_batches)
        return data
