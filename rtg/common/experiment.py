import os
import copy
import hashlib
import os
import random
import sys
import time
from datetime import datetime, timedelta
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import portalocker
import torch


from rtg import IO, device, log, yaml
from rtg.data.codec import Field, NLField, PretrainMatchField, SPField

from rtg.registry import CRITERION, MODEL, OPTIMIZER, SCHEDULE

from .schema import config_checks

seeded = False


__all__ = ['BaseExperiment', 'load_conf']


def load_conf(inp: Union[str, Path], update_env=True):
    with IO.reader(inp) as fh:
        config = yaml.load(fh)
    if update_env and isinstance(config.get('environment'), dict):
        for name, val in config['environment'].items():
            if name in os.environ and os.environ[name] != str(val):
                log.warning(f"Overriding env var {name}={os.environ[name]} → {val}")
            os.environ[name] = str(val)
    return config


class BaseExperiment:
    def __init__(
        self,
        work_dir: Union[str, Path],
        read_only=False,
        config: Union[str, Path, Optional[Dict[str, Any]]] = None,
    ):
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
        config_checks(self.config)
        self.codec_name = self.config.get('prep', {}).get('codec_lib', 'sentpiece')  # with default
        codec_libs = {'sentpiece': SPField, 'nlcodec': NLField, 'pretrainmatch': PretrainMatchField}
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

        self.shared_field = (
            self.Field(str(self._shared_field_file)) if self._shared_field_file.exists() else None
        )

        self.last_state_file = self.model_dir / 'last_state.pt'
        self.parent_model_state = self.data_dir / 'parent_model_state.pt'

    @property
    def problem_type(self):
        raise NotImplementedError()

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

    def store_model(
        self,
        optimizer_step: int,
        model_state,
        keep: int = -1,
        prefix='model'
    ):
        """
        saves model to a given path
        :param optimizer_step: optimizer step of the model
        :param model: model object itself
        :param keep: number of good models to keep, bad models will be deleted. default: keep all
        :param prefix: prefix to store model. default is "model"
        :param keeper_sort: criteria for choosing the old or bad models for deletion.
            Choices: {'total_score', 'step'}
        :return:
        """
        # TODO: improve this by skipping the model save if the model is not good enough to be saved
        if self.read_only:
            log.warning("Ignoring the store request; experiment is readonly")
            return
        name = f'{prefix}.step_{optimizer_step}.pkl'
        path = self.model_dir / name
        log.info(f"Saving optimizer step {optimizer_step} to {path}")
        torch.save(model_state, str(path))

        if keep > 0:
            del_models = []
            del_models = self.list_models(sort_by='step', desc=True)[keep:]
            for d_model, _step_num in del_models:
                log.info(f"Deleting model @ step={_step_num} :: {d_model}. Keep={keep} ")
                os.remove(str(d_model))

        if self.last_state_file.exists():
            self.last_state_file.unlink()
        self.last_state_file.symlink_to(name)  # in the same dir


    def list_models(self, sort_by: str = 'step', desc: bool = True) -> List[Path]:
        """
        Lists models in descending order of modification time
        :param sort_by: how to sort models ? default=step
        :param desc: True to sort in reverse (default); False to sort in ascending
        :return: list of model paths
        """
        paths = list(self.model_dir.glob(f'model.{sort_by}_*.pkl'))
        paths = [(p, p.name.replace('.pkl', '').replace(f'model.{sort_by}_', '')) for p in paths]
        if sort_by == 'step':
            paths = [(p, int(n)) for p, n in paths]
        else:
            paths = [(p, float(n)) for p, n in paths]

        return list(sorted(paths, key=lambda x: x[1], reverse=desc))

    def _get_first_model(self, sort_by: str, desc: bool) -> Tuple[Optional[Path], int]:
        """
        Gets the first model that matches the given sort criteria
        :param sort_by: sort mechanism
        :param desc: True for descending, False for ascending
        :return: Tuple[Optional[Path], step_num:int]
        """
        models = self.list_models(sort_by=sort_by, desc=desc)
        if models:
            return models[0]
        else:
            return None, 0

    def get_last_saved_model(self) -> Tuple[Optional[Path], int]:
        return self._get_first_model(sort_by='step', desc=True)

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
        if 'parent' in self.config and not self.parent_model_state.exists():
            self.inherit_parent()
        if self.has_prepared() and not force:
            log.warning("Already prepared")
            return
        args = args if args else self.config['prep']

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
        raise NotImplementedError()

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
                for key in key_set:  # Running average
                    state_dict[key] = (i * state_dict[key] + next_state[key]) / (i + 1)
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
        from ..registry import MODELS

        model = MODELS[self.model_type].Model(exp=self, **self.model_args)[0]
        state = self.maybe_ensemble_state(model_paths=model_paths, ensemble=ensemble)
        errors = model.load_state_dict(state)
        log.info(f"{errors}")
        return model

    def load_model_with_state(self, checkpt_state):
        from ..registry import MODELS

        chkpt = checkpt_state
        state = chkpt['model_state']
        model_type = chkpt['model_type']
        model_args = chkpt['model_args']
        model = MODELS[model_type].Model(exp=self, **model_args)[0]
        errors = model.load_state_dict(state)
        log.info(f"{errors}")
        log.info(f"Successfully restored the model state of : {model_type}")
        return model

    def get_conf_component(self, kind, extra_args=None):
        """Creates a component such as schedule, criterion, optimizer based on config"""
        from rtg.registry import registry

        assert kind in registry, f'component {kind} is unknown; valid: {registry.keys()}'
        if not kind in self.config:
            log.warning(f"{kind} not found in config; skipping")
            return None
        name, args = self.config[kind]['name'], self.config[kind].get('args') or {}
        assert name in registry[kind], f'{kind}={name} is invalid; valid: {registry[kind].keys()}'
        factory = registry[kind][name]
        extra_args = extra_args or {}
        log.info(f"creating {kind} {name} with args {args}")
        return factory(**extra_args, **args)

    def get_criterion(self, extra_args=None):
        return self.get_conf_component(CRITERION, extra_args=extra_args)

    def get_schedule(self):
        return self.get_conf_component(SCHEDULE)

    def get_optimizer(self, params):
        return self.get_conf_component(OPTIMIZER, extra_args=dict(params=params))
