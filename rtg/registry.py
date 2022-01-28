#!/usr/bin/env python
#
# Authors:
# - Thamme Gowda [tg (at) isi (dot) edu]
# - Lukas J. Ferrer [lferrer (at) isi (dot) edu]
# Created: 3/9/19
import json
import re
from enum import Enum
from dataclasses import dataclass
from typing import Any, Dict, Type, Callable

from rtg import log
from torch import optim


class ProblemType(str, Enum):
    TRANSLATION = "translation"
    CLASSIFICATION = "classification"


@dataclass
class ModelSpec:
    name: str
    Model: Any
    Trainer: Any
    Generator: Any
    Experiment: Type['BaseExperiment']


MODEL = 'model'
MODELS: Dict[str, ModelSpec] = {}

OPTIMIZER = 'optimizer'
OPTIMIZERS: Dict[str, Type[optim.Optimizer]] = dict(
    adam=optim.Adam,
    sgd=optim.SGD,
    adagrad=optim.Adagrad,
    adam_w=optim.AdamW,
    adadelta=optim.Adadelta,
    sparse_adam=optim.SparseAdam)
try:
    # this is still experimental
    import adabound
    OPTIMIZERS['ada_bound'] = adabound.AdaBound
except:
    pass

SCHEDULE = 'schedule'
SCHEDULES: Dict[str, Any] = {}

CRITERION = 'criterion'
CRITERIA: Dict[str, Any] = {}

TRANSFORM = 'transform'     # pre and post processing
TRANSFORMS: Dict[str, Callable[[str], str]] = {}   # str -> str

registry = {
    MODEL: MODELS,
    OPTIMIZER: OPTIMIZERS,
    SCHEDULE: SCHEDULES,
    CRITERION: CRITERIA,
    TRANSFORM: TRANSFORMS
}


def snake_case(word):
    """
    Converts a word (from CamelCase) to snake Case
    :param word:
    :return:
    """
    word = re.sub(r"([A-Z]+)([A-Z][a-z])", r'\1_\2', word)
    word = re.sub(r"([a-z\d])([A-Z])", r'\1_\2', word)
    word = word.replace("-", "_")
    return word.lower()


def register(kind, name=None):
    """
    A decorator for registering modules
    :param kind: what kind of component :py:const:MODEL, :py:const:OPTIMIZER, :py:const:SCHEDULE
    :param name: (optional) name for this component
    :return:
    """
    assert kind in registry

    def _register_model(cls):
        attrs = ['model_type', 'make_model', 'make_trainer', 'experiment_type']
        for attr in attrs:
            assert hasattr(cls, attr), f'{cls}.{attr} is expected but not defined'

        _name = name or cls.model_type
        assert _name, f'name is required for {cls}'
        assert isinstance(_name, str), f'name={_name} is not a string'
        assert _name not in registry[kind], f'{_name} model type is already registered.'
        m = ModelSpec(name=_name, Model=getattr(cls, 'make_model'),
                      Trainer=getattr(cls, 'make_trainer'),
                      Generator=getattr(cls, 'make_generator', None),
                      Experiment=getattr(cls, 'experiment_type'))
        registry[kind][_name] = m
        log.debug(f"registering model: {_name}")
        return cls

    def _wrap_cls(cls):
        registry[kind][name or snake_case(cls.__name__)] = cls
        return cls

    if kind == MODEL:
        return _register_model
    else:
        return _wrap_cls


def __register_all():
    # import, so register() calls can happen
    from importlib import import_module
    modules = [
        'rtg.module.tfmnmt',
        'rtg.module.skptfmnmt',
        'rtg.module.wvtfmnmt',
        'rtg.module.wvskptfmnmt',
        'rtg.module.rnnmt',
        'rtg.module.ext.tfmextemb',
        'rtg.module.ext.robertamt',
        'rtg.module.mtfmnmt',
        'rtg.module.hybridmt',
        'rtg.lm.rnnlm',
        'rtg.lm.tfmlm',
        'rtg.emb.word2vec',
        'rtg.emb.tfmcls',
        'rtg.module.criterion',
        'rtg.module.schedule',
        'rtg.module.subcls_tfmnmt',
    ]
    for name in modules:
        import_module(name)
    msg = []
    for k, v in registry.items():
        msg.append(f'{k}:\t' + ', '.join(v.keys()))
    msg = '\n  '.join(msg)
    log.debug(f"Registered all components; your choices are ::\n  {msg}")


if __name__ == '__main__':
    from rtg.exp import BaseExperiment
    # a simple test case

    @register(MODEL)
    class MyModel:
        model_type = 'mymodel'
        experiment_type = BaseExperiment

        @classmethod
        def make_model(self):
            pass

        @classmethod
        def make_trainer(self):
            pass

    print(registry[MODEL])
