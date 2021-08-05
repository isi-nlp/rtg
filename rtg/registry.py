#!/usr/bin/env python
#
# Authors:
# - Thamme Gowda [tg (at) isi (dot) edu]
# - Lukas J. Ferrer [lferrer (at) isi (dot) edu]
# Created: 3/9/19

from enum import Enum

class ProblemType(str, Enum):
    TRANSLATION = "translation"
    CLASSIFICATION = "classification"


import re
from dataclasses import dataclass
from typing import Any, Optional, Mapping, Dict, Type
from rtg.exp import BaseExperiment
from rtg.module.tfmnmt import TransformerTrainer
from rtg.module.skptfmnmt import SKPTransformerTrainer
from rtg.module.wvtfmnmt import WVTransformerTrainer
from rtg.module.wvskptfmnmt import WVSKPTransformerTrainer
from rtg.module.mtfmnmt import MTransformerTrainer
from rtg.module.rnnmt import SteppedRNNMTTrainer
from rtg.lm.rnnlm import RnnLmTrainer
from rtg.lm.tfmlm import TfmLmTrainer
from rtg.module.skptfmnmt import SkipTransformerNMT
from rtg.module.wvtfmnmt import WidthVaryingTransformerNMT
from rtg.module.wvskptfmnmt import WidthVaryingSkipTransformerNMT
from rtg.module.mtfmnmt import MTransformerNMT
from rtg.module.ext.tfmextemb import TfmExtEmbNMT
from rtg.module.hybridmt import HybridMT
from rtg.emb.word2vec import CBOW
from rtg.module.ext.robertamt import RoBERTaMT
from torch import optim

from rtg.module.generator import *

# TODO: use decorators https://github.com/isi-nlp/rtg/issues/246
trainers = {
    't2t': TransformerTrainer,
    'seq2seq': SteppedRNNMTTrainer,
    'tfmnmt': TransformerTrainer,
    'skptfmnmt': SKPTransformerTrainer,
    'wvtfmnmt': WVTransformerTrainer,
    'wvskptfmnmt': WVSKPTransformerTrainer,
    'rnnmt': SteppedRNNMTTrainer,
    'rnnlm': RnnLmTrainer,
    'tfmlm': TfmLmTrainer,
    'mtfmnmt': MTransformerTrainer,
    'wv_cbow': CBOW.make_trainer,
    'tfmextembmt': TfmExtEmbNMT.make_trainer,
    'hybridmt': HybridMT.make_trainer,
    'robertamt': RoBERTaMT.make_trainer
}

# model factories
factories = {
    't2t': TransformerNMT.make_model,
    'seq2seq': RNNMT.make_model,
    'tfmnmt': TransformerNMT.make_model,
    'skptfmnmt': SkipTransformerNMT.make_model,
    'wvtfmnmt': WidthVaryingTransformerNMT.make_model,
    'wvskptfmnmt': WidthVaryingSkipTransformerNMT.make_model,
    'rnnmt': RNNMT.make_model,
    'rnnlm': RnnLm.make_model,
    'tfmlm': TfmLm.make_model,
    'mtfmnmt': MTransformerNMT.make_model,
    'tfmextembmt': TfmExtEmbNMT.make_model,
    'hybridmt': HybridMT.make_model,
    'wv_cbow': CBOW.make_model,
    'robertamt': RoBERTaMT.make_model
}

# Generator factories
generators = {
    't2t': T2TGenerator,
    'seq2seq': Seq2SeqGenerator,
    'combo': ComboGenerator,
    'tfmnmt': T2TGenerator,
    'skptfmnmt': T2TGenerator,
    'wvtfmnmt': T2TGenerator,
    'wvskptfmnmt': T2TGenerator,
    'rnnmt': Seq2SeqGenerator,
    'rnnlm': RnnLmGenerator,
    'tfmlm': TfmLmGenerator,
    'mtfmnmt': MTfmGenerator,
    'hybridmt': MTfmGenerator,
    'tfmextembmt': TfmExtEembGenerator,
    'robertamt': T2TGenerator,

    'wv_cbow': CBOW.make_model  # FIXME: this is a place holder
}

#  TODO: simplify this; use decorators to register directly from class's code

####
MODEL = 'model'
OPTIMIZER = 'optimizer'
SCHEDULE = 'schedule'
CRITERION = 'criterion'

registry = {
    MODEL: dict(),
    OPTIMIZER: dict(
        adam=optim.Adam,
        sgd=optim.SGD,
        adagrad=optim.Adagrad,
        adam_w=optim.AdamW,
        adadelta=optim.Adadelta,
        sparse_adam=optim.SparseAdam),
    SCHEDULE: dict(),
    CRITERION: dict(),
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


@dataclass
class Model:
    name: str
    Model: Any
    Trainer: Any
    Generator: Any
    Experiment: Type[BaseExperiment]

    def experiment(self, work_dir, *args, **kwargs):
        return self.Experiment(work_dir, *args, **kwargs)


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
        m = Model(name=_name, Model=getattr(cls, 'make_model'),
                  Trainer=getattr(cls, 'make_trainer'),
                  Generator=getattr(cls, 'make_generator', None),
                  Experiment=getattr(cls, 'experiment_type'))
        registry[kind][_name] = m
        log.info(f"registering model: {_name}")
        # for backward compat, also add to the dictionaries, (until we transition fully)
        trainers[_name] = m.Trainer
        factories[_name] = m.Model
        generators[_name] = m.Generator
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
        'rtg.emb.tfmcls',
    ]
    for name in modules:
        import_module(name)

__register_all()

if __name__ == '__main__':
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