#!/usr/bin/env python
#
# Author: Thamme Gowda [tg (at) isi (dot) edu]
# Created: 1/31/19
from abc import ABCMeta, abstractmethod

from rtg import log, BaseExperiment
from rtg.common.model import BaseModel

"""
Stuff related to Language Model
"""


class LanguageModel(BaseModel, metaclass=ABCMeta):
    """base class for all models that generate sequence"""

    experiment_type = BaseExperiment

    @classmethod
    @abstractmethod
    def make_generator(cls, *args, **kwargs):
        raise NotImplementedError
