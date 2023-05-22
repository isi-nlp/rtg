#!/usr/bin/env python
#
# Author: Thamme Gowda [tg (at) isi (dot) edu]
# Created: 3/16/19


from abc import ABCMeta
from rtg import BaseModel

from .experiment import ClassificationExperiment


class ClassifierModel(BaseModel, metaclass=ABCMeta):
    """Base class for all classification models"""

    experiment_type = ClassificationExperiment
    
    def __init__(self, n_classes:int, **kwargs) -> None:
        self._n_classes = n_classes
        super().__init__()

    @property
    def n_classes(self):
        return self._n_classes

from .trainer import ClassifierTrainer
from . import transformer
