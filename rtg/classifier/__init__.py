#!/usr/bin/env python
#
# Author: Thamme Gowda [tg (at) isi (dot) edu]
# Created: 3/16/19


from abc import ABCMeta
from rtg import BaseModel

from .exp import ClassificationExperiment


class ClassifierModel(BaseModel, metaclass=ABCMeta):
    """Base class for all classification models"""

    experiment_type = ClassificationExperiment


from .trainer import ClassifierTrainer
from . import tfmcls