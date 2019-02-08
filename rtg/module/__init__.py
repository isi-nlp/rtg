from abc import ABCMeta, abstractmethod, ABC
import torch.nn as nn


class Model(nn.Module):

    @property
    @abstractmethod
    def model_dim(self):
        pass

    @property
    @abstractmethod
    def model_type(self):
        pass

    @property
    @abstractmethod
    def vocab_size(self):
        pass

    @classmethod
    @abstractmethod
    def make_model(cls, *args, **kwargs):
        raise NotImplementedError


class NMTModel(Model, metaclass=ABCMeta):
    """"
    base class for all Sequence to sequence (NMT) models
    """
    # TODO: move stuff here that is common to all
    pass
