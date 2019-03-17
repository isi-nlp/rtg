from abc import ABCMeta, abstractmethod, ABC
import torch.nn as nn


class Model(nn.Module):

    def init_params(self, scheme='xavier'):
        assert scheme == 'xavier'  # only supported scheme as of now
        # Initialize parameters with xavier uniform
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

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

    @classmethod
    @abstractmethod
    def make_trainer(cls, *args, **kwargs):
        raise NotImplementedError


class NMTModel(Model, metaclass=ABCMeta):
    """"
    base class for all Sequence to sequence (NMT) models
    """
    # TODO: move stuff here that is common to all
    pass

    @classmethod
    @abstractmethod
    def make_generator(cls, *args, **kwargs):
        raise NotImplementedError
