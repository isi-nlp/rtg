from abc import ABCMeta, abstractmethod, ABC
import torch
import torch.nn as nn
from rtg import log


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
    # FIXME: @abstractmethod
    def make_trainer(cls, *args, **kwargs):
        raise NotImplementedError

    def maybe_init_from_parent(self, exp: 'Experiment'):
        log.info("NOT initialising from parent model")
        pass


class NMTModel(Model, metaclass=ABCMeta):
    """"
    base class for all Sequence to sequence (NMT) models
    """
    # TODO: move stuff here that is common to all
    pass

    @classmethod
    # FIXME: @abstractmethod
    def make_generator(cls, *args, **kwargs):
        raise NotImplementedError

    def maybe_init_from_parent(self, exp: 'TranslationExperiment'):
        if exp.parent_model_state.exists():
            log.info("YES Initialising from a parent model")
            device = next(self.parameters()).device  # device of self model
            state = torch.load(exp.parent_model_state, map_location=device)
            error = self.load_state_dict(state, strict=False)
            log.info("YES Initialized from the parent model")
            if error.missing_keys or error.unexpected_keys:
                log.warning(f"Error keys: {error}")
        else:
            log.info("NOT initialising from parent model")

    def get_trainable_params(self, include=None, exclude=None):
        """
        sub-selects parameters of the model that the optimizer can mess to reach an optima.
        by default then include and exclude are None, it returns all parameters.
        :param include:  only include these. Default is None
        :param exclude:  exclude these. Default is None
        :return:
        """
        if include or exclude:
            raise NotImplementedError(f'Sub-selection of trainable params is not implemented for'
                                      f' "{self.model_type}", please do get_trainable_params(...)')
        else:
            log.info("Treating all parameters as trainable parameters")
            return list(self.parameters())   # default include all
