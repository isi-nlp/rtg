from abc import ABCMeta
from rtg.common.model import BaseModel
from rtg.lm import LanguageModel
from .exp import TranslationExperiment

class NMTModel(LanguageModel, metaclass=ABCMeta):
    """
    base class for all Sequence to sequence (NMT) models
    """

    experiment_type = TranslationExperiment




from . import decoder, generator, rnnmt, robertamt, tfmnmt

# dont do * import; the module/model names might clash
