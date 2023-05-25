# here, COMET refers to Unbabel's COMET model
# This implementation is NOT a strict replica of Unbabel's COMET model
# but inspired by it and uses the same name for some of the components
from .data import HFField, Example, Batch

#from .experiment import HFCometExperiment
from .rtg_comet import RTGCometClassifier, CometExperiment, CometTrainer
from .hf_comet import HFCometExperiment, HFCometClassifier, HFCometTrainer
