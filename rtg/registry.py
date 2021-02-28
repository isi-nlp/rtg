#!/usr/bin/env python
#
# Authors:
# - Thamme Gowda [tg (at) isi (dot) edu]
# - Lukas J. Ferrer [lferrer (at) isi (dot) edu]
# Created: 3/9/19


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