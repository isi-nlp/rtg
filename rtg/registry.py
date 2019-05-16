#!/usr/bin/env python
#
# Author: Thamme Gowda [tg (at) isi (dot) edu] 
# Created: 3/9/19

from rtg.module.tfmnmt import TransformerTrainer
from rtg.module.mtfmnmt import MTransformerTrainer
from rtg.module.rnnmt import SteppedRNNMTTrainer
from rtg.binmt.bicycle import BiNmtTrainer
from rtg.lm.rnnlm import RnnLmTrainer
from rtg.lm.tfmlm import TfmLmTrainer
from rtg.module.mtfmnmt import MTransformerNMT
from rtg.emb.word2vec import CBOW
from rtg.module.generator import *

trainers = {
    't2t': TransformerTrainer,
    'binmt': BiNmtTrainer,
    'seq2seq': SteppedRNNMTTrainer,
    'tfmnmt': TransformerTrainer,
    'rnnmt': SteppedRNNMTTrainer,
    'rnnlm': RnnLmTrainer,
    'tfmlm': TfmLmTrainer,
    'mtfmnmt': MTransformerTrainer,
    'wv_cbow': CBOW.make_trainer
}

# model factories
factories = {
    't2t': TransformerNMT.make_model,
    'seq2seq': RNNMT.make_model,
    'binmt': BiNMT.make_model,
    'tfmnmt': TransformerNMT.make_model,
    'rnnmt': RNNMT.make_model,
    'rnnlm': RnnLm.make_model,
    'tfmlm': TfmLm.make_model,
    'mtfmnmt': MTransformerNMT.make_model,
    'wv_cbow': CBOW.make_model
}

# Generator factories
generators = {'t2t': T2TGenerator,
              'seq2seq': Seq2SeqGenerator,
              'binmt': BiNMTGenerator,
              'combo': ComboGenerator,
              'tfmnmt': T2TGenerator,
              'rnnmt': Seq2SeqGenerator,
              'rnnlm': RnnLmGenerator,
              'tfmlm': TfmLmGenerator,
              'mtfmnmt': MTfmGenerator,
              'hybridmt': MTfmGenerator,
              'tfmnmt_nomax': T2TGenerator,
              'wv_cbow': CBOW.make_model  # FIXME: this is a place holder
              }
