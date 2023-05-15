import copy
from typing import  List

import torch
import torch.nn as nn
import torch.nn.functional as F

from rtg import get_my_args, log, register_model
from rtg.nmt.tfmnmt import (Embeddings, Encoder, EncoderLayer,
                            MultiHeadedAttention, PositionalEncoding,
                            PositionwiseFeedForward)

from . import ClassificationExperiment, ClassifierModel, ClassifierTrainer


class SentenceCompressor(nn.Module):
    """
    Compresses token representation into a single vector
    """

    def __init__(self, d_model: int, attn_heads: int = 0, dropout: float = 0.1):
        """ Reduces a sequence of vectors into a single vector
        :param d_model: Dimension of the input vector
        :param attn_heads: Number of heads in the multi-headed attention. if 0, it is set to d_model//64
        :param dropout: Dropout probability for compressor multi-headed attention
        """
        super(SentenceCompressor, self).__init__()
        self.cls_repr = nn.Parameter(torch.zeros(d_model))
        self.d_model = d_model
        if attn_heads == 0:
            assert d_model % 64 == 0, f"Multi-head attention requires d_model to be a multiple of 64. Given={d_model}"
            attn_heads = int(d_model // 64)
        self.attn = MultiHeadedAttention(h=attn_heads, d_model=d_model, dropout=dropout)

    def forward(self, src, src_mask):
        B, T, D = src.size()  # [Batch, Time, Dim]
        assert D == self.d_model
        query = self.cls_repr.view(1, 1, D).repeat(B, 1, 1)
        # Args: Query, Key, Value, Mask
        cls_repr = self.attn(query, src, src, src_mask)
        cls_repr = cls_repr.view(B, D)  # [B, D]
        return cls_repr


class ClassifierHead(nn.Module):
    scores = {
        'logits': lambda x, dim=None: x,
        'softmax': F.softmax,
        'probs': F.softmax,
        'log_softmax': F.log_softmax,
        'log_probs': F.log_softmax,
        'sigmoid': lambda x, dim=None: x.sigmoid(),
    }

    def __init__(self, d_model: int, n_classes: int):
        super().__init__()
        self.d_model = d_model
        self.n_classes = n_classes
        self.proj = nn.Linear(d_model, n_classes)

    def forward(self, repr, score='logits'):
        score = score or 'logits'
        B, D = repr.shape  # [Batch, Dim]
        assert D == self.d_model
        assert score in self.scores, f'"score", Given={score}, known={list(self.scores.keys())}'
        cls_repr = self.proj(repr)
        return self.scores[score](cls_repr, dim=-1)


@register_model()
class TransformerClassifier(ClassifierModel):
    model_type = 'transformer-classifier'
    experiment_type = ClassificationExperiment

    EncoderFactory = Encoder
    EncoderLayerFactory = EncoderLayer
    CompressorFactory = SentenceCompressor
    ClassifierHeadFactory = ClassifierHead

    def __init__(self, encoder: Encoder, src_embed, compressor: SentenceCompressor, classifier_head: ClassifierHead):
        super().__init__()
        self.encoder: Encoder = encoder
        self.src_embed = src_embed
        self.compressor = compressor
        self.classifier_head = classifier_head

    def get_trainable_params(self, include=None, exclude=None):
        if not include and not exclude or include == 'all':
            return super().get_trainable_params()
        if exclude:
            raise Exception("Exclude not supported yet. Please use include")
            # TODO: implement it later when it is really really needed!
        assert isinstance(include, list)
        # a valid example for include
        # 'src_embed', 'compressor', 'classifier', 'encoder:0,1,2,3,...,n-1',  # encoder:layers
        param_groups = []
        for sub_name in include:
            if hasattr(self, sub_name):
                log.info(f"Trainable parameters <-- {sub_name}")
                param_groups.extend(getattr(self, sub_name).parameters())
            elif sub_name.startswith('encoder:'):  # sub select layers
                sub_name, layers = sub_name.split(':')  # encoder:layer_idx
                layers = [int(x) for x in layers.split(',')]
                for layer_idx in layers:
                    log.info(f'Trainable parameters <-- {sub_name}[{layer_idx}] ')
                    layer = self.encoder.layers[layer_idx]
                    param_groups.extend(layer.parameters())
                if len(self.encoder.layers) - 1 in layers:  # the last layer is trainable, then norm
                    log.info(f'Trainable parameters <-- {sub_name}.norm')
                    param_groups.extend(self.encoder.norm.parameters())
            else:
                raise Exception(f'{sub_name} not supported or invalid')
        return param_groups

    @property
    def model_dim(self):
        return self.classifier_head.d_model

    @property
    def vocab_size(self) -> int:
        return self.classifier_head.n_classes

    def encode(self, src, src_mask):
        tok_repr = self.encoder(self.src_embed(src), src_mask)
        return self.compressor(tok_repr, src_mask)

    def forward(self, src, src_mask, score='logits'):
        "Take in and process masked src and target sequences."
        sent_repr = self.encode(src, src_mask)
        if score == 'embedding':  # sentence embedding
            return sent_repr
        return self.classifier_head(sent_repr, score=score)

    @classmethod
    def make_model(
        cls,
        src_vocab: int,
        tgt_vocab: int,
        enc_layers=6,
        hid_size=512,
        ff_size=2048,
        n_heads=8,
        attn_bias=True,
        attn_dropout=0.1,
        dropout=0.1,
        activation='relu',
        exp: ClassificationExperiment = None,
    ):
        "Helper: Construct a model from hyper parameters."

        # get all args for reconstruction at a later phase
        args = get_my_args(exclusions=['cls', 'exp'])
        assert activation in {'relu', 'elu', 'gelu'}
        assert enc_layers > 0, "Zero encoder layers! Hmm🤔"

        log.info(f"Make model, Args={args}")
        c = copy.deepcopy
        attn = MultiHeadedAttention(n_heads, hid_size, dropout=attn_dropout, bias=attn_bias)
        ff = PositionwiseFeedForward(hid_size, ff_size, dropout, activation=activation)
        encoder = cls.EncoderFactory(cls.EncoderLayerFactory(hid_size, c(attn), c(ff), dropout), enc_layers)
        src_emb = nn.Sequential(Embeddings(hid_size, src_vocab), PositionalEncoding(hid_size, dropout))
        classifier_head = cls.ClassifierHeadFactory(d_model=hid_size, n_classes=tgt_vocab)
        compressor = cls.CompressorFactory(d_model=hid_size, attn_heads=n_heads, dropout=dropout)

        model = cls(encoder, src_emb, compressor=compressor, classifier_head=classifier_head)

        model.init_params()
        return model, args

    @classmethod
    def make_trainer(cls, *args, **kwargs):
        return ClassifierTrainer(*args, model_factory=cls.make_model, **kwargs)


