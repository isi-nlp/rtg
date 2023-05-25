from typing import List, Callable
import copy


import torch
from torch import nn


from rtg import log, device, get_my_args, register_model, Batch
from rtg.classifier import ClassifierModel, ClassificationExperiment, ClassifierTrainer
from rtg.comet.experiment import CometExperiment

from rtg.classifier.transformer import (
    Embeddings,
    Encoder,
    EncoderLayer,
    MultiHeadedAttention,
    PositionalEncoding,
    PositionwiseFeedForward,   
    ClassifierHead, 
    SentenceCompressor,
    TransformerClassifier
)
from rtg.nmt.transformer import Encoder


@register_model()
class RTGCometClassifier(TransformerClassifier):
    model_type = 'rtg-comet-cls'
    experiment_type = CometExperiment
   
    def __init__(self, *args, freeze_encoder=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.freeze_encoder = freeze_encoder

    def encode(self, src, src_mask):
        tok_repr = self.encoder(self.src_embed(src), src_mask)
        return self.compressor(tok_repr, src_mask)

    def forward(self, src, src_mask, score='logits'):
        "Take in and process masked src and target sequences."
        sent_repr = self.encode(src, src_mask)
        if score == 'embedding':  # sentence embedding
            return sent_repr
        return self.classifier_head(sent_repr, score=score)


    def forward(self, seq1, seq2, seq1_mask, seq2_mask, score='logits'):
        # def forward(self, src, src_mask, score='logits', freeze_encoder=True):
        with torch.set_grad_enabled(not self.freeze_encoder):
            seq1_repr = self.encode(seq1, seq1_mask)
            seq2_repr = self.encode(seq2, seq2_mask)
        combo_repr = self.comet_repr(seq1_repr, seq2_repr)
        return self.classifier_head(combo_repr, score=score)

    def comet_repr(self, seq1_repr, seq2_repr):
        # [seq1, seq2, |seq1-seq2|, seq1.seq2]
        return torch.cat(
            [seq1_repr, seq2_repr, torch.abs(seq1_repr - seq2_repr), seq1_repr * seq2_repr], dim=1
        )

    @classmethod
    def make_model(
        cls,
        exp: ClassificationExperiment,
        src_vocab: int,
        n_classes: int,
        enc_layers=6,
        hid_size=512,
        ff_size=2048,
        n_heads=8,
        attn_bias=True,
        attn_dropout=0.1,
        dropout=0.1,
        activation='relu',
        freeze_encoder=False,
    ) -> ClassifierModel:
        args = get_my_args(exclusions=['exp', 'cls'])
        log.info(f"Creating model {cls.__name__} with args: {args}")
        
        
        c = copy.deepcopy
        attn = MultiHeadedAttention(n_heads, hid_size, dropout=attn_dropout, bias=attn_bias)
        ff = PositionwiseFeedForward(hid_size, ff_size, dropout, activation=activation)
        encoder = cls.EncoderFactory(cls.EncoderLayerFactory(hid_size, c(attn), c(ff), dropout), enc_layers)
        
        src_emb = nn.Sequential(Embeddings(hid_size, src_vocab), PositionalEncoding(hid_size, dropout))
        
        compressor_attn = MultiHeadedAttention(h=n_heads, d_model=hid_size, dropout=dropout)
        compressor = cls.CompressorFactory(d_model=hid_size, attn=compressor_attn)

        classifier_head = cls.ClassifierHeadFactory(input_dim=hid_size * 4, n_classes=n_classes)
        model = cls(encoder, src_emb, compressor=compressor, classifier_head=classifier_head)
        model.init_params()
        return model, args

    @classmethod
    def make_trainer(cls, *args, **kwargs):
        return CometTrainer(*args, model_factory=cls.make_model, **kwargs)


class CometTrainer(ClassifierTrainer):
    def _batch_step(self, batch: Batch, take_step=False, train_mode=False):
        """Take a single step of training or validation on a batch
        :param batch: batch object
        :param take_step: whether to take optimizer step  (requires train_mode=True). Useful for gradient accumulation.
        :param train_mode: whether to run in train mode i.e., with grads no grads
        """
        x1_mask = (batch.x1s != batch.pad_val).unsqueeze(1)
        x2_mask = (batch.x2s != batch.pad_val).unsqueeze(1)
        scores = self.model(
            seq1=batch.x1s,
            seq2=batch.x2s,
            seq1_mask=x1_mask,
            seq2_mask=x2_mask,
            score=self.criterion.input_type,
        )
        loss = self.loss_func(scores=scores, labels=batch.ys, train_mode=train_mode, take_step=take_step)
        return loss, scores
