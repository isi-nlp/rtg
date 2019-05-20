#!/usr/bin/env python
# Transformer model with external embeddings
# Author: Thamme Gowda [tg (at) isi (dot) edu] 
# Created: 2019-05-17

from rtg.module.tfmnmt import (
    TransformerTrainer, TransformerNMT, attention, MultiHeadedAttention, PositionalEncoding,
    Encoder, EncoderLayer, PositionwiseFeedForward, Decoder, DecoderLayer, Embeddings, Generator)
from torch import nn
from rtg import TranslationExperiment as Experiment, log
from rtg.dataprep import PAD_TOK_IDX as padding_idx
import inspect
import torch
import copy


class TfmExtEmbNMT(TransformerNMT):

    @property
    def model_type(self):
        return "tfmextembmt"

    def init_ext_embedding(self, src_ext_emb_wt, tgt_ext_emb_wt, freeze=True):
        src_vocab, src_dim = src_ext_emb_wt.shape
        tgt_vocab, tgt_dim = tgt_ext_emb_wt.shape
        assert src_dim == self.ext_emb_dim
        assert tgt_dim == self.ext_emb_dim

        log.info(f"Initializing SRC ext embs, freeze={freeze}")
        self.src_ext_emb = nn.Embedding(src_vocab, src_dim, padding_idx=padding_idx,
                                        _weight=src_ext_emb_wt)
        self.src_ext_emb.weight.requires_grad = not freeze  # Freeze

        log.info(f"Initializing TGT ext embs, freeze={freeze}")
        self.tgt_ext_emb = nn.Embedding(tgt_vocab, tgt_dim, padding_idx=padding_idx,
                                        _weight=tgt_ext_emb_wt)
        self.tgt_ext_emb.weight.requires_grad = not freeze  # Freeze


    def forward(self, src, tgt, src_mask, tgt_mask, gen_probs=False, log_probs=True):
        "Take in and process masked src and target sequences."
        enc_outs = self.encode(src, src_mask)
        src_ext_emb = self.src_ext_emb(src)
        tgt_ext_emb = self.tgt_ext_emb(tgt)
        feats = self.decode(enc_outs, src_mask, tgt, tgt_mask, src_ext_emb, tgt_ext_emb)
        return self.generator(feats, log_probs=log_probs) if gen_probs else feats

    def decode(self, memory, src_mask, tgt, tgt_mask, src_ext_emb, tgt_ext_emb):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask, src_ext_emb, tgt_ext_emb)

    @classmethod
    def make_model(cls, src_vocab, tgt_vocab, n_layers=6, hid_size=512, ff_size=2048, n_heads=8,
                   dropout=0.1, tied_emb='three-way', ext_emb_dim: int = 0, exp: Experiment = None):
        # get all args for reconstruction at a later phase
        _, _, _, args = inspect.getargvalues(inspect.currentframe())
        for exclusion in ['cls', 'exp']:
            del args[exclusion]  # exclude some args
        assert ext_emb_dim > 0

        c = copy.deepcopy
        attn = MultiHeadedAttention(n_heads, hid_size, dropout=dropout)
        ff = PositionwiseFeedForward(hid_size, ff_size, dropout)

        encoder = Encoder(EncoderLayer(hid_size, c(attn), c(ff), dropout), n_layers)
        src_attn_ext = MultiHeadedAttentionExt(n_heads, hid_size, query_dim=hid_size + ext_emb_dim,
                                           dropout=dropout)
        decoder = DecoderExt(DecoderLayerExt(hid_size, c(attn), src_attn_ext, c(ff), dropout), n_layers)

        src_emb = nn.Sequential(Embeddings(hid_size, src_vocab),
                                PositionalEncoding(hid_size, dropout))
        tgt_emb = nn.Sequential(Embeddings(hid_size, tgt_vocab),
                                PositionalEncoding(hid_size, dropout))
        generator = Generator(hid_size, tgt_vocab)

        model = cls(encoder, decoder, src_emb, tgt_emb, generator)
        model.ext_emb_dim = ext_emb_dim

        if tied_emb:
            model.tie_embeddings(tied_emb)

        model.init_params()
        return model, args

    @classmethod
    def make_trainer(cls, *args, **kwargs):
        return TfmExtEmbTrainer(*args, **kwargs)


class TfmExtEmbTrainer(TransformerTrainer):

    def __init__(self, *args, model_factory=TfmExtEmbNMT.make_model, **kwargs):
        super().__init__(*args, model_factory=model_factory, **kwargs)
        assert isinstance(self.model, TfmExtEmbNMT)  # type check


class MultiHeadedAttentionExt(nn.Module):
    def __init__(self, h, d_model, query_dim, dropout=0.1):
        "Take in model size and number of heads."
        super().__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = nn.ModuleList([
            nn.Linear(query_dim, d_model),  # query
            nn.Linear(query_dim, d_model),  # key
            nn.Linear(d_model, d_model),  # Value
            nn.Linear(d_model, d_model)])  # result
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)  # [BatchSize x 1 x Time x SeqLen]  1=Broadcast for all heads
        batch_size = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = \
            [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]
        # Q,K,V  --> input, linear: [BatchSize x SeqLen x ModelDim]
        #        --> view: [BatchSize x SeqLen x Heads x ModelDim/Heads ]
        #        --> transpose: [BatchSize x Heads x SeqLen x ModelDim/Heads ]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)
        # x : [BatchSize x Heads x SeqLen x ModelDim/Heads ]

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)
        # x : transpose [BatchSize x SeqLen x Heads x ModelDim/Heads ]
        # x : view [BatchSize x SeqLen x ModelDim ]

        return self.linears[-1](x)


class DecoderLayerExt(DecoderLayer):
    "Decoder is made of self-attn, src-attn, and feed forward (defined below)"

    def src_attn_ext(self, tgt, tgt_ext, src_mem, src_ext, src_mask):
        key = torch.cat([src_mem, src_ext], dim=-1)
        qry = torch.cat([tgt, tgt_ext], dim=-1)
        return self.src_attn(qry, key, value=src_mem, mask=src_mask)

    def forward(self, x, memory, src_mask, tgt_mask, src_ext_embs, tgt_ext_embs):
        "Follow Figure 1 (right) for connections."
        x = self.sublayer[0](x, lambda _x: self.self_attn(_x, _x, _x, tgt_mask))
        x = self.sublayer[1](x, lambda _x: self.src_attn_ext(_x, tgt_ext_embs,
                                                             memory, src_ext_embs, src_mask))
        return self.sublayer[2](x, self.feed_forward)


class DecoderExt(Decoder):

    def forward(self, x, memory, src_mask, tgt_mask, src_ext_embs, tgt_ext_embs):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask, src_ext_embs, tgt_ext_embs)
        return self.norm(x)

