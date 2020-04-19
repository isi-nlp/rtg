#!/usr/bin/env python
#
# Author: Thamme Gowda [tg (at) isi (dot) edu] 
# Created: 2/15/19


import inspect
import copy
import math
import torch
import torch.nn as nn

from rtg.module.tfmnmt import (Encoder, EncoderLayer, PositionwiseFeedForward, PositionalEncoding,
                               Generator, MultiHeadedAttention, Embeddings, AbstractTransformerNMT,
                               TransformerTrainer, SublayerConnection, clones)
from rtg import TranslationExperiment as Experiment, log
from rtg.data.codec import Field

cls_idx = Field.cls_idx
pad_idx = Field.cls_idx


class MEmbeddings(nn.Module):

    def __init__(self, d_model, vocab):
        super().__init__()
        self.lut = nn.Embedding(vocab, d_model, padding_idx=pad_idx)
        self.vocab = vocab
        self.d_model = d_model
        # merge [sent_repr; embedding] --> d_model
        self.merge = nn.Linear(2 * d_model, d_model)
        self.scaler = math.sqrt(self.d_model)

    def forward(self, x):
        # this gets wrapped in Sequential, which takes only one arg
        word_ids, sent_repr = x
        batch, max_len = word_ids.shape
        embs = self.lut(word_ids) * self.scaler

        sent_repr = sent_repr * self.scaler

        embs = embs.view(batch, max_len, self.d_model)
        assert sent_repr.shape == (batch, self.d_model)
        sent_repr = sent_repr.view(batch, 1, self.d_model).expand_as(embs)
        concatd = torch.cat([sent_repr, embs], dim=-1)
        merged = self.merge(concatd)
        return merged


class DecoderBlock(nn.Module):
    """
    A block in decoder that makes use of sentence representation
    TODO: block is a boring name; there gotta be a more creative name for this step
    """

    def __init__(self, d_model, dropout=0.1, mode='add_attn'):
        super().__init__()
        assert mode in ('add_attn', 'cat_attn')
        self.mode = mode
        if mode == 'add_attn':
            self.w1 = nn.Linear(d_model, d_model)
            self.w2 = nn.Linear(d_model, d_model)
        elif mode == 'cat_attn':
            self.w = nn.Linear(d_model + d_model, d_model)
        else:
            raise Exception()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sent_repr):
        # Assumption x:[Batch x SeqLen x ModelDim]
        # sent_repr: x:[Batch x ModelDim] --> [Batch x SeqLen x ModelDim]
        #  for efficiency we expand sent_repr at caller as
        #        sent_repr = sent_repr.unsqueeze(1).expand_as(x)
        #  and assume they are good to concat here

        if self.mode == 'add_attn':
            scores = self.w1(x) + self.w2(sent_repr)
        elif self.mode == 'cat_attn':
            scores = self.w(torch.cat([x, sent_repr], dim=-1))
        else:
            raise Exception()
        weights = scores.sigmoid()
        weights = self.dropout(weights)
        return sent_repr * weights  # element wise scale


class MDecoderLayer(nn.Module):
    "Decoder is made of self-attn, dec-block, and feed forward (defined below)"

    def __init__(self, size, self_attn, dec_block, feed_fwd, dropout):
        super().__init__()
        self.size = size
        self.self_attn = self_attn
        self.dec_block = dec_block
        self.feed_fwd = feed_fwd
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, tgt_mask, sent_repr):
        """ decoder layer: self_attn, dec_block, feedforward"""
        x = self.sublayer[0](x, lambda _x: self.self_attn(_x, _x, _x, tgt_mask))
        x = self.sublayer[1](x, lambda _x: self.dec_block(_x, sent_repr))
        x = self.sublayer[2](x, self.feed_fwd)
        return x


class MDecoder(nn.Module):
    "Generic N layer decoder with masking."

    def __init__(self, layer: MDecoderLayer, n_layers: int):
        super().__init__()
        self.layers = clones(layer, n_layers)
        self.norm = nn.LayerNorm(layer.size)

    def forward(self, x, tgt_mask, sent_repr):
        for layer in self.layers:
            x = layer(x, tgt_mask, sent_repr)
        return self.norm(x)


class MTransformerNMT(AbstractTransformerNMT):

    @property
    def model_type(self):
        return 'mtfmnmt'

    @classmethod
    def make_model(cls, src_vocab, tgt_vocab, n_layers=6, hid_size=512, ff_size=2048,
                   n_heads=8, attn_dropout=0.1, dropout=0.1, activation='relu',
                   tied_emb='three-way', plug_mode='cat_attn', exp: Experiment = None):
        """
        Helper: Construct a model from hyper parameters."
        :return: model, args
        """
        assert plug_mode in {'cat_attn', 'add_attn', 'cat_emb'}
        # get all args for reconstruction at a later phase
        _, _, _, args = inspect.getargvalues(inspect.currentframe())
        for exclusion in ['cls', 'exp']:
            del args[exclusion]  # exclude some args
        # In case you are wondering, why I didnt use **kwargs here:
        #   these args are read from conf file where user can introduce errors, so the parameter
        #   validation and default value assignment is implicitly done by function call for us :)
        log.info(f"making mtfmnmt model: {args}")
        c = copy.deepcopy
        attn = MultiHeadedAttention(n_heads, hid_size, dropout=attn_dropout)
        ff = PositionwiseFeedForward(hid_size, ff_size, dropout, activation=activation)

        enc_layer = EncoderLayer(hid_size, c(attn), c(ff), dropout)
        encoder = Encoder(enc_layer, n_layers)  # clones n times
        src_emb = nn.Sequential(Embeddings(hid_size, src_vocab),
                                PositionalEncoding(hid_size, dropout))

        if plug_mode == 'cat_emb':
            tgt_emb = nn.Sequential(MEmbeddings(hid_size, tgt_vocab),
                                    PositionalEncoding(hid_size, dropout))
            decoder = c(encoder)  # decoder is same as encoder, except embeddings have concat
        else:
            dec_block = DecoderBlock(hid_size, dropout, mode=plug_mode)
            dec_layer = MDecoderLayer(hid_size, c(attn), c(dec_block), c(ff), dropout)
            decoder = MDecoder(dec_layer, n_layers)
            tgt_emb = nn.Sequential(Embeddings(hid_size, tgt_vocab),
                                    PositionalEncoding(hid_size, dropout))

        generator = Generator(hid_size, tgt_vocab)

        model = cls(encoder, decoder, src_emb, tgt_emb, generator)
        if tied_emb:
            model.tie_embeddings(tied_emb)

        model.init_params()
        return model, args

    def forward(self, src, tgt, src_mask, tgt_mask, gen_probs=False, log_probs=True):
        "Take in and process masked src and target sequences."
        assert src.shape[0] == tgt.shape[0]
        sent_repr = self.encode(src, src_mask)
        feats = self.decode(sent_repr, tgt, tgt_mask)
        return self.generator(feats, log_probs=log_probs) if gen_probs else feats

    def encode(self, src, src_mask):
        batch_size = src.shape[0]
        # ADD CLS token
        cls_col = torch.full((batch_size, 1), fill_value=cls_idx, device=src.device,
                             dtype=torch.long)
        src = torch.cat([cls_col, src], dim=1)
        # assuming first col of mask is proper
        src_mask = torch.cat([src_mask[:, :, :1], src_mask], dim=-1)

        embs = self.src_embed(src)
        enc_feats = self.encoder(embs, src_mask)
        sent_repr = enc_feats[:, 0, :]  # CLS token features
        return sent_repr

    def decode(self, sent_repr, tgt, tgt_mask, **extra):
        assert not extra  # no extra stuff;  added to keep the lint happy

        if isinstance(self.decoder, Encoder):
            # Sentence representation gets concatenated to embeddings
            embs = self.tgt_embed((tgt, sent_repr))
            return self.decoder(embs, tgt_mask)
        else:
            # sentence representation goes to decoder layer
            embs = self.tgt_embed(tgt)
            sent_repr = sent_repr.unsqueeze(1).expand_as(embs)
            return self.decoder(embs, tgt_mask, sent_repr)


class MTransformerTrainer(TransformerTrainer):

    def __init__(self, *args, model_factory=MTransformerNMT.make_model, **kwargs):
        super().__init__(*args, model_factory=model_factory, **kwargs)
        assert isinstance(self.model, MTransformerNMT) or \
            (isinstance(self.model, nn.DataParallel) and isinstance(self.model.module, MTransformerNMT))


def __test_model__():
    from rtg.data.dummy import DummyExperiment
    from rtg import my_tensor as tensor

    vocab_size = 24
    args = {
        'src_vocab': vocab_size,
        'tgt_vocab': vocab_size,
        'n_layers': 4,
        'hid_size': 128,
        'ff_size': 256,
        'n_heads': 4
    }

    from rtg.module.decoder import Decoder

    exp = DummyExperiment("work.tmp.mtfmnmt", config={'model_type': 'mtfmnmt'}, read_only=True,
                          vocab_size=vocab_size)
    exp.model_args = args
    trainer = MTransformerTrainer(exp=exp, warmup_steps=200)
    decr = Decoder.new(exp, trainer.model)

    assert 2 == exp.tgt_vocab.bos_idx
    src = tensor([[4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
                  [13, 12, 11, 10, 9, 8, 7, 6, 5, 4]])
    src_lens = tensor([src.size(1)] * src.size(0))

    def check_pt_callback(**args):
        res = decr.greedy_decode(src, src_lens, max_len=12)
        for score, seq in res:
            log.info(f'{score:.4f} :: {seq}')

    batch_size = 50
    steps = 2000
    check_point = 50
    trainer.train(steps=steps, check_point=check_point, batch_size=batch_size,
                  check_pt_callback=check_pt_callback)


if __name__ == '__main__':
    __test_model__()
