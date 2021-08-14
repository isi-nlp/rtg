# Transformer aka "Attention is all you need"
# Thanks to http://nlp.seas.harvard.edu/2018/04/03/attention.html
import copy
import math
import time
import gc
from abc import ABC
from typing import Callable, Optional, Union
import traceback

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.cuda.amp import autocast
from tqdm import tqdm

from rtg import device, log, TranslationExperiment as Experiment
from rtg.utils import get_my_args
from rtg.utils import get_my_args
from rtg.data.dataset import BatchIterable
from rtg.module import NMTModel
from rtg.module.trainer import TrainerState, SteppedTrainer, EarlyStopper
from rtg.module.criterion import Criterion
from torch.optim.optimizer import Optimizer
from dataclasses import dataclass
from sacrebleu import corpus_bleu
from rtg.distrib import DistribTorch


dtorch = DistribTorch.instance()


def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class Generator(nn.Module):
    "Define standard linear + softmax generation step."

    # compile a pool of post processors
    scores = {
        'logits': lambda x, dim=None: x,
        'softmax': F.softmax,
        'log_softmax': F.log_softmax,
        'sigmoid': lambda x, dim=None: x.sigmoid(),
        'embedding': None,
        'identity': None
    }

    def __init__(self, d_model: int, vocab: int):
        super().__init__()
        self.d_model = d_model
        self.vocab = vocab
        self.proj = nn.Linear(d_model, vocab)
        self.warn_msgs = set()

    def forward(self, x, score=None, gen_probs=True, log_probs=True):
        """
        :param x: features or hidden states
        :param score: what scores are do you want in return? Your options are
            'logits' -- scores without any normalization
            'softmax' -- raw probs for multi class
            'log_softmax' -- log probs for multiclass
            'sigmoid' -- for multilabel task
        :param gen_probs: (deprecated, use 'score=logits') False to get logits; default is True
        :param log_probs: (deprecated, use score='log_softmax' or 'softmax').
            False to get raw probs from softmax, True to get probs from log_softmax.
        :return: scores based on choice of score=xxx
        """
        # made this mess to preserve backward compatibility
        if not score:
            score = 'logits'
            if gen_probs:
                score = 'log_softmax' if log_probs else 'softmax'
            warn_msg = f'API deprecated. use "score={score}" attribute.'
            if warn_msg not in self.warn_msgs:  # warn only Once
                self.warn_msgs.add(warn_msg)
                log.warning(warn_msg)
                traceback.print_stack(limit=6)
        assert score in self.scores, f'{self.scores.keys()} supported but given "{score}"'
        if score == 'embedding' or score == 'identity':
            return x
        x = self.proj(x)
        return self.scores[score](x, dim=-1)


class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"

    def __init__(self, size, self_attn: 'MultiHeadedAttention',
                 feed_forward: 'PositionwiseFeedForward', dropout):
        super().__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        "Follow Figure 1 (left) for connections."
        x = self.sublayer[0](x, lambda _x: self.self_attn(_x, _x, _x, mask))
        return self.sublayer[1](x, self.feed_forward)


class Encoder(nn.Module):
    "Core encoder is a stack of N layers"

    def __init__(self, layer: EncoderLayer, N: int):
        super().__init__()
        self.layers = clones(layer, N)
        self.norm = nn.LayerNorm(layer.size)

    def forward(self, x, mask):
        "Pass the input (and mask) through each layer in turn."
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class DecoderLayer(nn.Module):
    "Decoder is made of self-attn, src-attn, and feed forward (defined below)"

    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super().__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask):
        "Follow Figure 1 (right) for connections."
        m = memory
        x = self.sublayer[0](x, lambda _x: self.self_attn(_x, _x, _x, tgt_mask))
        x = self.sublayer[1](x, lambda _x: self.src_attn(_x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)


class Decoder(nn.Module):
    "Generic N layer decoder with masking."

    def __init__(self, layer: DecoderLayer, n_layers: int):
        super().__init__()
        self.layers = clones(layer, n_layers)
        self.norm = nn.LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


class AbstractTransformerNMT(NMTModel, ABC):
    """
    Abstract instance of a standard Encoder-Decoder architecture.
    Base for this and many other models.
    """

    def __init__(self, encoder: Encoder, decoder: Decoder,
                 src_embed, tgt_embed,
                 generator: Optional[Generator], tgt_vocab=None):
        super().__init__()
        self.encoder: Encoder = encoder
        self.decoder: Decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator
        self.tgt_vocab = tgt_vocab if tgt_vocab else generator.vocab

    @property
    def model_dim(self):
        return self.generator.d_model

    @property
    def vocab_size(self):
        return self.tgt_vocab

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)

    def forward(self, src, tgt, src_mask, tgt_mask, gen_probs=False, log_probs=True):
        "Take in and process masked src and target sequences."
        enc_outs = self.encode(src, src_mask)
        feats = self.decode(enc_outs, src_mask, tgt, tgt_mask)
        return self.generator(feats, log_probs=log_probs) if gen_probs else feats

    def init_src_embedding(self, weights):
        log.info("Initializing source embeddings")
        log.info(f"Embedding matrix object ids: "
                 f" src_inp: {id(self.src_embed[0].lut.weight.data)}"
                 f" tgt_inp: {id(self.tgt_embed[0].lut.weight.data)} "
                 f" tgt_out: {id(self.generator.proj.weight.data)}")
        assert weights.shape == self.src_embed[0].lut.weight.shape
        self.src_embed[0].lut.weight.data.copy_(weights.data)
        # self.generator.proj.weight = self.tgt_embed[0].lut.weight

    def init_tgt_embedding(self, weights, input=True, output=True):
        log.info(f"Are embedding tied ? see object ids: "
                 f" src_inp: {id(self.src_embed[0].lut.weight.data)}"
                 f" tgt_inp: {id(self.tgt_embed[0].lut.weight.data)} "
                 f" tgt_out: {id(self.generator.proj.weight.data)}")
        if input:
            log.info(f"Initializing target input embeddings:"
                     f" {weights.shape} ==> {self.tgt_embed[0].lut.weight.shape}")
            assert weights.shape == self.tgt_embed[0].lut.weight.shape
            self.tgt_embed[0].lut.weight.data.copy_(weights.data)
        if output:
            log.info(f"Initializing target output embeddings:"
                     f" {weights.shape} ==> {self.generator.proj.weight.shape}")
            assert weights.shape == self.generator.proj.weight.shape
            self.generator.proj.weight.data.copy_(weights.data)

    def tie_embeddings(self, tie: str):
        assert tie in ('one-way', 'two-way', 'three-way')
        log.info(f"Tying embeddings: {tie}")
        if tie in ('two-way', 'three-way'):
            # src get tied to tgt, so vocab must match
            assert self.src_embed[0].vocab == self.tgt_embed[0].vocab
            # TODO: count doesnt guarantee that the shared BPE was enabled, so check that from conf

        if tie in ('one-way', 'three-way'):
            log.info(f"Tying embeddings: TgtOut == TgtInp")
            self.generator.proj.weight = self.tgt_embed[0].lut.weight
        if tie in ('two-way', 'three-way'):
            log.info(f"Tying embeddings: SrcInp == TgtInp")
            self.src_embed[0].lut.weight = self.tgt_embed[0].lut.weight

    def get_trainable_params(self, include=None, exclude=None):
        if not include and not exclude or include == 'all':
            return super().get_trainable_params()
        if exclude:
            raise Exception("Exclude not supported yet. Please use include")
            # TODO: implement it later when it is really really needed!
        assert isinstance(include, list)
        # a valid example for include
        valid_include = [
            'src_embed', 'tgt_embed', 'generator',
            'encoder:0,1,2,3,4,5',  # encoder:layers
            'decoder:0,1,2,3,4,5'  # decoder:layers
        ]
        param_groups = []
        for sub_name in include:
            if hasattr(self, sub_name):
                log.info(f"Trainable parameters <-- {sub_name}")
                param_groups.extend(getattr(self, sub_name).parameters())
            elif sub_name.startswith('encoder:') or sub_name.startswith('decoder:'):
                # subselect layers
                sub_name, layers = sub_name.split(':')  # encoder;layers_idx

                layers = [int(x) for x in layers.split(',')]
                sub_module = dict(encoder=self.encoder, decoder=self.decoder)[sub_name]
                for layer_idx in layers:
                    log.info(f'Trainable parameters <-- {sub_name}[{layer_idx}] ')
                    layer = sub_module.layers[layer_idx]
                    param_groups.extend(layer.parameters())
                if len(sub_module.layers) - 1 in layers:  # the last layer is trainable, then norm
                    log.info(f'Trainable parameters <-- {sub_name}.norm')
                    param_groups.extend(sub_module.norm.parameters())
            else:
                raise Exception(f'{sub_name} not supported')

        return param_groups

    @classmethod
    def make_model(cls, src_vocab, tgt_vocab, enc_layers=6, dec_layers=6, hid_size=512,
                   ff_size=2048, n_heads=8, dropout=0.1, tied_emb='three-way', activation='relu',
                   exp: Experiment = None):
        raise NotImplementedError()

    def make_generator(cls, *args, **kwargs):
        from .generator import T2TGenerator
        return T2TGenerator(*args, **kwargs)

class TransformerNMT(AbstractTransformerNMT):
    """
    A standard Encoder-Decoder Transformer architecture.
    """
    # Factories; looks a bit complicated, but very useful if child classes want to customize these.
    GeneratorFactory = Generator
    EncoderLayerFactory = EncoderLayer
    DecoderLayerFactory = DecoderLayer
    EncoderFactory = Encoder
    DecoderFactory = Decoder

    def __init__(self, encoder: Encoder, decoder: Decoder,
                 src_embed, tgt_embed,
                 generator: Optional[Generator], tgt_vocab=None):
        super().__init__(encoder=encoder, decoder=decoder,
                         src_embed=src_embed, tgt_embed=tgt_embed,
                         generator=generator, tgt_vocab=tgt_vocab)

    @property
    def model_type(self):
        return 'tfmnmt'

    @classmethod
    def make_model(cls, src_vocab, tgt_vocab, enc_layers=6, dec_layers=6, hid_size=512,
                   ff_size=2048,
                   n_heads=8, attn_bias=True, attn_dropout=0.1, dropout=0.2, activation='relu',
                   tied_emb='three-way', exp: Experiment = None):
        "Helper: Construct a model from hyper parameters."

        # get all args for reconstruction at a later phase
        args = get_my_args(exclusions=['cls', 'exp'])
        assert activation in {'relu', 'elu', 'gelu'}
        log.info(f"Make model, Args={args}")
        c = copy.deepcopy
        attn = MultiHeadedAttention(n_heads, hid_size, dropout=attn_dropout, bias=attn_bias)
        ff = PositionwiseFeedForward(hid_size, ff_size, dropout, activation=activation)

        if enc_layers == 0:
            log.warning("Zero encoder layers!")
        encoder = cls.EncoderFactory(cls.EncoderLayerFactory(hid_size, c(attn), c(ff), dropout),
                                     enc_layers)

        assert dec_layers > 0
        decoder = cls.DecoderFactory(
            cls.DecoderLayerFactory(hid_size, c(attn), c(attn), c(ff), dropout), dec_layers)

        src_emb = nn.Sequential(Embeddings(hid_size, src_vocab),
                                PositionalEncoding(hid_size, dropout))
        tgt_emb = nn.Sequential(Embeddings(hid_size, tgt_vocab),
                                PositionalEncoding(hid_size, dropout))
        generator = cls.GeneratorFactory(hid_size, tgt_vocab)

        model = cls(encoder, decoder, src_emb, tgt_emb, generator)

        if tied_emb:
            model.tie_embeddings(tied_emb)

        model.init_params()
        return model, args

    @classmethod
    def make_trainer(cls, *args, **kwargs):
        return TransformerTrainer(*args, **kwargs)


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size, dropout):
        super().__init__()
        self.norm = nn.LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))


def attention(query, key, value, mask=None, dropout=None):
    """
    Compute 'Scaled Dot Product Attention'
    :param query:
    :param key:
    :param value:
    :param mask:
    :param dropout:
    :return:
    """

    d_k = query.size(-1)
    # Beware: this is a batch multiplier!
    # See https://pytorch.org/docs/stable/torch.html?highlight=matmul#torch.matmul
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    # scores: [BatchSize x Heads x Time=SeqLen x SeqLen ]
    if mask is not None:
        # How masking works:
        # src_mask is [BatchSize x 1=Heads x 1=Time x SeqLen ]  --> used in enc self_attn
        # tgt_mask is [BatchSize x 1=Heads x SeqLen=Time x SeqLen ]
        #               --> used in dec self_attn and dec_to_enc_attn
        # 1=Heads gets broad casted for all the heads
        # 1=Time is not broad casting, since it is used with encoder, we can encode the
        #    whole encoder seqs at once (unlike decoder, which goes at one time step at a time)
        # SeqLen=Time is a magic for the Decoding sequences to only rely on the previous time steps
        #
        # Now, if you got this, take a moment to thank http://nlp.seas.harvard.edu/rush.html
        # for devising this concise code. I needed a lot of time to understand how this code works!
        #
        #scores = scores.masked_fill(mask == 0, -1e9)
        low_val = -2**15 if dtorch.fp16 else -1e9
        scores = scores.masked_fill(mask == 0, low_val)
    p_attn = F.softmax(scores, dim=-1)  # [BatchSize x Heads x Time=SeqLen x SeqLen ]
    if dropout is not None:
        p_attn = dropout(p_attn)
    # Beware: this is a batch multiplier!
    ctx_vals = torch.matmul(p_attn, value)
    return ctx_vals, p_attn


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1, bias=True):
        "Take in model size and number of heads."
        super().__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model, bias=bias), 4)
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


class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."

    def __init__(self, d_model, d_ff, dropout=0.1, activation='relu'):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        activations = dict(relu=F.relu, elu=F.elu)

        if activation == 'gelu':
            activations['gelu'] = F.gelu
            # probably you are using old torch; please upgrade to torch 1.2+

        self.activation = activations[activation]

    def forward(self, x):
        return self.w_2(self.dropout(self.activation(self.w_1(x))))


class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super().__init__()
        self.vocab = vocab
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, dropout, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
        return self.dropout(x)


@dataclass
class SimpleLossFunction:
    """
    A simple loss function that computes the loss using the criterion given
    """
    generator: Generator
    criterion: Criterion
    opt: Optimizer

    def __call__(self, x_feats, y_seqs, normalizer, train_mode=True, take_step=True, get_out=False):
        # B x T x D --> B x T x V
        x_probs = self.generator(x_feats, score=self.criterion.input_type)
        scores = x_probs.contiguous().view(-1, x_probs.size(-1))  # B x T x V --> B.T x V
        truth = y_seqs.contiguous().view(-1)  # B x T --> B.T
        loss = self.criterion(scores, truth).sum() / normalizer

        if train_mode:  # don't do this for validation set
            dtorch.backward(loss)
            if take_step:
                dtorch.step(self.opt)
        result = loss.item()
        if get_out:
            result = (result, x_probs.argmax(dim=-1))
        return result


@dataclass
class ChunkedLossCompute(SimpleLossFunction):
    chunk_size: int = 10

    def __call__(self, y_feats, y_seqs, normalizer: Union[int, float],
                 train_mode=True, chunk_size=None, take_step=True, get_out=False):
        """

        :param y_feats:
        :param y_seqs:
        :param normalizer:
        :param train_mode: Should the gradients be propagated
        :param chunk_size:  Chunk  size along the time dim
        :param take_step: should the optimizer.step() be called
        :param get_out: should the best outputs be returned
        :return: total_loss if get_outs=False (default)
                (total_loss, outputs) if get_out=True
        """
        chunk_size = chunk_size or self.chunk_size
        assert chunk_size > 0
        total = 0
        _y_feats = y_feats.detach().clone()
        _y_feats.requires_grad = True  # yet collect grads
        out_chunks = []
        for i in range(0, _y_feats.shape[1], chunk_size):
            # grad network is cut here
            chunked_feats = _y_feats[:, i:i + chunk_size]
            chunked_dist = self.generator(chunked_feats, score=self.criterion.input_type)
            if get_out:
                top_idxs = chunked_dist.argmax(dim=-1) # # B x C x V -> B x C
                out_chunks.append(top_idxs)
            # B x C x V -> B.C x V
            chunked_dist = chunked_dist.contiguous().view(-1, chunked_dist.shape[-1])
            chunked_ys = y_seqs[:, i:i + chunk_size].contiguous().view(-1)  # B x C -> B.C
            loss = self.criterion(chunked_dist, chunked_ys).sum() / normalizer
            total += loss.detach().item()
            if train_mode:
                dtorch.backward(loss)
        if train_mode:
            out_grad = _y_feats.grad.data
            y_feats.backward(gradient=out_grad)
            if take_step:
                dtorch.step(optimizer=self.opt)
        if get_out:
            outs = torch.cat(out_chunks, dim=1)
            return total, outs
        else:
            return total


class TransformerTrainer(SteppedTrainer):

    def __init__(self, exp: Experiment,
                 model: Optional[TransformerNMT] = None,
                 optim: str = 'ADAM',
                 model_factory=TransformerNMT.make_model,
                 **optim_args):
        super().__init__(exp, model, model_factory=model_factory, optim=optim, **optim_args)
        trainer_args = self.exp.config.get('trainer', {}).get('init_args', {})
        chunk_size = trainer_args.get('chunk_size', -1)
        self.grad_accum_interval = trainer_args.get('grad_accum', 1)
        assert self.grad_accum_interval > 0

        if self.n_gpus > 1:  # Multi GPU mode
            raise Exception(f"Please use: python -m rtg.distrib.launch -G {self.n_gpus} \n "
                            f" or set single GPU by: export CUDA_VISIBLE_DEVICES=0 ")

        generator = self.core_model.generator
        if not chunk_size or chunk_size < 1:
            self.loss_func = SimpleLossFunction(generator=generator, criterion=self.criterion,
                                                opt=self.opt)
        else:
            log.info(f"Using Chunked Loss Generator. chunk_size={chunk_size}")
            self.loss_func = ChunkedLossCompute(generator=generator, criterion=self.criterion,
                                                opt=self.opt, chunk_size=chunk_size)

    def run_valid_epoch(self, data_iter: BatchIterable, dec_bos_cut=False, do_bleu=True):
        """
        :param data_iter: data iterator
        :param dec_bos_cut: cut first step of input as first step of decoder
        :return: loss value
        """
        start = time.time()
        total_tokens = 0
        total_loss = 0.0
        num_batches = 0
        hyps, refs = [], []   # BLEU
        model = self.core_model
        assert not model.training
        with tqdm(data_iter, total=data_iter.num_items,
                  unit='sentence', dynamic_ncols=True) as data_bar:
            # TODO: BLEU1
            for i, batch in enumerate(data_bar):
                if self.n_gpus <= 1:
                    batch = batch.to(device)
                num_toks = batch.y_toks
                x_seqs = batch.x_seqs
                if do_bleu and not bool(batch.y_raw):
                    log.warning("BLEU is not possible; raw sentences are not set to validation batches")
                    do_bleu = False
                if do_bleu:
                    refs.extend(batch.y_raw)

                if dec_bos_cut:
                    bos_step = x_seqs[:, :1]
                    x_seqs = x_seqs[:, 1:]
                else:
                    bos_step = torch.full((len(batch), 1), fill_value=batch.bos_val,
                                          dtype=torch.long, device=batch.y_seqs.device)

                x_mask = (x_seqs != batch.pad_val).unsqueeze(1)
                y_seqs_with_bos = torch.cat([bos_step, batch.y_seqs], dim=1)
                y_mask = batch.make_autoreg_mask(y_seqs_with_bos)
                out = model(x_seqs, y_seqs_with_bos, x_mask, y_mask)

                # [Batch x Time x D]
                # skip the last time step (the one with EOS as input)
                out = out[:, :-1, :]
                # assumption:  y_seqs has EOS, and not BOS
                loss = self.loss_func(out, batch.y_seqs, num_toks, train_mode=False,
                                      get_out=do_bleu)
                if do_bleu:
                    loss, outs = loss
                    outs = outs.tolist()
                    for out in outs:
                        hyp = self.exp.tgt_vocab.decode_ids(out)
                        hyps.append(hyp)

                total_loss += loss
                total_tokens += num_toks
                num_batches += 1
                elapsed = time.time() - start
                data_bar.set_postfix_str(
                    f'Loss:{loss:.4f}, {int(num_toks / elapsed)}toks/s', refresh=False)

                start = time.time()
                data_bar.update(len(batch))

        score = total_loss / num_batches
        if do_bleu:
            # this is non standard BLEU: greedy(beam=1), tokenized with whatever was used for training
            bleu = corpus_bleu(hyps, [refs], tokenize='none', force=True)
            log.info(f'\n {bleu.format()}')
            data = {f'P{i+1}':p for i, p in enumerate(bleu.precisions)}
            data['bleu']=  bleu.score
            self.tbd.add_scalars('validn_greedytokbleu', data, self.opt.curr_step)
        return score

    def overfit_batch(self, batch, max_iters=100, stop_loss=0.01):
        """
        Try to over fit given batch (for testing purpose only, as suggested in
         https://twitter.com/karpathy/status/1013244313327681536 )
        """
        tokens = 0
        loss = float('inf')
        for i in tqdm(range(max_iters), dynamic_ncols=True):
            num_toks = batch.y_toks
            out = self.model(batch.x_seqs, batch.y_seqs, batch.x_mask, batch.y_mask)
            # skip the BOS token in  batch.y_seqs
            loss = self.loss_func(out, batch.y_seqs_nobos, num_toks)
            tokens += num_toks
            if abs(loss) < abs(stop_loss):
                log.info(f"Stopping early at iter {i}.. Loss = {loss:.4f}")
                return i, loss
        return max_iters - 1, loss

    def train(self, steps: int, check_point: int, batch_size: int,
              check_pt_callback: Optional[Callable] = None, fine_tune=False, dec_bos_cut=False,
              keep_models=10, sort_by='eq_len_rand_batch', log_interval: int = 10,
              keep_in_mem=False, early_stop=None, **args):

        """
        :param steps: how many optimizer steps to train (also, means how many batches)
        :param check_point: after how many checkpoints to
        :param batch_size: how many target tokens in batch max ( = max_len * num_sentences)
        :param check_pt_callback: function to call back after checkpt
        :param fine_tune: should the fine tune corpus be used instead of training corpus
        :param dec_bos_cut: copy the first time step of input as decoder's BOS
        :param keep_models: how many checkpts to keep
        :param keep_in_mem: keep training data in memory
        :param early_stop: {patience: N validations, by: loss, enabled: True}
        :param args: any extra args
        :return:
        """
        log_resources = args.pop('log_resources', False)
        log_embedding = args.pop('log_embedding', False)
        split_ratio = args.pop('split_ratio', 0.)
        dynamic_epoch = args.pop('dynamic_epoch', False)
        assert log_interval > 0

        # Gradient accumulation
        opt_steps = steps
        batches = steps * self.grad_accum_interval
        start_batch = self.start_step * self.grad_accum_interval
        check_point = check_point * self.grad_accum_interval
        if isinstance(batch_size, int):
            max_toks, max_sents = batch_size, float('inf')
        else:
            max_toks, max_sents = batch_size
        if args:
            # no extra args. let user know if an extra arg is passed
            raise Exception(f" Found extra args: {args}")
        log.info(f'Going to train for {opt_steps} optimizer steps over {batches} batches'
                 f' (from {self.start_step} steps);'
                 f' batch_size={batch_size} toks; sort_by={sort_by};'
                 f' check point size:{check_point}; fine_tune={fine_tune};'
                 f' dec_bos_cut={dec_bos_cut}')

        distr = DistribTorch.instance()
        if batches <= start_batch:
            raise Exception(f'The model was already trained to {self.start_step} steps. '
                            f'Please increase the steps or clear the existing models')

        train_data = self.exp.get_train_data(
            batch_size=batch_size, steps=batches - start_batch, sort_by=sort_by, batch_first=True,
            fine_tune=fine_tune, keep_in_mem=keep_in_mem, split_ratio=split_ratio, dynamic_epoch=dynamic_epoch
        )
        val_data = None
        if distr.is_global_main:
            val_data = self.exp.get_val_data(batch_size=max_toks, shuffle=False, batch_first=True,
                                             sort_desc=False)

        train_state = TrainerState(self.model, check_point=check_point)
        train_state.train_mode(True)
        unsaved_state = False
        cuda_available = torch.cuda.is_available()

        batch_count = -1
        stopper = None
        early_stopped = False   # or converged
        if early_stop:
            stopper = EarlyStopper(cur_step=self.start_step, **early_stop)

        with tqdm(train_data, initial=start_batch, total=batches, unit='batch',
                  dynamic_ncols=True, disable=not distr.is_global_main) as data_bar:
            for batch in data_bar:
                batch_count += 1
                take_step = (batch_count % self.grad_accum_interval) == 0

               # if update_interval == 0:
               #     self.model.zero_grad()

                #  if not dataparallel, then move
                if self.n_gpus <= 1:
                    batch = batch.to(device)
                num_toks = batch.y_toks
                x_seqs = batch.x_seqs
                if dec_bos_cut:
                    bos_step = x_seqs[:, :1]
                    x_seqs = x_seqs[:, 1:]
                else:
                    bos_step = torch.full((len(batch), 1), fill_value=batch.bos_val,
                                          dtype=torch.long, device=batch.y_seqs.device)

                # Prep masks
                x_mask = (x_seqs != batch.pad_val).unsqueeze(1)
                y_seqs_with_bos = torch.cat([bos_step, batch.y_seqs], dim=1)
                y_mask = batch.make_autoreg_mask(y_seqs_with_bos)

                with autocast(enabled=dtorch.fp16):
                    # [Batch x Time x D]
                    out = self.model(x_seqs, y_seqs_with_bos, x_mask, y_mask)

                    # skip the last time step (the one with EOS as input)
                    out = out[:, :-1, :]

                    # assumption:  y_seqs has EOS, and not BOS
                    loss = self.loss_func(out, batch.y_seqs, num_toks, train_mode=True,
                                          take_step=take_step)

                if stopper and take_step:
                    stopper.step()
                # Log
                unsaved_state = True
                if self.opt.curr_step % log_interval == 0:
                    self.tbd.add_scalars('training', {'step_loss': loss,
                                                      'learn_rate': self.opt.curr_lr},
                                         self.opt.curr_step)
                    if log_resources and cuda_available:
                        self._log_resources(batch)

                progress_msg, is_check_pt = train_state.step(num_toks, loss)
                progress_msg += f', LR={self.opt.curr_lr:0.8f}'
                data_bar.set_postfix_str(progress_msg, refresh=False)
                del batch

                # Save checkpoint
                if is_check_pt:
                    train_loss = train_state.reset()
                    log.info(f"Chkpt Train loss={train_loss}; Runs validation? {distr.is_global_main}")
                    if distr.is_global_main:
                        train_state.train_mode(False)
                        with torch.no_grad():
                            val_loss = self.run_valid_epoch(val_data, dec_bos_cut=dec_bos_cut)
                            self.make_check_point(train_loss, val_loss=val_loss, keep_models=keep_models,
                                                  log_embedding=log_embedding)
                            if check_pt_callback:
                                check_pt_callback(model=self.model,
                                                  step=self.opt.curr_step,
                                                  train_loss=train_loss)
                        train_state.train_mode(True)

                        if stopper:
                            stopper.validation(val_loss)
                            if stopper.is_stop():
                                log.info(f"Stopping at {stopper.cur_step} because {stopper.by}"
                                         f" didnt improve over {stopper.patience} checkpoints")
                                early_stopped = True
                                break
                    unsaved_state = False
                    gc.collect()
                    distr.barrier()

        # End of training
        if unsaved_state and distr.is_global_main:
            train_loss = train_state.reset()
            train_state.train_mode(False)
            val_loss = self.run_valid_epoch(val_data, dec_bos_cut=dec_bos_cut)
            self.make_check_point(train_loss, val_loss=val_loss, keep_models=keep_models)

        distr.barrier()
        return early_stopped

    def _log_resources(self, batch):
        self.tbd.add_scalars('resources_mem',
                             {'mem_allocd': torch.cuda.memory_allocated(device),
                              'mem_cached': torch.cuda.memory_cached(device),
                              'max_mem_allocd': torch.cuda.max_memory_allocated(device),
                              'max_mem_cached': torch.cuda.max_memory_cached(device),
                              'num_y_toks': batch.y_toks,
                              'num_x_toks': batch.x_toks,
                              'num_sentences': len(batch),
                              'max_x_len': batch.x_seqs.shape[1],
                              'max_y_len': batch.y_seqs.shape[1]
                              }, self.opt.curr_step)


def __test_model__():
    model_args = {
        'enc_layers': 0,
        'dec_layers': 4,
        'hid_size': 64,
        'ff_size': 64,
        'n_heads': 4,
        'activation': 'relu'
    }

    # if you are running this in pycharm, please set Working Dir=<rtg repo base dir> for run config
    dir = 'experiments/sample-exp'
    exp = Experiment(work_dir=dir, read_only=True)

    exp.model_type = 'tfmnmt'
    exp.model_args.update(model_args)
    exp.optim_args[1].update(dict(criterion='smooth_kld', warmup_steps=500,
                                  weighing={'gamma': [0.0, 0.5]}))

    trainer = TransformerTrainer(exp=exp, **exp.optim_args[1])
    assert 2 == exp.tgt_vocab.bos_idx
    batch_size = 256
    steps = 2000
    check_point = 200
    trainer.train(steps=steps, check_point=check_point, batch_size=batch_size)


if __name__ == '__main__':
    __test_model__()
