# Transformer aka "Attention is all you need"
# Thanks to http://nlp.seas.harvard.edu/2018/04/03/attention.html
import os
import copy
import math
import time
import inspect
import gc
from abc import ABC
from typing import Callable, Optional, List, Union
import traceback

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from tqdm import tqdm

from rtg import device, log, my_tensor as tensor, TranslationExperiment as Experiment
from rtg.utils import get_my_args
from rtg.dataprep import Batch, BatchIterable
from rtg.module import NMTModel
from rtg.module.trainer import TrainerState, SteppedTrainer
from rtg.module.criterion import Criterion
from torch.optim.optimizer import Optimizer
from dataclasses import dataclass


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

    def __init__(self, size, self_attn, feed_forward, dropout):
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
        #self.generator.proj.weight = self.tgt_embed[0].lut.weight

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

    @classmethod
    def make_model(cls, src_vocab, tgt_vocab, enc_layers=6, dec_layers=6, hid_size=512,
                   ff_size=2048, n_heads=8, dropout=0.1, tied_emb='three-way', activation='relu',
                   exp: Experiment = None):
        raise NotImplementedError


class TransformerNMT(AbstractTransformerNMT):
    """
    A standard Encoder-Decoder Transformer architecture.
    """

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
    def make_model(cls, src_vocab, tgt_vocab, enc_layers=6, dec_layers=6, hid_size=512, ff_size=2048,
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
        encoder = Encoder(EncoderLayer(hid_size, c(attn), c(ff), dropout), enc_layers)

        assert dec_layers > 0
        decoder = Decoder(DecoderLayer(hid_size, c(attn), c(attn), c(ff), dropout), dec_layers)

        src_emb = nn.Sequential(Embeddings(hid_size, src_vocab),
                                PositionalEncoding(hid_size, dropout))
        tgt_emb = nn.Sequential(Embeddings(hid_size, tgt_vocab),
                                PositionalEncoding(hid_size, dropout))
        generator = Generator(hid_size, tgt_vocab)

        model = cls(encoder, decoder, src_emb, tgt_emb, generator)

        if tied_emb:
            model.tie_embeddings(tied_emb)

        model.init_params()
        return model, args


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
        scores = scores.masked_fill(mask == 0, -1e9)
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

    def __call__(self, x_feats, y_seqs, normalizer, train_mode=True, take_step=True):
        x_probs = self.generator(x_feats, score=self.criterion.input_type)  # B x T x D --> B x T x V

        scores = x_probs.contiguous().view(-1, x_probs.size(-1))  # B x T x V --> B.T x V
        truth = y_seqs.contiguous().view(-1)  # B x T --> B.T
        loss = self.criterion(scores, truth).sum() / normalizer

        if train_mode:  # don't do this for validation set
            loss.backward()
            if take_step:
                self.opt.step()
                self.opt.zero_grad()

        return loss.item()


@dataclass
class ChunkedLossCompute(SimpleLossFunction):
    chunk_size: int = 10

    def __call__(self, y_feats, y_seqs, normalizer: Union[int, float],
                 train_mode=True, chunk_size=None, take_step=True):
        chunk_size = chunk_size or self.chunk_size
        assert chunk_size > 0
        total = 0
        _y_feats = y_feats.detach().clone()
        _y_feats.requires_grad = True  # yet collect grads

        for i in range(0, _y_feats.shape[1], chunk_size):
            # grad network is cut here
            chunked_feats = _y_feats[:, i:i + chunk_size]
            chunked_dist = self.generator(chunked_feats, score=self.criterion.input_type)

            chunked_dist = chunked_dist.contiguous().view(-1, chunked_dist.shape[
                -1])  # B x C x V -> B.C x V
            chunked_ys = y_seqs[:, i:i + chunk_size].contiguous().view(-1)  # B x C -> B.C
            loss = self.criterion(chunked_dist, chunked_ys).sum() / normalizer
            total += loss.detach().item()
            if train_mode:
                loss.backward()
        if train_mode:
            out_grad = _y_feats.grad.data
            y_feats.backward(gradient=out_grad)
            if take_step:
                self.opt.step()
                self.opt.zero_grad()

        return total


class MultiGPULossFunction(ChunkedLossCompute):

    def __init__(self, dp_module: nn.DataParallel, criterion: Criterion, opt: Optimizer,
                 chunk_size: int, devices: List, out_device=None):
        super().__init__(None, criterion, opt, chunk_size)
        self.multi_gpu = len(devices) > 1
        assert self.multi_gpu
        self.devices = devices
        self.out_device = out_device if out_device is not None else devices[0]
        assert isinstance(dp_module, nn.DataParallel)
        self.dp_module: nn.DataParallel = dp_module
        self.sct_criteria = nn.parallel.replicate(self.criterion, devices=self.devices)

    def __call__(self, y_feats, y_seqs, normalizer: Union[int, float],
                 train_mode=True, chunk_size=None, take_step=True):

        batch_dim = 0
        assert y_feats.shape[batch_dim] == y_seqs.shape[batch_dim]

        # disconnect y_feats nodes from rest of graph
        _y_feats = y_feats.data.clone().detach()
        _y_feats.requires_grad = True  # even though detached, we still need grads here

        # naming: sct = Scattered  chk = Chunked
        # Scatter is horizontal split (i.e. along batch) ; Chunk is vertical split (ie. along time)
        # Scatter is handled by pytorch's dataparallel utils
        sct_feats = nn.parallel.scatter(_y_feats, target_gpus=self.devices, dim=batch_dim)
        sct_ys = nn.parallel.scatter(y_seqs, target_gpus=self.devices, dim=batch_dim)
        kwargs_tup = [dict(score=self.criterion.input_type) for _ in sct_feats]
        assert len(sct_feats) == len(sct_ys)

        n_scts = len(sct_feats)  # if the batch is smaller than n_gpus; only use a subset

        sct_criteria = self.sct_criteria[:n_scts]
        # use generator from data parallel, because the generator params maybe tied to embeddings
        # TODO: I am not 100% sure if this actually works
        generator = self.dp_module.module.generator
        sct_generators = nn.parallel.replicate(generator, devices=self.devices)[:n_scts]

        chunk_size = chunk_size or self.chunk_size
        assert chunk_size > 0
        seq_len = y_feats.shape[1]  # B x L x D
        assert seq_len == y_seqs.shape[-1]  # B x L
        total_loss = 0

        for i in range(0, seq_len, chunk_size):
            chk_sct_feats = [sf[:, i:i + chunk_size] for sf in sct_feats]

            chk_sct_flat_ys = [sy[:, i:i + chunk_size].contiguous().view(-1) for sy in sct_ys]

            chk_sct_dist = nn.parallel.parallel_apply(sct_generators, chk_sct_feats,
                                                      kwargs_tup=kwargs_tup)
            chk_sct_flt_dist = [chk_dist.contiguous().view(-1, chk_dist.shape[-1]) for chk_dist in
                                chk_sct_dist]
            args_pair = list(zip(chk_sct_flt_dist, chk_sct_flat_ys))
            chk_sct_loss = nn.parallel.parallel_apply(sct_criteria, args_pair)

            # update total loss
            chk_losses = nn.parallel.gather(chk_sct_loss, target_device=self.out_device)
            chk_loss = chk_losses.sum() / normalizer
            total_loss += chk_loss.item()

            if train_mode:
                chk_loss.backward()  # backward for the chunked part

        # back prop all loss through the rest of the network
        if train_mode:
            # back prop the rest of network
            y_feats.backward(gradient=_y_feats.grad.data)
            if take_step:
                self.opt.step()
                self.opt.zero_grad()

        return total_loss


class TransformerTrainer(SteppedTrainer):

    def __init__(self, exp: Experiment,
                 model: Optional[TransformerNMT] = None,
                 optim: str = 'ADAM',
                 model_factory=TransformerNMT.make_model,
                 **optim_args):
        super().__init__(exp, model, model_factory=model_factory, optim=optim, **optim_args)
        generator = self.model.generator
        self.n_gpus = torch.cuda.device_count()
        trainer_args = self.exp.config.get('trainer', {}).get('init_args', {})
        chunk_size = trainer_args.get('chunk_size', 10)
        self.grad_accum_interval = trainer_args.get('grad_accum', 1)
        assert self.grad_accum_interval > 0

        log.info(f"Going to use {self.n_gpus} GPUs; "
                 f" Chunk_size={chunk_size} CUDA_VISIBLE_DEVICES="
                 f"{os.environ.get('CUDA_VISIBLE_DEVICES')}")

        if self.n_gpus > 1:  # Multi GPU mode
            device_ids = list(range(self.n_gpus))
            log.warning("Multi GPU mode <<this feature is not well tested>>")
            self.model = nn.DataParallel(self.model, dim=0, device_ids=device_ids)

            self.loss_func = MultiGPULossFunction(self.model, criterion=self.criterion, opt=self.opt,
                                                  chunk_size=chunk_size, devices=device_ids)
        else:
            self.loss_func = ChunkedLossCompute(generator=generator, criterion=self.criterion,
                                                opt=self.opt, chunk_size=chunk_size)

    def run_valid_epoch(self, data_iter: BatchIterable, dec_bos_cut=False):
        """
        :param data_iter: data iterator
        :param dec_bos_cut: cut first step of input as first step of decoder
        :return: loss value
        """
        start = time.time()
        total_tokens = 0
        total_loss = 0.0
        num_batches = 0
        with tqdm(data_iter, total=data_iter.num_batches,
                  unit='batch', dynamic_ncols=True) as data_bar:
            for i, batch in enumerate(data_bar):
                batch = batch.to(device)
                num_toks = batch.y_toks
                x_seqs = batch.x_seqs
                if dec_bos_cut:
                    bos_step = x_seqs[:, :1]
                    x_seqs = x_seqs[:, 1:]
                else:
                    bos_step = torch.full((len(batch), 1), fill_value=Batch.bos_val,
                                          dtype=torch.long, device=device)

                x_mask = (x_seqs != batch.pad_value).unsqueeze(1)
                y_seqs_with_bos = torch.cat([bos_step, batch.y_seqs], dim=1)
                y_mask = Batch.make_target_mask(y_seqs_with_bos)
                out = self.model(x_seqs, y_seqs_with_bos, x_mask, y_mask)
                # [Batch x Time x D]
                # skip the last time step (the one with EOS as input)
                out = out[:, :-1, :]
                # assumption:  y_seqs has EOS, and not BOS
                loss = self.loss_func(out, batch.y_seqs, num_toks, False)
                total_loss += loss
                total_tokens += num_toks
                num_batches += 1
                elapsed = time.time() - start
                data_bar.set_postfix_str(
                    f'Loss:{loss:.4f}, {int(num_toks / elapsed)}toks/s', refresh=False)
                start = time.time()

        score = total_loss / num_batches
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
              keep_models=10, sort_by='eq_len_rand_batch', log_interval: int = 10, **args):
        """

        :param steps: how many optimizer steps to train (also, means how many batches)
        :param check_point: after how many checkpoints to
        :param batch_size: how many target tokens in batch max ( = max_len * num_sentences)
        :param check_pt_callback: function to call back after checkpt
        :param fine_tune: should the fine tune corpus be used instead of training corpus
        :param dec_bos_cut: copy the first time step of input as decoder's BOS
        :param keep_models: how many checkpts to keep
        :param args: any extra args
        :return:
        """
        log_resources = args.pop('log_resources', False)
        log_embedding = args.pop('log_embedding', False)
        assert log_interval > 0

        # Gradient accumulation
        opt_steps = steps
        batches = steps * self.grad_accum_interval
        start_batch = self.start_step * self.grad_accum_interval
        check_point = check_point * self.grad_accum_interval

        if args:
            # no extra args. let user know if an extra arg is passed
            raise Exception(f" Found extra args: {args}")
        log.info(f'Going to train for {opt_steps} optimizer steps over {batches} batches'
                 f' (from {self.start_step} steps);'
                 f' batch_size={batch_size} toks; sort_by={sort_by};'
                 f' check point size:{check_point}; fine_tune={fine_tune};'
                 f' dec_bos_cut={dec_bos_cut}')
        if self.n_gpus > 1:
            batch_size *= self.n_gpus
            log.info(f"# GPUs = {self.n_gpus}, batch_size is set to {batch_size}")

        if batches <= start_batch:
            raise Exception(f'The model was already trained to {self.start_step} steps. '
                            f'Please increase the steps or clear the existing models')
        train_data = self.exp.get_train_data(batch_size=batch_size, steps=batches - start_batch,
                                             sort_by=sort_by, batch_first=True, fine_tune=fine_tune)
        val_data = self.exp.get_val_data(batch_size, shuffle=False, batch_first=True,
                                         sort_desc=False)

        train_state = TrainerState(self.model, check_point=check_point)
        train_state.train_mode(True)
        unsaved_state = False
        cuda_available = torch.cuda.is_available()
        update_interval = 0
        with tqdm(train_data, initial=start_batch, total=batches, unit='batch',
                  dynamic_ncols=True) as data_bar:
            for batch in data_bar:
                if update_interval == 0:
                    self.model.zero_grad()

                # Prep batch
                batch = batch.to(device)
                num_toks = batch.y_toks
                x_seqs = batch.x_seqs
                if dec_bos_cut:
                    bos_step = x_seqs[:, :1]
                    x_seqs = x_seqs[:, 1:]
                else:
                    bos_step = torch.full((len(batch), 1), fill_value=Batch.bos_val,
                                          dtype=torch.long, device=device)

                # Prep masks
                x_mask = (x_seqs != batch.pad_value).unsqueeze(1)
                y_seqs_with_bos = torch.cat([bos_step, batch.y_seqs], dim=1)
                y_mask = Batch.make_target_mask(y_seqs_with_bos)

                # [Batch x Time x D]
                out = self.model(x_seqs, y_seqs_with_bos, x_mask, y_mask)

                # skip the last time step (the one with EOS as input)
                out = out[:, :-1, :]

                # Trigger optimizer step after gradient accumulation
                opt_step = update_interval == (self.grad_accum_interval-1)

                # assumption:  y_seqs has EOS, and not BOS
                loss = self.loss_func(out, batch.y_seqs, num_toks, train_mode=True, take_step=opt_step)

                # Log
                unsaved_state = True
                if self.opt.curr_step % log_interval == 0:
                    self.tbd.add_scalars('training', {'step_loss': loss,
                                                      'learn_rate': self.opt.curr_lr},
                                         self.opt.curr_step)
                    if log_resources and cuda_available:
                        self._log_resources(batch)
                progress_msg, is_check_pt = train_state.step(num_toks, loss)
                progress_msg += f', LR={self.opt.curr_lr:g}'
                data_bar.set_postfix_str(progress_msg, refresh=False)
                del batch

                # Save checkpoint
                if is_check_pt:
                    train_loss = train_state.reset()
                    train_state.train_mode(False)
                    val_loss = self.run_valid_epoch(val_data, dec_bos_cut=dec_bos_cut)
                    self.make_check_point(train_loss, val_loss=val_loss, keep_models=keep_models,
                                          log_embedding=log_embedding)
                    if check_pt_callback:
                        check_pt_callback(model=self.model,
                                          step=self.opt.curr_step,
                                          train_loss=train_loss)
                    train_state.train_mode(True)
                    unsaved_state = False
                    gc.collect()

                # Track gradient accumulation updates
                update_interval += 1
                if update_interval >= self.grad_accum_interval:
                    update_interval = 0

        # End of training
        if unsaved_state:
            train_loss = train_state.reset()
            train_state.train_mode(False)
            val_loss = self.run_valid_epoch(val_data, dec_bos_cut=dec_bos_cut)
            self.make_check_point(train_loss, val_loss=val_loss, keep_models=keep_models)

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
    from rtg.dummy import DummyExperiment
    vocab_size = 30
    args = {
        'src_vocab': vocab_size,
        'tgt_vocab': vocab_size,
        'enc_layers': 0,
        'dec_layers': 4,
        'hid_size': 64,
        'ff_size': 64,
        'n_heads': 4,
        'activation': 'relu'
    }
    if False:
        for n, p in model.named_parameters():
            print(n, p.shape)

    from rtg.module.decoder import Decoder

    config = {
        'model_type': 'tfmnmt',
        'trainer': {'init_args': {'chunk_size': 2, 'grad_accum': 2}},
        'optim': {
            'args': {
                # "cross_entropy", "smooth_kld", "binary_cross_entropy",
                # "triplet_loss", "smooth_kld_and_triplet_loss"
                # 'criterion': "triplet_loss",
                # 'criterion': "smooth_kld",
                'criterion': "smooth_kld_and_triplet_loss",
                'label_smoothing': 0.1,
                'margin': 0.2,
                'mode': 'dot',
                'neg_sampling': 'hard',
                'neg_region': 0.05,
                'alpha': 1.0
            }
        }
    }

    exp = DummyExperiment("work.tmp.t2t", config=config, read_only=True,
                          vocab_size=vocab_size)
    exp.model_args = args
    trainer = TransformerTrainer(exp=exp, warmup_steps=200, **config['optim']['args'])
    decr = Decoder.new(exp, trainer.model)

    assert 2 == Batch.bos_val
    src = tensor([[4, 5, 6, 7, 8, 9, 10, 11, 12, 13, Batch.eos_val, Batch.pad_value],
                  [13, 12, 11, 10, 9, 8, 7, 6, Batch.eos_val, Batch.pad_value, Batch.pad_value,
                   Batch.pad_value]])
    src_lens = tensor([src.size(1)] * src.size(0))

    def check_pt_callback(**args):
        res = decr.greedy_decode(src, src_lens, max_len=12)
        for score, seq in res:
            log.info(f'{score:.4f} :: {seq}')

    batch_size = 50
    steps = 500
    check_point = 25
    trainer.train(steps=steps, check_point=check_point, batch_size=batch_size,
                  check_pt_callback=check_pt_callback)


if __name__ == '__main__':
    __test_model__()
