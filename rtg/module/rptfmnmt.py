from typing import *

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.autograd import Variable

from rtg import device, log, my_tensor as tensor, TranslationExperiment as Experiment
from rtg.dataprep import Batch, BatchIterable
from rtg.module import NMTModel
from rtg.module.trainer import TrainerState, SteppedTrainer
from torch.optim.optimizer import Optimizer
from dataclasses import dataclass


class CrossMultiHead(nn.Module):
    ""

    def __init__(self, d_model: int, d_inner: int, n_heads: int, max_pos: int):
        super().__init__()
        self.d_model = d_model
        self.d_inner = d_inner
        self.n_heads = n_heads
        self.max_pos = max_pos

        # Compute all heads in one matrix multiplication for efficiency
        self.linear_q = nn.Linear(d_model, d_inner * n_heads, bias=False)
        self.linear_kv = nn.Linear(d_model, 2 * d_inner * n_heads, bias=False)

        # Final projection kernel
        self.linear_z = nn.Linear(n_heads * d_inner, d_model, bias=False)

        # Scaling ops
        self.scale = 1 / (d_inner ** 0.5)

    def forward(self, x: Tensor, kv: Tensor, mask=None):
        batch_size, seq_len = x.shape[:2]

        # 1) Batch_size x Seq_len x d_model :: d_model x h_heads * d_inner
        # -> Batch_size x Seq_len x h_heads x d_inner
        Q = self.linear_q(x).view(batch_size, seq_len, self.n_heads, self.d_inner)
        K, V = (ten.view(batch_size, seq_len, self.n_heads, self.d_inner)
                for ten in torch.chunk(self.linear_kv(kv), 2, dim=-1))

        raw_attention = torch.einsum('bshi,bkhi->bskh', Q, K) * self.scale
        if mask is not None and mask.any().item():
            raw_attention = raw_attention.masked_fill(mask[..., None], -float('inf'))
        attention_score = torch.softmax(raw_attention, dim=2)

        z = torch.einsum('bskh,bkhi->bshi', attention_score, V).contiguous()
        z = self.linear_z(z.view(batch_size, seq_len, self.n_heads * self.d_inner))

        return z


class MultiHeadAttention(nn.Module):
    def __init__(self, d_input: int, d_inner: int, n_heads: int = 4,
                 dropout: float = 0.1, dropatt: float = 0.):
        super().__init__()
        self.d_input = d_input
        self.d_inner = d_inner
        self.n_heads = n_heads

        # this layer applies the linear transformation required
        # for the keys and values for all heads at once for efficiency
        self.linear_kv = nn.Linear(
            d_input,
            (d_inner * n_heads * 2),  # 2 is for keys and values
            bias=False,  # we don't apply bias, making this a simple matrix multiplication
        )
        # for queries (will not be concatenated with memorized states so separate)
        self.linear_q = nn.Linear(
            d_input, d_inner * n_heads,
            bias=False
        )
        # for positional embeddings
        self.linear_p = nn.Linear(
            d_input, d_inner * n_heads,
            bias=False
        )
        self.scale = 1 / (d_inner ** 0.5)  # for scaled dot product attention
        self.dropatt = nn.Dropout(dropatt)
        # we will use this to project back to the input dimension
        self.lout = nn.Linear(d_inner * n_heads, d_input, bias=False)
        self.norm = nn.LayerNorm(d_input)
        self.dropout = nn.Dropout(dropout)

    def _rel_shift(self, x):
        zero_pad = torch.zeros((x.size(0), 1, *x.size()[2:]),
                               device=x.device, dtype=x.dtype)
        return (torch.cat([zero_pad, x], dim=1)
                .view(x.size(1) + 1, x.size(0), *x.size()[2:])[1:]
                .view_as(x))

    def forward(self, input_: torch.FloatTensor,  # (cur_seq, b, d_in)
                pos_embs: torch.FloatTensor,  # (cur_seq + prev_seq, d_in)
                memory: torch.FloatTensor,  # (prev_seq, b, d_in)
                u: torch.FloatTensor,  # (H, d)
                v: torch.FloatTensor,  # (H, d)
                mask: Optional[torch.FloatTensor] = None,
                ):
        """
        pos_embs: we pass the positional embeddings in separately
            because we need to handle relative positions
        input shape: (seq, bs, self.d_input)
        pos_embs shape: (seq + prev_seq, bs, self.d_input)
        output shape: (seq, bs, self.d_input)
        """
        cur_seq = input_.shape[0]  # sequence length of current segment
        prev_seq = memory.shape[0]  # sequence length of previous segment
        H, d = self.n_heads, self.d_inner
        input_with_memory = torch.cat([memory, input_], dim=0)  # concatenate recurrent memory
        # across sequence dimension

        # we will use the following symbols to represent the shape of the tensors
        # cs: current sequence length, b: batch, H: number of heads
        # d: inner dimension, ps: previous sequence length
        # The key and value are now conditioned on the preceding context
        k_tfmd, v_tfmd = \
            torch.chunk(self.linear_kv(input_with_memory), 2, dim=-1)  # (cs + ps, b, H * d)
        q_tfmd = self.linear_q(input_)  # (cs, b, H * d)

        # apply scaled dot product attention
        # look at the following dimensions carefully, since this is the key operation
        # in the Transformer/Transformer XL architecture

        _, bs, _ = q_tfmd.shape
        assert bs == k_tfmd.shape[1]

        # content-based attention term ((a) + (c) in the paper)
        # this is the standard attention term in the original Transformer, except without positional embeddings
        # which are handled separately in the Transformer XL (see below)
        # here, i corresponds to the number of queries = number of current inputs/targets (seq-wise)
        # j corresponds to the number of key/values = number of vectors that we can use to compute the
        # vector for each query
        content_attn = torch.einsum("ibhd,jbhd->ijbh", [
            (q_tfmd.view(cur_seq, bs, H, d) +  # (a)
             u),  # (c): u represents the global (independent of the query)
            # bias towards certain key/values = words
            # Note: maybe this could be a per-attention head parameter?
            k_tfmd.view(cur_seq + prev_seq, bs, H, d)  # There is no positional information to be found here
        ])  # (cs, cs + ps, b, H)

        # position-based attention term ((b) + (d) in the paper)
        # this attention is solely based on the position of the key/values
        # (i.e. it does not take the content of the key/values into account)
        p_tfmd = self.linear_p(pos_embs)  # (cs + ps, b, H * d)
        position_attn = torch.einsum("ibhd,jhd->ijbh", [
            (q_tfmd.view(cur_seq, bs, H, d) +  # (b)
             v),  # (d): v represents the global (independent of the query)
            # bias towards certain positions
            p_tfmd.view(cur_seq + prev_seq, H, d)  # Notice there is not content information
            # regarding keys and values here!
        ])  # (cs, cs + ps, b, H)

        #  Compute positional attention efficiently
        position_attn = self._rel_shift(position_attn)

        # the attention is the sum of content-based and position-based attention
        attn = content_attn + position_attn

        if mask is not None and mask.any().item():
            attn = attn.masked_fill(
                mask[..., None], -float('inf'))
        attn = torch.softmax(attn * self.scale,  # rescale to prevent values from exploding
                             dim=1)  # normalize across the value sequence dimension
        attn = self.dropa(attn)

        attn_weighted_values = (torch.einsum("ijbh,jbhd->ibhd",
                                             [attn,  # (cs, cs + ps, b, H)
                                              v_tfmd.view(cur_seq + prev_seq, bs, H, d),  # (cs + ps, b, H, d)
                                              ])  # (cs, b, H, d)
                                .contiguous()  # we need to change the memory layout to make `view` work
                                .view(cur_seq, bs, H * d))  # (cs, b, H * d)

        # Project back to input dimension and add residual connection
        output = input_ + self.dropo(self.lout(attn_weighted_values))
        output = self.norm(output)
        return output
