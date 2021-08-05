#!/usr/bin/env python
#
# Author: Thamme Gowda [tg (at) isi (dot) edu] 
# Created: 2/28/21
from typing import List, Union

import torch
from rtg.module.criterion import Criterion
from rtg.module.tfmnmt import ChunkedLossCompute, dtorch
from torch import nn as nn
from torch.optim import Optimizer


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
                 train_mode=True, chunk_size=None, take_step=True, get_out=False):

        batch_dim = 0
        assert y_feats.shape[batch_dim] == y_seqs.shape[batch_dim]

        # disconnect y_feats nodes from rest of graph
        _y_feats = y_feats.data.clone().detach()
        _y_feats.requires_grad = True  # even though detached, we still need grads here
        out_device = y_feats.device
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
        if get_out:
            sct_chk_outs = [[] for _ in range(n_scts)]   # [S x [T2 x [B2 x C]]
        # B2 is B / S ... ie. data parallel divided mini-mini batches
        # T2 is T / C  ... ie  T chunked into mini-T
        for i in range(0, seq_len, chunk_size):
            # TODO: remove for loops that runs on CPUs; make it GPU loops
            # [S x [B2 x C x D]]   ; S is scatter in a python list with items of B x C x D tensors;
            #                     each scattered is on a different GPU mem
            chk_sct_feats = [sf[:, i:i + chunk_size] for sf in sct_feats]
            # [S x [ B2 x C ]] -> [S x [B2.D]]
            chk_sct_flat_ys = [sy[:, i:i + chunk_size].contiguous().view(-1) for sy in sct_ys]
            # [S x [B2 x C x D]] --> [S x [B2 x C x V]]
            chk_sct_dist = nn.parallel.parallel_apply(sct_generators, chk_sct_feats,
                                                      kwargs_tup=kwargs_tup)
            if get_out:
                # [S x [B2 x C x V]] --> [S x [B2 x C]]
                sct_outs = [chk_dist.argmax(dim=-1) for chk_dist in chk_sct_dist]
                for i, out in enumerate(sct_outs):
                    sct_chk_outs[i].append(out)

            chk_sct_flt_dist = [chk_dist.contiguous().view(-1, chk_dist.shape[-1]) for chk_dist in
                                chk_sct_dist]
            args_pair = list(zip(chk_sct_flt_dist, chk_sct_flat_ys))
            chk_sct_loss = nn.parallel.parallel_apply(sct_criteria, args_pair)

            # update total loss
            chk_losses = nn.parallel.gather(chk_sct_loss, target_device=out_device)
            chk_loss = chk_losses.sum() / normalizer
            total_loss += chk_loss.item()

            if train_mode:
                dtorch.backward(chk_loss)
                 # backward for the chunked part

        # back prop all loss through the rest of the network
        if train_mode:
            # back prop the rest of network
            y_feats.backward(gradient=_y_feats.grad.data)
            if take_step:
                dtorch.step(self.opt)

        if get_out:
            # [S x [NumChunks x [B/S x ChunkSize]]]
            # first cat [NumChunks x [B/S x ChunkSize] along time dim to make full seq
            sct_outs = [torch.cat(chk_outs, dim=1) for  chk_outs  in sct_chk_outs]  # [S x [B x T]]
            # next cat along the batch, gather from all GPUs to CPU=-1
            outs = nn.parallel.gather(sct_outs, target_device=-1, dim=0)
            return total_loss, outs
        return total_loss