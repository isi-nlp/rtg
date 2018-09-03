# Tensor 2 Tensor aka Attention is all you need
# Thanks to http://nlp.seas.harvard.edu/2018/04/03/attention.html
from typing import Iterator
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
from torch.autograd import Variable
from tqdm import tqdm
from rtg import device, log, TranslationExperiment as Experiment, my_tensor as tensor
from rtg.dataprep import BatchIterable, Batch
from rtg.utils import Optims


class T2TModel(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many
    other models.
    """

    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(T2TModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator
        self.tgt_vocab = generator.vocab

    def forward(self, src, tgt, src_mask, tgt_mask):
        "Take in and process masked src and target sequences."
        return self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask)

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)

    @staticmethod
    def make_model(src_vocab, tgt_vocab, n_layers=4, hid_size=512, ff_size=512, n_heads=4, dropout=0.1):
        "Helper: Construct a model from hyperparameters."

        # args for reconstruction of model
        args = {'src_vocab': src_vocab,
                'tgt_vocab': tgt_vocab,
                'n_layers': n_layers,
                'hid_size': hid_size,
                'ff_size': ff_size,
                'n_heads': n_heads,
                'dropout': dropout
                }
        c = copy.deepcopy
        attn = MultiHeadedAttention(n_heads, hid_size)
        ff = PositionwiseFeedForward(hid_size, ff_size, dropout)

        encoder = Encoder(EncoderLayer(hid_size, c(attn), c(ff), dropout), n_layers)
        decoder = Decoder(DecoderLayer(hid_size, c(attn), c(attn), c(ff), dropout), n_layers)

        src_emb = nn.Sequential(Embeddings(hid_size, src_vocab), PositionalEncoding(hid_size, dropout))
        tgt_emb = nn.Sequential(Embeddings(hid_size, tgt_vocab), PositionalEncoding(hid_size, dropout))
        generator = Generator(hid_size, tgt_vocab)
        model = T2TModel(encoder, decoder, src_emb, tgt_emb, generator)

        # This was important from their code.
        # Initialize parameters with Glorot / fan_avg.
        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        return model, args


class Generator(nn.Module):
    "Define standard linear + softmax generation step."

    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.d_model = d_model
        self.vocab = vocab
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        return F.log_softmax(self.proj(x), dim=-1)


def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class Encoder(nn.Module):
    "Core encoder is a stack of N layers"

    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        "Pass the input (and mask) through each layer in turn."
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)

        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))


class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"

    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        "Follow Figure 1 (left) for connections."
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)


class Decoder(nn.Module):
    "Generic N layer decoder with masking."

    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


class DecoderLayer(nn.Module):
    "Decoder is made of self-attn, src-attn, and feed forward (defined below)"

    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask):
        "Follow Figure 1 (right) for connections."
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)


def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)


class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
        return self.dropout(x)


class NoamOpt:
    "Optim wrapper that implements rate."

    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def zero_grad(self):
        self.optimizer.zero_grad()

    def rate(self, step=None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.factor * (self.model_size ** (-0.5) * min(step ** (-0.5), step * self.warmup ** (-1.5)))

    @staticmethod
    def get_std_opt(model):
        return NoamOpt(model.src_embed[0].d_model, 2, 4000,
                       torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))


class LabelSmoothing(nn.Module):
    def __init__(self, size: int, padding_idx: int, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self._size = size
        assert 0.0 <= smoothing <= 1.0
        self.padding_idx = padding_idx
        self.criterion = nn.KLDivLoss(size_average=False)
        fill_val = smoothing / (size - 2)
        one_hot = torch.full(size=(1, size), fill_value=fill_val, device=device)
        one_hot[0][self.padding_idx] = 0
        self.register_buffer('one_hot', one_hot)
        self.confidence = 1.0 - smoothing

    def forward(self, x, target):
        assert x.size(1) == self._size
        gtruth = target.view(-1)
        tdata = gtruth.data
        mask = torch.nonzero(tdata.eq(self.padding_idx)).squeeze()
        log_likelihood = torch.gather(x.data, 1, tdata.unsqueeze(1))

        smoothed_truth = self.one_hot.repeat(gtruth.size(0), 1)
        smoothed_truth.scatter_(1, tdata.unsqueeze(1), self.confidence)
        if mask.numel() > 0:
            log_likelihood.index_fill_(0, mask, 0)
            smoothed_truth.index_fill_(0, mask, 0)
        loss = self.criterion(x, Variable(smoothed_truth, requires_grad=False))
        # loss is a scalar value (0-dim )
        # but data parallel expects tensors (for gathering along a dim), so doing this
        return loss.unsqueeze(0)


class SimpleLossCompute:
    "A simple loss compute and train function."

    def __init__(self, generator, criterion, opt):
        self.generator = generator
        self.criterion = criterion
        self.opt = opt

    def __call__(self, x, y, norm, train_mode=True):
        x = self.generator(x)
        scores = x.contiguous().view(-1, x.size(-1))
        truth = y.contiguous().view(-1)
        assert norm != 0
        loss = self.criterion(scores, truth).sum() / norm
        if train_mode:  # dont do this for validation set
            loss.backward()
            self.opt.step()
            self.opt.zero_grad()
        return loss.item() * norm


class MultiGPULossFunction(SimpleLossCompute):
    """
    Loss function that uses Multiple GPUs
    TODO: generate outputs in chunks
    """
    def __init__(self, generator, criterion, devices, opt, out_device=None):
        super(MultiGPULossFunction, self).__init__(generator, criterion, opt)
        self.multi_gpu = False
        if len(devices) > 1:
            self.multi_gpu = True
            self.device_ids = devices
            self.out_device = out_device if out_device is not None else devices[0]
            # Send out to different gpus.
            self.criterion = nn.parallel.replicate(criterion, devices=devices)
            self.generator = nn.parallel.replicate(generator, devices=self.device_ids)

    def __call__(self, outs, targets, norm, train_mode=True):
        if not self.multi_gpu:
            # let the parent class deal with this
            return super(MultiGPULossFunction, self).__call__(outs, targets, norm, train_mode)

        batch_dim = 0
        assert outs.shape[batch_dim] == targets.shape[batch_dim]
        sct_outs = nn.parallel.scatter(outs, target_gpus=self.device_ids, dim=batch_dim)
        sct_tgts = nn.parallel.scatter(targets, target_gpus=self.device_ids, dim=batch_dim)
        assert len(sct_outs) == len(sct_tgts)
        sct_generators = self.generator[:len(sct_outs)]
        sct_criteria = self.criterion[:len(sct_outs)]
        sct_preds = nn.parallel.parallel_apply(sct_generators, sct_outs)
        pairs = [(pred.contiguous().view(-1, pred.size(-1)),
                  tgt.contiguous().view(-1)) for pred, tgt in zip(sct_preds, sct_tgts)]
        sct_losses = nn.parallel.parallel_apply(sct_criteria, pairs)
        sent_losses = nn.parallel.gather(sct_losses, target_device=self.out_device, dim=batch_dim)
        total_loss = (sent_losses.sum() / norm)
        total_loss_val = total_loss.item()
        if train_mode:
            total_loss.backward()
            self.opt.step()
            self.opt.zero_grad()
        return total_loss_val * norm


class T2TTrainer:

    def __init__(self, exp: Experiment = None, model: T2TModel = None, optim='ADAM', **optim_args):
        self.start_epoch = 0
        self.exp = exp
        if model:
            self.model = model
        else:
            args = exp.model_args
            assert args
            log.info(f"Creating model with args: {args}")
            self.model, args = T2TModel.make_model(**args)
            exp.model_args = args

            last_model, last_epoch = self.exp.get_last_saved_model()
            if last_model:
                self.start_epoch = last_epoch + 1
                log.info(f"Resuming training from epoch:{self.start_epoch}, model={last_model}")
                self.model.load_state_dict(torch.load(last_model))

        # making optimizer
        optim_args['lr'] = optim_args.get('lr', 0.001)
        optim_args['betas'] = optim_args.get('betas', [0.9, 0.98])
        optim_args['eps'] = optim_args.get('eps', 1e-9)

        generator = self.model.generator
        warm_up_steps = 2000
        smoothing = 0.1
        noam_factor = 2
        criterion = LabelSmoothing(size=generator.vocab, padding_idx=Batch.pad_value, smoothing=smoothing)

        self.model = self.model.to(device)
        device_ids = list(range(torch.cuda.device_count()))
        log.info(f"Going to use {torch.cuda.device_count()} GPUs ; ids:{device_ids}")
        if len(device_ids) > 1:
            # Multi GPU mode
            self.model = nn.DataParallel(self.model, dim=0, device_ids=device_ids)

        inner_opt = Optims[optim].new(self.model.parameters(), **optim_args)
        noam_opt = NoamOpt(generator.d_model, noam_factor, warm_up_steps, inner_opt)
        self.loss_func = MultiGPULossFunction(generator, criterion, devices=device_ids, opt=noam_opt)
        self.exp.optim_args = optim, optim_args
        if not self.exp.read_only:
            self.exp.persist_state()

    def run_epoch(self, data_iter: Iterator[Batch], num_batches=None, print_every=10, train_mode=True):
        """
        :param data_iter: data iterator
        :param num_batches: number of batches in the iterator, None if dont know
        :param print_every: How often the loss progress be updated on progress bar?
        :param train_mode: is it a training or validation mode
        :return:
        """
        start = time.time()
        total_tokens = 0
        total_loss = 0.0
        self.model.train(train_mode)
        with tqdm(data_iter, total=num_batches, unit='batch') as data_bar:
            for i, batch in enumerate(data_bar):
                batch = batch.to(device)
                num_toks = batch.y_toks
                out = self.model(batch.x_seqs, batch.y_seqs, batch.x_mask, batch.y_mask)
                # skip the BOS token in  batch.y_seqs
                loss = self.loss_func(out, batch.y_seqs_nobos, num_toks, train_mode)
                total_loss += loss
                total_tokens += num_toks
                elapsed = time.time() - start
                data_bar.set_postfix_str(f'Loss:{loss / num_toks:.4f}, {int(num_toks / elapsed)}toks/s', refresh=False)
                start = time.time()
                # force free memory
                del batch

        score = total_loss / total_tokens
        return score

    def overfit_batch(self, batch, max_iters=100, stop_loss=0.01):
        """
        Try to over fit given batch (for testing purpose only, as suggested in
         https://twitter.com/karpathy/status/1013244313327681536)
        """
        tokens = 0
        loss = float('inf')
        for i in tqdm(range(max_iters)):
            num_toks = batch.y_toks
            out = self.model(batch.x_seqs, batch.y_seqs, batch.x_mask, batch.y_mask)
            # skip the BOS token in  batch.y_seqs
            loss = self.loss_func(out, batch.y_seqs_nobos, num_toks)
            tokens += num_toks
            if abs(loss) < abs(stop_loss):
                log.info(f"Stopping early at iter {i}.. Loss = {loss:.4f}")
                return i, loss
        return max_iters - 1, loss

    def train(self, num_epochs: int, batch_size: int, **args):
        log.info(f'Going to train for {num_epochs} epochs; batch_size={batch_size}')
        keep_models = args.get('keep_models', 4)  # keep last _ models and delete the old
        if args.get('resume_train'):
            num_epochs += self.start_epoch
        elif num_epochs <= self.start_epoch:
            raise Exception(f'The model was already trained to {self.start_epoch} epochs. '
                            f'Please increase epoch or clear the existing models')
        train_data = BatchIterable(self.exp.train_file, batch_size=batch_size, shuffle=True)
        val_data = BatchIterable(self.exp.valid_file, batch_size=batch_size, shuffle=True)
        losses = []
        self.model.train()  # Train mode
        for ep in range(self.start_epoch, num_epochs):
            log.info(f"Running epoch:: {ep}")
            train_loss = self.run_epoch(train_data, num_batches=train_data.num_batches, train_mode=True)
            val_loss = self.run_epoch(val_data, num_batches=val_data.num_batches, train_mode=False)
            log.info(f"Finished epoch {ep}. Training Loss {train_loss}, Validation Loss:{val_loss}")

            # Unwrap model state from DataParallel and persist
            state = (self.model.module if isinstance(self.model, nn.DataParallel) else self.model)
            self.exp.store_model(ep, state.state_dict(), train_score=train_loss,
                                 val_score=val_loss, keep=keep_models)
            self.start_epoch += 1
            losses.append((ep, train_loss, val_loss))
        summary = '\n'.join(f'{ep:02}\t{tl:.4f}\t{vl:.4f}' for ep, tl, vl in losses)
        log.info(f"==Summary==:\nEpoch\t TrainLoss \t ValidnLoss \n {summary}")


def __test_model__():
    from rtg.dummy import BatchIterable

    vocab_size = 14
    model, _ = T2TModel.make_model(vocab_size, vocab_size, n_layers=4, hid_size=128, ff_size=256, n_heads=4)
    from rtg.module.decoder import Decoder

    exp = Experiment("work", config={'model_type': 't2t'}, read_only=True)
    trainer = T2TTrainer(exp=exp, model=model)

    decr = Decoder.new(exp, model)

    assert 2 == Batch.bos_val
    src = tensor([[2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
                  [2, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4]])
    src_lens = tensor([src.size(1)] * src.size(0))

    def print_res(res):
        for score, seq in res:
            log.info(f'{score:.4f} :: {seq}')

    val_data = list(BatchIterable(vocab_size, 50, 5, reverse=False, batch_first=True))
    for epoch in range(50):
        model.train()
        train_data = BatchIterable(vocab_size, 30, 20, reverse=False, batch_first=True)
        train_loss = trainer.run_epoch(train_data, num_batches=train_data.num_batches, train_mode=True)
        val_loss = trainer.run_epoch(val_data, num_batches=len(val_data), train_mode=False)
        log.info(f"Epoch {epoch}, training Loss: {train_loss:.4f} \t validation loss:{val_loss:.4f}")
        model.eval()
        res = decr.greedy_decode(src, src_lens, max_len=12)
        print_res(res)


if __name__ == '__main__':
    __test_model__()
