# Tensor 2 Tensor aka Attention is all you need
# Thanks to http://nlp.seas.harvard.edu/2018/04/03/attention.html
from typing import Iterator
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
from torch.autograd import Variable
from tqdm import tqdm
from tgnmt import device, log, TranslationExperiment as Experiment, debug_mode, my_tensor as tensor
from tgnmt.dataprep import BatchIterable, Batch
from tgnmt.utils import log_tensor_sizes


class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many
    other models.
    """

    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(EncoderDecoder, self).__init__()
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
    def make_model(src_vocab, tgt_vocab, N=6,
                   d_model=512, d_ff=2048, h=8, dropout=0.1):
        "Helper: Construct a model from hyperparameters."

        # args for reconstruction of model
        args = {'src_vocab': src_vocab, 'tgt_vocab': tgt_vocab,
                'N': N, 'd_model': d_model, 'd_ff': d_ff, 'h': h,
                'drop_out': dropout}
        c = copy.deepcopy
        attn = MultiHeadedAttention(h, d_model)
        ff = PositionwiseFeedForward(d_model, d_ff, dropout)

        encoder = Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N)
        decoder = Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N)

        src_emb = nn.Sequential(Embeddings(d_model, src_vocab), PositionalEncoding(d_model, dropout))
        tgt_emb = nn.Sequential(Embeddings(d_model, tgt_vocab), PositionalEncoding(d_model, dropout))
        generator = Generator(d_model, tgt_vocab)
        model = EncoderDecoder(encoder, decoder, src_emb, tgt_emb, generator)

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
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
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

    def rate(self, step=None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.factor * (self.model_size ** (-0.5) * min(step ** (-0.5), step * self.warmup ** (-1.5)))

    @staticmethod
    def get_std_opt(model):
        return NoamOpt(model.src_embed[0].d_model, 2, 4000,
                       torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))


class LabelSmoothingOld(nn.Module):
    # NOTE: this one doesnt run on pytorch 0.4.0 and
    "Implement label smoothing. "

    def __init__(self, size, padding_idx, smoothing=0.0):
        super(LabelSmoothingOld, self).__init__()
        self.criterion = nn.KLDivLoss(size_average=False)
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size

    def forward(self, x, target):
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = target.data != self.padding_idx  # Return a tensor of 1 if not padding, 0 if padding
        assert mask.size() == true_dist.size()
        true_dist = torch.mul(true_dist, mask)  # zero out padding
        return self.criterion(x, Variable(true_dist, requires_grad=False))


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
        return loss


class SimpleLossCompute:
    "A simple loss compute and train function."

    def __init__(self, generator, criterion, opt=None):
        self.generator = generator
        self.criterion = criterion
        self.opt = opt

    def __call__(self, x, y, norm):
        x = self.generator(x)
        scores = x.contiguous().view(-1, x.size(-1))
        truth = y.contiguous().view(-1)
        assert norm != 0
        loss = self.criterion(scores, truth) / norm
        loss.backward()
        if self.opt is not None:
            self.opt.step()
            self.opt.optimizer.zero_grad()
        return loss.item() * norm


class Trainer:

    def __init__(self, exp: Experiment = None, model: EncoderDecoder=None, lr=0.0001):
        self.start_epoch = 0
        self.exp = exp
        if model:
            self.model = model
        else:
            args = exp.model_args
            assert args
            log.info(f"Creating model with args: {args}")
            self.model, _ = EncoderDecoder.make_model(**args)
            last_model, last_epoch = self.exp.get_last_saved_model()
            if last_model:
                self.start_epoch = last_epoch + 1
                log.info(f"Resuming training from epoch:{self.start_epoch}, model={last_model}")
                self.model.load_state_dict(torch.load(last_model))
        self.model = self.model.to(device)
        adam_opt = torch.optim.Adam(self.model.parameters(), lr=lr, betas=(0.9, 0.98), eps=1e-9)
        noam_opt = NoamOpt(self.model.src_embed[0].d_model, 2, 4000, adam_opt)

        criterion = LabelSmoothing(size=self.model.tgt_vocab, padding_idx=0, smoothing=0.1)
        self.loss_func = SimpleLossCompute(self.model.generator, criterion, noam_opt)

    def run_epoch(self, data_iter: Iterator[Batch], num_batches=None, print_every=30):
        "Standard Training and Logging Function"
        start = time.time()
        total_tokens = 0
        total_loss = 0.0
        tokens = 0
        for i, batch in tqdm(enumerate(data_iter), total=num_batches):
            num_toks = batch.y_toks
            out = self.model(batch.x_seqs, batch.y_seqs, batch.x_mask, batch.y_mask)
            # skip the BOS token in  batch.y_seqs
            loss = self.loss_func(out, batch.y_seqs_nobos, num_toks)
            total_loss += loss
            total_tokens += num_toks
            tokens += num_toks
            if i + 1 % print_every == 0:
                elapsed = time.time() - start
                log.info(f"Step: {i} Loss: {loss / num_toks:.4f} Tokens per Sec: { tokens / elapsed:.2f}")
                start = time.time()
                tokens = 0
            # force free memory
            del batch
            if debug_mode:
                log_tensor_sizes()

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
        return max_iters-1, loss

    def train(self, num_epochs: int, batch_size: int, **args):
        log.info(f'Going to train for {num_epochs} epochs; batch_size={batch_size}')
        keep_models = args.get('keep_models', 4)  # keep last _ models and delete the old
        if args.get('resume_train'):
            num_epochs += self.start_epoch
        elif num_epochs <= self.start_epoch:
            raise Exception(f'The model was already trained to {self.start_epoch} epochs. '
                            f'Please increase epoch or clear the existing models')
        train_data = BatchIterable(self.exp.train_file, batch_size=batch_size)
        self.model.train()  # Train mode
        for ep in range(self.start_epoch, num_epochs):
            log.info(f"Running epoch {ep+1}")
            try:
                loss = self.run_epoch(train_data)
            except RuntimeError as e:
                if 'out of memory' in str(e).lower():
                    log_tensor_sizes()
                raise e
            log.info(f"Finished epoch {ep+1}")
            self.exp.store_model(ep, self.model.state_dict(), loss, keep=keep_models)
            self.start_epoch += 1


if __name__ == '__main__':
    from tgnmt.dummy import BatchIterable
    V = 14
    criterion = LabelSmoothing(size=V, padding_idx=Batch.pad_value, smoothing=0.1)
    model, _ = EncoderDecoder.make_model(V, V, N=4, d_model=128, d_ff=256, h=4)
    from tgnmt.module.decoder import Decoder
    exp = Experiment("work", config={'model_type': 't2t'}, read_only=True)
    trainer = Trainer(exp=exp, model=model)

    decr = Decoder.new(exp, model)

    assert 2 == Batch.bos_val
    src = tensor([[2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
                  [2, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4]])

    src_lens = tensor(src.size(1))

    def print_res(res):
        for score, seq in res:
            log.info(f'{score:.4f} :: {seq}')

    first_batch = list(iter(BatchIterable(V, 50, 1, reverse=False, batch_first=True)))[0]
    itr, score = trainer.overfit_batch(first_batch, max_iters=100)
    log.info(f"First Batch: {itr} iters with final loss: {score}")

    for epoch in range(15):
        model.train()
        data = BatchIterable(V, 50, 30, reverse=False, batch_first=True)
        loss = trainer.run_epoch(data, num_batches=data.num_batches)
        log.info(f"Epoch {epoch}, training Loss: {loss:.4f}")
        model.eval()
        res = decr.greedy_decode(src, src_lens, max_len=12)
        print_res(res)

