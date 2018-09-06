import torch
from torch import nn
import time

import torch.nn.functional as F

from rtg import my_tensor as tensor, device
from rtg.dataprep import PAD_TOK_IDX, BOS_TOK_IDX, Batch, BatchIterable
from rtg import log, TranslationExperiment as Experiment
from rtg.utils import Optims
from typing import Optional, Any, List, Mapping, Dict, Union
from tqdm import tqdm
import random


class Embedder(nn.Embedding):
    """
    This module takes words (word IDs, not the text ) and creates vectors.
    For the inverse operation see  `Generator` module
    """

    def __init__(self, name: str, vocab_size: int, emb_size: int,
                 weights: Optional[torch.Tensor] = None):
        self.name = name
        self.vocab_size = vocab_size
        self.emb_size = emb_size
        super(Embedder, self).__init__(self.vocab_size, self.emb_size, padding_idx=PAD_TOK_IDX,
                                       _weight=weights)


class Generator(nn.Module):
    """
    This module takes vectors and produces word ids.
    Note: In theory, this is not inverse of `Embedder`, however in practice it is an approximate
    inverse operation of `Embedder`
    """

    def __init__(self, name: str, vec_size: int, vocab_size: int):
        super(Generator, self).__init__()
        self.name = name
        self.vec_size = vec_size
        self.vocab_size = vocab_size
        self.proj = nn.Linear(self.vec_size, self.vocab_size)

    def forward(self, x):
        return F.log_softmax(self.proj(x), dim=-1)


class SeqEncoder(nn.Module):

    def __init__(self, embedder: Embedder, out_size: int, n_layers: int,
                 bidirectional: bool = True):
        super().__init__()
        self.emb = embedder
        self.emb_size = self.emb.emb_size
        self.out_size = out_size
        self.n_layers = n_layers
        self.bidirectional = bidirectional

        out_size = self.out_size
        if self.bidirectional:
            assert self.out_size % 2 == 0
            out_size = out_size // 2
        self.rnn_node = nn.LSTM(self.emb_size, out_size, num_layers=self.n_layers,
                                bidirectional=self.bidirectional, batch_first=True)

    def forward(self, input_seqs: torch.Tensor, input_lengths, hidden=None, pre_embedded=False):
        assert len(input_seqs) == len(input_lengths)
        if pre_embedded:
            embedded = input_seqs
            batch_size, seq_len, emb_size = input_seqs.shape
            assert emb_size == self.emb_size
        else:
            batch_size, seq_len = input_seqs.shape
            embedded = self.emb(input_seqs).view(batch_size, seq_len, self.emb_size)

        packed = nn.utils.rnn.pack_padded_sequence(embedded, input_lengths, batch_first=True)
        outputs, hidden = self.rnn_node(packed, hidden)
        outputs, output_lengths = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True,
                                                                   padding_value=PAD_TOK_IDX)
        # Sum bidirectional outputs
        # outputs = outputs[:, :, :self.hid_size] + outputs[:, :, self.hid_size:]
        return outputs, self.to_dec_state(hidden)

    def to_dec_state(self, enc_state):
        if self.bidirectional:
            # h_t, c_t = enc_state
            batch_size = enc_state[0].shape[1]
            # [num_layers * 2, batch_size, hid // 2]
            #    -> [num_layers, 2, batch_size, hid // 2]
            #    -> [num_layers, batch_size, hid]
            #
            return [
                hc.view(self.n_layers, 2, batch_size, self.out_size // 2)
                    .view(self.n_layers, batch_size, self.out_size) for hc in enc_state]
        return enc_state


class SeqDecoder(nn.Module):

    def __init__(self, prev_emb_node: Embedder, generator: Generator, n_layers: int):
        super(SeqDecoder, self).__init__()
        self.prev_emb = prev_emb_node
        self.generator = generator
        self.n_layers = n_layers
        self.emb_size = self.prev_emb.emb_size
        self.hid_size = self.generator.vec_size
        self.rnn_node = nn.LSTM(self.emb_size, self.hid_size,
                                num_layers=self.n_layers, bidirectional=False, batch_first=True)

    def forward(self, enc_outs, prev_out, last_hidden, gen_probs=True):
        # Note: we run this one step at a time

        # Get the embedding of the current input word (last output word)
        batch_size = prev_out.size(0)
        assert len(enc_outs) == batch_size
        # S=B x 1 x N
        embedded = self.prev_emb(prev_out).view(batch_size, 1, self.prev_emb.emb_size)
        # Get current hidden state from input word and last hidden state
        rnn_output, hidden = self.rnn_node(embedded, last_hidden)

        # [B x N ] <- [B x S=1 x N]
        rnn_output = rnn_output.squeeze(1)

        if gen_probs:
            # Finally predict next token
            next_word_distr = self.generator(rnn_output)
            # Return final output, hidden state, and attention weights (for visualization)
            return next_word_distr, hidden, None
        else:
            return rnn_output, hidden, None


class GeneralAttn(nn.Module):
    """
    Attention model
    """

    def __init__(self, hid_size):
        super(GeneralAttn, self).__init__()
        self.inp_size = hid_size
        self.out_size = hid_size
        self.attn = nn.Linear(self.inp_size, self.out_size)

    def forward(self, this_rnn_out, encoder_outs):
        # hidden      : [B, D]
        # encoder_out : [B, S, D]
        #    [B, D] --> [B, S, D] ;; repeat hidden sequence_len times
        this_run_out = this_rnn_out.unsqueeze(1).expand_as(encoder_outs)
        #  A batched dot product implementation using element wise product followed by sum
        #    [B, S]  <-- [B, S, D]
        # element wise multiply, then sum along the last dim (i.e. model_dim)
        weights = (encoder_outs * this_run_out).sum(dim=-1)

        # Normalize energies to weights in range 0 to 1
        return F.softmax(weights, dim=1)


class AttnSeqDecoder(SeqDecoder):
    def __init__(self, prev_emb_node: Embedder, generator: Generator, n_layers: int):
        super(AttnSeqDecoder, self).__init__(prev_emb_node, generator, n_layers)
        self.attn = GeneralAttn(self.hid_size)
        self.merge = nn.Linear(self.hid_size + self.attn.out_size, self.hid_size)

    def forward(self, enc_outs, prev_out, last_hidden, gen_probs=True):
        # Note: we run this one step at a time

        # Get the embedding of the current input word (last output word)
        batch_size = prev_out.size(0)
        embedded = self.prev_emb(prev_out)
        embedded = embedded.view(batch_size, 1, self.prev_emb.emb_size)

        # Get current hidden state from input word and last hidden state
        rnn_output, hidden = self.rnn_node(embedded, last_hidden)
        # [B x N ] <- [B x S=1 x  N]
        rnn_output = rnn_output.squeeze(1)

        # Calculate attention from current RNN state and all encoder outputs;
        # apply to encoder outputs to get weighted average
        attn_weights = self.attn(rnn_output, enc_outs)  # B x S
        #   attn_weights : B x S     --> B x 1 x S
        #   enc_outs     : B x S x N
        # Batch multiply : [B x 1 x S] [B x S x N] --> [B x 1 x N]
        context = attn_weights.unsqueeze(1).bmm(enc_outs)

        # Attentional vector using the RNN hidden state and context vector
        # concatenated together (Luong eq. 5)
        # rnn_output = rnn_output.squeeze(0)  # S=1 x B x N -> B x N
        context = context.squeeze(1)  # B x S=1 x N -> B x N
        concat_input = torch.cat((rnn_output, context), 1)
        concat_output = torch.tanh(self.merge(concat_input))

        if gen_probs:
            # predict next token
            output_probs = self.generator(concat_output)
            # Return final output, hidden state, and attention weights (for visualization)
            return output_probs, hidden, attn_weights
        else:
            return concat_output, hidden, attn_weights


class Seq2SeqBridge(nn.Module):
    """Vector to Vector (a slightly different setup than seq2seq
    starts with a decoder, then an encoder (short-circuits (or skips) embedder and generator)
    """

    def __init__(self, dec: SeqDecoder, enc: SeqEncoder):
        super().__init__()
        self.dec = dec
        self.enc = enc

    def forward(self, enc_outs, enc_hids, max_len):
        batch_size = len(enc_outs)
        assert batch_size == enc_hids[0].shape[1] == enc_hids[1].shape[1]

        dec_inps = tensor([[BOS_TOK_IDX]] * batch_size, dtype=torch.long)
        dec_hids = enc_hids
        result = torch.zeros((batch_size, max_len, self.dec.hid_size), device=device)
        for t in range(max_len):
            dec_outs, dec_hids, _ = self.dec(enc_outs, dec_inps, dec_hids, gen_probs=False)
            result[:, t, :] = dec_outs

        # TODO: check how hidden state flows
        enc_outs, enc_hids = self.enc(result, [max_len] * batch_size, pre_embedded=True)
        return enc_outs, enc_hids


class Seq2Seq(nn.Module):

    def __init__(self, enc: SeqEncoder, dec: SeqDecoder, bridge: Seq2SeqBridge = None):
        super(Seq2Seq, self).__init__()
        self.enc = enc
        self.dec = dec
        # since no linear projects, all sizes must be same
        assert aeq(enc.out_size, enc.emb_size, dec.hid_size, dec.emb_size)
        self.model_dim = self.enc.out_size
        self.bridge = bridge

    def encode(self, x_seqs, x_lens, hids=None, max_y_len=256):
        enc_outs, enc_hids = self.enc(x_seqs, x_lens, hids)
        if self.bridge:
            # used in BiNMT to make a cycle, such as Enc1 -> [[Dec2 -> Enc2]] -> Dec2
            enc_outs, enc_hids = self.bridge(enc_outs, enc_hids, max_y_len)
        return enc_outs, enc_hids

    def forward(self, batch: Batch):
        assert batch.batch_first
        batch_size = len(batch)
        enc_outs, enc_hids = self.encode(batch.x_seqs, batch.x_len, hids=None,
                                         max_y_len=batch.max_y_len)

        dec_inps = tensor([[BOS_TOK_IDX]] * batch_size, dtype=torch.long)
        dec_hids = enc_hids
        """
        # extract vector at given last stamp (as per the seq length)
        t_dim = 1
        lastt_idx = (batch.x_len - 1).view(-1, 1).expand(-1, self.enc.out_size).unsqueeze(t_dim)
        lastt_out = enc_outs.gather(dim=t_dim, index=lastt_idx).squeeze(t_dim)
        lastt_out = lastt_out.expand(self.dec.n_layers, batch_size, self.dec.generator.vec_size)
        dec_hids = (lastt_out, lastt_out)   # copy enc output to h and c of LSTM
        """
        outp_probs = torch.zeros((batch.max_y_len - 1, batch_size), device=device)

        for t in range(1, batch.max_y_len):
            word_probs, dec_hids, _ = self.dec(enc_outs, dec_inps, dec_hids)

            # expected output;; log probability for these indices should be high
            expct_word_idx = batch.y_seqs[:, t].view(batch_size, 1)
            expct_word_log_probs = word_probs.gather(dim=1, index=expct_word_idx)
            outp_probs[t - 1] = expct_word_log_probs.squeeze()

            # Randomly switch between gold and the prediction next word
            if random.choice((False, True)):
                dec_inps = expct_word_idx  # Next input is current target
            else:
                pred_word_idx = word_probs.argmax(dim=1)
                dec_inps = pred_word_idx.view(batch_size, 1)
        return outp_probs.t()

    @staticmethod
    def make_model(src_lang, tgt_lang, src_vocab: int, tgt_vocab: int, emb_size: int = 300,
                   hid_size: int = 300, n_layers: int = 2):
        args = {
            'src_lang': src_lang,
            'tgt_lang': tgt_lang,
            'src_vocab': src_vocab,
            'tgt_vocab': tgt_vocab,
            'emb_size': emb_size,
            'hid_size': hid_size,
            'n_layers': n_layers
        }
        src_embedder = Embedder(src_lang, src_vocab, emb_size)
        tgt_embedder = Embedder(tgt_lang, tgt_vocab, emb_size)
        tgt_generator = Generator(tgt_lang, vec_size=hid_size, vocab_size=tgt_vocab)
        enc = SeqEncoder(src_embedder, hid_size, n_layers=n_layers, bidirectional=True)
        # dec = SeqDecoder(tgt_embedder, tgt_generator, n_layers=n_layers)
        dec = AttnSeqDecoder(tgt_embedder, tgt_generator, n_layers=n_layers)

        model = Seq2Seq(enc, dec)
        # Initialize parameters with Glorot / fan_avg.
        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        return model, args


def aeq(*items):
    for i in items[1:]:
        if items[0] != i:
            return False
    return True


class BiNMT(nn.Module):
    """
    Bi directional NMT that can be trained with monolingual data
    """

    def __init__(self, enc1: SeqEncoder, dec1: SeqDecoder, enc2: SeqEncoder, dec2: SeqDecoder):
        super().__init__()
        self.enc1, self.dec1 = enc1, dec1
        self.enc2, self.dec2 = enc2, dec2

        # check that sizes are compatible
        # since no linear projects at the moment, all sizes must be same
        assert aeq(enc1.out_size, enc2.out_size, dec1.emb_size, dec2.emb_size,
                   dec1.hid_size, dec2.hid_size)
        self.model_dim = enc1.out_size

        self.paths = {
            'E1D1': Seq2Seq(enc1, dec1),  # ENC1 --> DEC1
            'E2D2': Seq2Seq(enc2, dec2),  # ENC2 --> DEC2
            # ENC1 --> DEC2 --> ENC2 --> DEC1
            'E1D2E2D1': Seq2Seq(enc1, dec1, bridge=Seq2SeqBridge(dec2, enc2)),
            # ENC2 --> DEC1 --> ENC1 --> DEC2
            'E2D1E1D2': Seq2Seq(enc2, dec2, bridge=Seq2SeqBridge(dec1, enc1))
        }
        # TODO: parallel data when available (semi supervised)
        # 1. ENC1 --> DEC2
        # 2. ENC2 --> DEC1

    def forward(self, batch: Batch, path: str):
        if path not in self.paths:
            raise Exception(f'path={path} is unsupported. Valid options:{self.paths.keys()}')
        return self.paths[path](batch)

    @staticmethod
    def make_model(src_lang, tgt_lang, src_vocab: int, tgt_vocab: int, emb_size: int = 300,
                   hid_size: int = 300, n_layers: int = 2):
        args = {
            'src_lang': src_lang,
            'tgt_lang': tgt_lang,
            'src_vocab': src_vocab,
            'tgt_vocab': tgt_vocab,
            'emb_size': emb_size,
            'hid_size': hid_size,
            'n_layers': n_layers
        }
        src_embedder = Embedder(src_lang, src_vocab, emb_size)
        tgt_embedder = Embedder(tgt_lang, tgt_vocab, emb_size)

        src_generator = Generator(src_lang, vec_size=hid_size, vocab_size=src_vocab)
        tgt_generator = Generator(tgt_lang, vec_size=hid_size, vocab_size=tgt_vocab)

        src_enc = SeqEncoder(src_embedder, hid_size, n_layers=n_layers, bidirectional=True)
        tgt_enc = SeqEncoder(tgt_embedder, hid_size, n_layers=n_layers, bidirectional=True)

        # dec = SeqDecoder(tgt_embedder, tgt_generator, n_layers=n_layers)
        src_dec = AttnSeqDecoder(src_embedder, src_generator, n_layers=n_layers)
        tgt_dec = AttnSeqDecoder(tgt_embedder, tgt_generator, n_layers=n_layers)

        model = BiNMT(src_enc, src_dec, tgt_enc, tgt_dec)
        # Initialize parameters with Glorot / fan_avg.
        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        return model, args


class NoamOpt:
    """Optim wrapper that implements rate."""

    # taken from Tensor2Tensor/Transformer model. Thanks to Alexander Rush of HarvardNLP

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
        return rate

    def zero_grad(self):
        self.optimizer.zero_grad()

    def rate(self, step=None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.factor * (
                self.model_size ** (-0.5) * min(step ** (-0.5), step * self.warmup ** (-1.5)))

    @staticmethod
    def get_std_opt(model):
        return NoamOpt(model.src_embed[0].d_model, 2, 4000,
                       torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))


class BaseTrainer:

    def __init__(self, exp: Experiment, model, optim='ADAM', maybe_restore=True,
                 model_factory=None, **optim_args):
        self.exp = exp
        self.start_epoch = 0
        if model is None:
            model, exp.model_args = model_factory(**exp.model_args)
        if maybe_restore:
            last_check_pt, last_epoch = self.exp.get_last_saved_model()
            if last_check_pt:
                log.info(f"Resuming training from epoch:{self.start_epoch}, model={last_check_pt}")
                self.start_epoch = last_epoch + 1
                model.load_state_dict(torch.load(last_check_pt))

        if torch.cuda.device_count() > 1:
            raise RuntimeError('Please export CUDA_VISIBLE_DEVICES to a single GPU id')
        log.info(f"Moving model to device = {device}")
        self.model = model.to(device=device)
        self.model.train()
        warmup = optim_args.pop('warmup_steps', 2000)
        optim_args['lr'] = optim_args.get('lr', 0.001)
        optimizer = Optims[optim].new(self.model.parameters(), **optim_args)
        self.optimizer = NoamOpt(model.model_dim * 2, 2, warmup, optimizer)
        optim_args['warmup_steps'] = warmup
        self.exp.optim_args = optim, optim_args
        if not exp.read_only:
            self.exp.persist_state()

    @staticmethod
    def sequence_mask(lengths, max_len):
        batch_size = lengths.size(0)
        # create a row [0, 1, ... s] and duplicate this row batch_size times --> [B, S]
        seq_range_expand = torch.arange(0, max_len, dtype=torch.long,
                                        device=device).expand(batch_size, max_len)
        # make lengths vectors to [B x 1] and duplicate columns to [B, S]
        seq_length_expand = lengths.unsqueeze(1).expand_as(seq_range_expand)
        return seq_range_expand < seq_length_expand  # 0 if padding, 1 otherwise

    def overfit_batch(self, batch, max_iters=200):
        log.info("Trying to overfit a batch")
        losses = []
        for i in range(max_iters):
            batch.to(device)
            self.model.zero_grad()
            outp_log_probs = self.model(batch)
            tok_mask = self.sequence_mask(batch.y_len, batch.max_y_len - 1)
            per_tok_loss = -outp_log_probs
            loss = (per_tok_loss * tok_mask.float()).sum().float() / batch.y_toks
            loss_value = loss.item()
            losses.append(loss_value)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            if i % 4 == 0:
                log.info(f"{i} :: {loss_value:.6f}")
            if len(losses) > 5 and sum(x * x for x in losses[-5:]) == 0.0:
                log.info("Converged...")
                break


class Seq2SeqTrainer(BaseTrainer):

    def __init__(self, exp: Experiment, model=None, optim='ADAM', **optim_args):
        assert exp.model_type == 'seq2seq'
        super().__init__(exp, model, optim=optim, model_factory=Seq2Seq.make_model, **optim_args)

    def train(self, num_epochs: int, batch_size: int, **args):
        log.info(f'Going to train for {num_epochs} epochs; batch_size={batch_size}')

        train_data = BatchIterable(self.exp.train_file, batch_size=batch_size, batch_first=True,
                                   shuffle=True)
        val_data = BatchIterable(self.exp.valid_file, batch_size=batch_size, batch_first=True,
                                 shuffle=False, copy_xy=True)
        keep_models = args.pop('keep_models', 4)
        if args.pop('resume_train'):
            num_epochs += self.start_epoch
        elif num_epochs <= self.start_epoch:
            raise Exception(f'The model was already trained to {self.start_epoch} epochs. '
                            f'Please increase epoch or clear the existing models')
        losses = []
        for ep in range(self.start_epoch, num_epochs):
            train_loss = self.run_epoch(train_data, train_mode=True)
            log.info(f'Epoch {ep+1} complete.. Training loss in this epoch {train_loss}...')
            val_loss = self.run_epoch(val_data, train_mode=False)
            log.info(f'Validation of {ep+1} complete.. Validation loss in this epoch {val_loss}...')
            losses.append((ep, train_loss, val_loss))
            if keep_models > 0:
                self.exp.store_model(epoch=ep, model=self.model.state_dict(),
                                     train_score=train_loss,
                                     val_score=val_loss, keep=keep_models)
        summary = '\n'.join(f'{ep:02}\t{tl:.4f}\t{vl:.4f}' for ep, tl, vl in losses)
        log.info(f"==Summary==:\nEpoch\t TrainLoss \t ValidnLoss \n {summary}")

    def run_epoch(self, data_iter, num_batches=None, train_mode=True):
        """
        run a pass over data set
        :param data_iter: batched data set
        :param num_batches: number of batches in the dataset (for tqdm progress bar), None if unknown
        :param train_mode: is it training mode (False if validation mode)
        :return: total loss
        """
        tot_loss = 0.0
        start = time.time()
        self.model.train(train_mode)
        with tqdm(data_iter, total=num_batches, unit='batch') as data_bar:
            for i, batch in enumerate(data_bar):
                batch = batch.to(device)
                # Step clear gradients
                self.model.zero_grad()
                # Step Run forward pass.
                outp_log_probs = self.model(batch)

                tok_mask = self.sequence_mask(batch.y_len, batch.max_y_len - 1)
                per_tok_loss = -outp_log_probs
                loss = (per_tok_loss * tok_mask.float()).sum().float() / batch.y_toks
                tot_loss += loss.item()
                learn_rate = ""
                if train_mode:
                    loss.backward()
                    learn_rate = self.optimizer.step()
                    self.optimizer.zero_grad()
                    learn_rate = f'LR={learn_rate:g}'
                elapsed = time.time() - start
                bar_msg = f'Loss:{loss:.4f}, {int(batch.y_toks/elapsed)}toks/s {learn_rate}'
                data_bar.set_postfix_str(bar_msg, refresh=False)
                del batch
        return tot_loss


class BiNmtTrainer(BaseTrainer):

    def __init__(self, exp: Experiment, model=None, optim='ADAM', **optim_args):
        assert exp.model_type == 'binmt'
        super().__init__(exp, model, optim=optim, model_factory=BiNMT.make_model, **optim_args)

    def train(self, num_epochs: int, batch_size: int, **args):
        log.info(f'Going to train for {num_epochs} epochs; batch_size={batch_size}')

        train_data = BatchIterable(self.exp.mono_train_src, batch_size=batch_size, batch_first=True,
                                   shuffle=True, copy_xy=True)
        val_data = BatchIterable(self.exp.mono_valid_src, batch_size=batch_size, batch_first=True,
                                 shuffle=False, copy_xy=True)
        keep_models = args.pop('keep_models', 4)
        if args.pop('resume_train'):
            num_epochs += self.start_epoch
        elif num_epochs <= self.start_epoch:
            raise Exception(f'The model was already trained to {self.start_epoch} epochs. '
                            f'Please increase epoch or clear the existing models')
        losses = []
        for ep in range(self.start_epoch, num_epochs):
            train_loss = self.run_epoch(train_data, train_mode=True)
            log.info(f'Epoch {ep+1} complete.. Training loss in this epoch {train_loss}...')
            val_loss = self.run_epoch(val_data, train_mode=False)
            log.info(f'Validation of {ep+1} complete.. Validation loss in this epoch {val_loss}...')
            losses.append((ep, train_loss, val_loss))
            if keep_models > 0:
                self.exp.store_model(epoch=ep, model=self.model.state_dict(),
                                     train_score=train_loss,
                                     val_score=val_loss, keep=keep_models)
        summary = '\n'.join(f'{ep:02}\t{tl:.4f}\t{vl:.4f}' for ep, tl, vl in losses)
        log.info(f"==Summary==:\nEpoch\t TrainLoss \t ValidnLoss \n {summary}")

    def run_epoch(self, data_iter, num_batches=None, train_mode=True, path='E1D2E2D1'):
        """
        run a pass over data set
        :param data_iter: batched data set
        :param num_batches: number of batches in the dataset (for tqdm progress bar), None if unknown
        :param train_mode: is it training mode (False if validation mode)
        :return: total loss
        """
        tot_loss = 0.0
        start = time.time()
        self.model.train(train_mode)
        with tqdm(data_iter, total=num_batches, unit='batch') as data_bar:
            for i, batch in enumerate(data_bar):
                batch = batch.to(device)
                # Step clear gradients
                self.model.zero_grad()
                # Step Run forward pass.
                outp_log_probs = self.model(batch, path)
                tok_mask = self.sequence_mask(batch.y_len, batch.max_y_len - 1)
                per_tok_loss = -outp_log_probs
                loss = (per_tok_loss * tok_mask.float()).sum().float() / batch.y_toks
                tot_loss += loss.item()
                learn_rate = ""
                if train_mode:
                    loss.backward()
                    learn_rate = self.optimizer.step()
                    self.optimizer.zero_grad()
                    learn_rate = f'LR={learn_rate:g}'
                elapsed = time.time() - start
                bar_msg = f'Loss:{loss:.4f}, {int(batch.y_toks/elapsed)}toks/s {learn_rate}'
                data_bar.set_postfix_str(bar_msg, refresh=False)
                del batch
        return tot_loss


def __test_seq2seq_model__():
    from rtg.dummy import BatchIterable
    from rtg.module.decoder import Decoder

    vocab_size = 20
    exp = Experiment("tmp.work", config={'model_type': 'seq2seq'}, read_only=True)
    num_epoch = 100

    src = tensor([[2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
                  [2, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4]])
    src_lens = tensor([src.size(1)] * src.size(0))

    for reverse in (False,):
        # train two models;
        #  first, just copy the numbers, i.e. y = x
        #  second, reverse the numbers y=(V + reserved - x)
        log.info(f"====== REVERSE={reverse}; VOCAB={vocab_size}======")
        model, args = Seq2Seq.make_model('DummyA', 'DummyB', vocab_size, vocab_size,
                                         emb_size=100, hid_size=100, n_layers=2)
        trainer = Seq2SeqTrainer(exp=exp, model=model, lr=0.01, warmup_steps=1000)

        decr = Decoder.new(exp, model)
        assert 2 == Batch.bos_val

        def print_res(res):
            for score, seq in res:
                log.info(f'{score:.4f} :: {seq}')

        val_data = list(BatchIterable(vocab_size, 50, 5, reverse=reverse, batch_first=True))
        for epoch in range(num_epoch):
            model.train()
            train_data = BatchIterable(vocab_size, 30, 50, seq_len=12, reverse=reverse,
                                       batch_first=True)
            train_loss = trainer.run_epoch(train_data, num_batches=train_data.num_batches,
                                           train_mode=True)
            val_loss = trainer.run_epoch(val_data, num_batches=len(val_data), train_mode=False)
            log.info(
                f"Epoch {epoch}, training Loss: {train_loss:.4f} \t validation loss:{val_loss:.4f}")
            model.eval()
            res = decr.greedy_decode(src, src_lens, max_len=17)
            print_res(res)


def __test_binmt_model__():
    from rtg.dummy import BatchIterable
    from rtg.module.decoder import Decoder

    vocab_size = 20
    exp = Experiment("tmp.work", config={'model_type': 'binmt'}, read_only=True)
    num_epoch = 100

    src = tensor([[2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
                  [2, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4]])
    src_lens = tensor([src.size(1)] * src.size(0))

    for reverse in (False,):
        # train two models;
        #  first, just copy the numbers, i.e. y = x
        #  second, reverse the numbers y=(V + reserved - x)
        log.info(f"====== REVERSE={reverse}; VOCAB={vocab_size}======")
        model, args = BiNMT.make_model('DummyA', 'DummyB', vocab_size, vocab_size,
                                       emb_size=100, hid_size=100, n_layers=2)
        trainer = BiNmtTrainer(exp=exp, model=model, lr=0.01, warmup_steps=1000)

        decr = Decoder.new(exp, model)
        assert 2 == Batch.bos_val

        def print_res(res):
            for score, seq in res:
                log.info(f'{score:.4f} :: {seq}')

        val_data = list(BatchIterable(vocab_size, 50, 5, reverse=reverse, batch_first=True))
        for epoch in range(num_epoch):
            model.train()
            train_data = BatchIterable(vocab_size, 30, 50, seq_len=12, reverse=reverse,
                                       batch_first=True)
            train_loss = trainer.run_epoch(train_data, num_batches=train_data.num_batches,
                                           train_mode=True)
            val_loss = trainer.run_epoch(val_data, num_batches=len(val_data), train_mode=False)
            log.info(
                f"Epoch {epoch}, training Loss: {train_loss:.4f} \t validation loss:{val_loss:.4f}")
            model.eval()
            res = decr.greedy_decode(src, src_lens, max_len=17)
            print_res(res)


if __name__ == '__main__':
    #__test_binmt_model__()
    __test_seq2seq_model__()

