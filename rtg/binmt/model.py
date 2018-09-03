import torch
from torch import nn

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

    def __init__(self, emb_node: Embedder, out_size: int, n_layers: int,
                 bidirectional: bool = True):
        super().__init__()
        self.emb_node = emb_node
        self.out_size = out_size
        self.n_layers = n_layers
        self.bidirectional = bidirectional

        out_size = self.out_size
        if self.bidirectional:
            assert self.out_size % 2 == 0
            out_size = out_size // 2
        self.rnn_node = nn.LSTM(self.emb_node.emb_size, out_size, num_layers=self.n_layers,
                                bidirectional=self.bidirectional, batch_first=True)

    def forward(self, input_seqs: torch.Tensor, input_lengths, hidden=None):
        assert len(input_seqs) == len(input_lengths)
        batch_size, seq_len = input_seqs.shape

        embedded = self.emb_node(input_seqs).view(batch_size, seq_len, self.emb_node.emb_size)
        packed = nn.utils.rnn.pack_padded_sequence(embedded, input_lengths, batch_first=True)
        outputs, hidden = self.rnn_node(packed, hidden)
        outputs, output_lengths = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True,
                                                                   padding_value=PAD_TOK_IDX)
        # Sum bidirectional outputs
        # outputs = outputs[:, :, :self.hid_size] + outputs[:, :, self.hid_size:]
        return outputs, hidden


class SeqDecoder(nn.Module):

    def __init__(self, prev_emb_node: Embedder, generator: Generator, n_layers: int):
        super(SeqDecoder, self).__init__()
        self.prev_emb_node = prev_emb_node
        self.generator = generator
        self.n_layers = n_layers
        self.rnn_node = nn.LSTM(self.prev_emb_node.emb_size, self.generator.vec_size,
                                num_layers=self.n_layers, bidirectional=False, batch_first=True)

    def forward(self, enc_outs, prev_out, last_hidden):
        # Note: we run this one step at a time

        # Get the embedding of the current input word (last output word)
        batch_size = prev_out.size(0)
        assert len(enc_outs) == batch_size
        # S=B x 1 x N
        embedded = self.prev_emb_node(prev_out).view(batch_size, 1, self.prev_emb_node.emb_size)
        # Get current hidden state from input word and last hidden state
        rnn_output, hidden = self.rnn_node(embedded, last_hidden)
        # [B x N ] <- [B x S=1 x N]
        rnn_output = rnn_output.squeeze(1)
        # Finally predict next token
        next_word_distr = self.generator(rnn_output)
        # Return final output, hidden state, and attention weights (for visualization)
        return next_word_distr, hidden


class Seq2Seq(nn.Module):

    def __init__(self, enc: SeqEncoder, dec: SeqDecoder):
        super(Seq2Seq, self).__init__()
        self.enc = enc
        self.dec = dec

    def enc_to_dec_state(self, enc_state):
        if self.enc.bidirectional:
            # h_t, c_t = enc_state
            batch_size = enc_state[0].shape[1]
            # [num_layers * 2, batch_size, hid // 2]
            #    -> [num_layers, 2, batch_size, hid // 2]
            #    -> [num_layers, batch_size, hid]
            #
            return [hc.view(self.enc.n_layers, 2, batch_size, self.enc.out_size // 2)
                        .view(self.enc.n_layers, batch_size, self.enc.out_size) for hc in enc_state]
        return enc_state

    def forward(self, batch: Batch):
        assert batch.batch_first
        batch_size = len(batch)
        enc_outs, enc_hids = self.enc(batch.x_seqs, batch.x_len, None)
        dec_inps = tensor([[BOS_TOK_IDX]] * batch_size, dtype=torch.long)
        dec_hids = self.enc_to_dec_state(enc_hids)
        outp_probs = torch.zeros((batch.max_y_len-1, batch_size), device=device)

        for t in range(1, batch.max_y_len):
            word_probs, dec_hids = self.dec(enc_outs, dec_inps, dec_hids)

            # expected output;; log probability for these indices should be high
            expct_word_idx = batch.y_seqs[:, t].view(batch_size, 1)
            expct_word_log_probs = word_probs.gather(dim=1, index=expct_word_idx)
            outp_probs[t-1] = expct_word_log_probs.squeeze()

            # Randomly switch between gold and the prediction next word
            if random.choice((True, False)):
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
        enc = SeqEncoder(src_embedder, hid_size, n_layers=n_layers, bidirectional=False)
        dec = SeqDecoder(tgt_embedder, tgt_generator, n_layers=n_layers)

        model = Seq2Seq(enc, dec)
        # This was important from their code.
        # Initialize parameters with Glorot / fan_avg.
        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        return model, args


class Seq2SeqTrainer:

    def __init__(self, exp: Experiment, model=None, optim='ADAM', **optim_args):
        self.exp = exp
        self.start_epoch = 0
        if model is None:
            model, args = Seq2Seq.make_model(**exp.model_args)
            last_check_pt, last_epoch = self.exp.get_last_saved_model()
            if last_check_pt:
                log.info(f"Resuming training from epoch:{self.start_epoch}, model={last_check_pt}")
                self.start_epoch = last_epoch + 1
                model.load_state_dict(torch.load(last_check_pt))
            exp.model_args = args

        if torch.cuda.device_count() > 1:
            raise RuntimeError('Please export CUDA_VISIBLE_DEVICES to a single GPU id')
        log.info(f"Moving model to device = {device}")
        self.model = model.to(device=device)
        self.model.train()
        optim_args['lr'] = optim_args.get('lr', 0.001)
        self.optimizer = Optims[optim].new(self.model.parameters(), **optim_args)
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

    def train(self, num_epochs: int, batch_size: int, **args):
        log.info(f'Going to train for {num_epochs} epochs; batch_size={batch_size}')

        train_data = BatchIterable(self.exp.train_file, batch_size=batch_size, batch_first=False,
                                   shuffle=True)
        val_data = BatchIterable(self.exp.valid_file, batch_size=batch_size, batch_first=True,
                                 shuffle=False)
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
        self.model.train(train_mode)
        for i, batch in tqdm(enumerate(data_iter), total=num_batches, unit='batch'):
            batch = batch.to(device)
            # Step clear gradients
            self.model.zero_grad()
            # Step Run forward pass.
            outp_log_probs = self.model(batch)
            tok_mask = self.sequence_mask(batch.y_len, batch.max_y_len-1)
            per_tok_loss = -outp_log_probs
            loss = (per_tok_loss * tok_mask.float()).sum().float() / batch.y_toks
            tot_loss += loss.item()

            if train_mode:
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
            del batch
        return tot_loss

    def overfit_batch(self, batch, max_iters=200):
        log.info("Trying to overfit a batch")
        losses = []
        for i in range(max_iters):
            batch.to(device)
            self.model.zero_grad()
            outp_log_probs = self.model(batch)
            tok_mask = self.sequence_mask(batch.y_len, batch.max_y_len-1)
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


class BiNMT(nn.Module):
    pass


def __test_model__():
    from rtg.dummy import BatchIterable
    from rtg.module.decoder import Decoder

    vocab_size = 15
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
        trainer = Seq2SeqTrainer(exp=exp, model=model, lr=0.01)

        if False:
            batch_1 = list(BatchIterable(vocab_size, 50, 5, reverse=reverse, batch_first=True))[0]
            trainer.overfit_batch(batch_1, max_iters=1000)
            continue

        decr = Decoder.new(exp, model)
        assert 2 == Batch.bos_val

        def print_res(res):
            for score, seq in res:
                log.info(f'{score:.4f} :: {seq}')

        val_data = list(BatchIterable(vocab_size, 50, 5, reverse=reverse, batch_first=True))
        for epoch in range(num_epoch):
            model.train()
            train_data = BatchIterable(vocab_size, 30, 50, reverse=reverse, batch_first=True)
            train_loss = trainer.run_epoch(train_data, num_batches=train_data.num_batches,
                                           train_mode=True)
            val_loss = trainer.run_epoch(val_data, num_batches=len(val_data), train_mode=False)
            log.info(
                f"Epoch {epoch}, training Loss: {train_loss:.4f} \t validation loss:{val_loss:.4f}")
            model.eval()
            res = decr.greedy_decode(src, src_lens, max_len=12)
            print_res(res)


if __name__ == '__main__':
    __test_model__()
