#!/usr/bin/env python
#
# Author: Thamme Gowda [tg at isi dot edu] 
# Created: 10/17/18
from rtg.module.generator import Seq2SeqGenerator
from rtg.module.rnnmt import *
from typing import Mapping
from time import time
from rtg import cpu_device, device, log
import math
from rtg.module.trainer import NoamOpt, Optims
import itertools

# NOTE: this is out of sync

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

        self.model_dim: int = enc1.out_size

        self.paths: Mapping[str, RNNMT] = {
            'E1D1': RNNMT(enc1, dec1),  # ENC1 --> DEC1
            'E2D2': RNNMT(enc2, dec2),  # ENC2 --> DEC2
            # ENC1 --> DEC2 --> ENC2 --> DEC1
            'E1D2E2D1': RNNMT(enc1, dec1, bridge=Seq2SeqBridge(dec2, enc2)),
            # ENC2 --> DEC1 --> ENC1 --> DEC2
            'E2D1E1D2': RNNMT(enc2, dec2, bridge=Seq2SeqBridge(dec1, enc1)),
            ## parallel
            'E1D2': RNNMT(enc1, dec2),
            'E2D1': RNNMT(enc2, dec1),
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
                   hid_size: int = 300, n_layers: int = 2, attention=False, dropout=0.5,
                   exp: Optional[Experiment] = None):
        args = {
            'src_lang': src_lang,
            'tgt_lang': tgt_lang,
            'src_vocab': src_vocab,
            'tgt_vocab': tgt_vocab,
            'emb_size': emb_size,
            'hid_size': hid_size,
            'n_layers': n_layers,
            'attention': attention,
            'dropout': dropout
        }
        src_embedder = Embedder(src_lang, src_vocab, emb_size)
        tgt_embedder = Embedder(tgt_lang, tgt_vocab, emb_size)

        src_generator = Generator(src_lang, vec_size=hid_size, vocab_size=src_vocab)
        tgt_generator = Generator(tgt_lang, vec_size=hid_size, vocab_size=tgt_vocab)

        src_enc = SeqEncoder(src_embedder, hid_size, n_layers=n_layers, bidirectional=True,
                             dropout=dropout)
        tgt_enc = SeqEncoder(tgt_embedder, hid_size, n_layers=n_layers, bidirectional=True,
                             dropout=dropout)

        if attention:
            log.info("Using attention models for decoding")
            src_dec = AttnSeqDecoder(src_embedder, src_generator, n_layers=n_layers,
                                     dropout=dropout)
            tgt_dec = AttnSeqDecoder(tgt_embedder, tgt_generator, n_layers=n_layers,
                                     dropout=dropout)
        else:
            log.info("NOT Using attention models for decoding")
            src_dec = SeqDecoder(src_embedder, src_generator, n_layers=n_layers, dropout=dropout)
            tgt_dec = SeqDecoder(tgt_embedder, tgt_generator, n_layers=n_layers, dropout=dropout)
        model = BiNMT(src_enc, src_dec, tgt_enc, tgt_dec)
        # Initialize parameters with Glorot / fan_avg.
        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        return model, args


# TODO: Label Smoothing


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
        optim_args['weight_decay'] = optim_args.get('weight_decay', 1e-5)
        self.step_size = optim_args.pop('step_size', 512)

        optimizer = Optims[optim].new(self.model.parameters(), **optim_args)
        self.optimizer = NoamOpt(model.model_dim * 2, 2, warmup, optimizer)

        optim_args['warmup_steps'] = warmup
        optim_args['step_size'] = self.step_size
        self.exp.optim_args = optim, optim_args
        if not exp.read_only:
            self.exp.persist_state()

    @staticmethod
    def _get_batch_size_suggestion(step_size, batch_size_approx):
        """
        :param step_size: fixed step_size
        :param batch_size_approx: approximate batch_size
        :return: a tuple of nearest lower, higher good batch_size
        """
        assert 0 < step_size and 0 < batch_size_approx <= step_size
        lower, higher = None, None
        # closest lower multiple
        for i in range(batch_size_approx, 0, -1):
            if i % step_size == 0:
                lower = i
                break

        # closest highest multiple
        for i in range(batch_size_approx, step_size + 1):
            if i % step_size == 0:
                higher = i
                break
        return lower, higher

    @staticmethod
    def _get_step_size_suggestion(step_size_approx, batch_size):
        """
        Suggest a nearest good step_size for the given batch_size
        :param step_size_approx: an approximate step_size
        :param batch_size: a fixed batch_size
        :return: a tuple of nearest lower, higher good step_sizes
        """
        assert 0 < step_size_approx and 0 < batch_size <= step_size_approx
        quotient = step_size_approx / batch_size
        return math.floor(quotient) * batch_size, math.ceil(quotient) * batch_size

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


class BiNmtTrainer(BaseTrainer):
    # TODO: this is obsolete; needs to revise

    def __init__(self, exp: Experiment, model=None, optim='ADAM', **optim_args):
        assert exp.model_type == 'binmt'
        super().__init__(exp, model, optim=optim, model_factory=BiNMT.make_model, **optim_args)

    def _forward_batch(self, batch: Batch, path: str):
        batch = batch.to(device)
        # Step clear gradients
        self.model.zero_grad()
        # Step Run forward pass.
        outp_log_probs = self.model(batch, path)
        tok_mask = self.sequence_mask(batch.y_len, batch.max_y_len - 1)
        per_tok_loss = -outp_log_probs
        loss_node = (per_tok_loss * tok_mask.float()).sum().float() / batch.y_toks
        return loss_node

    def _run_epoch(self, batch_size: int, train_mode: bool):
        torch.set_grad_enabled(train_mode)
        if train_mode:
            mono_src = BatchIterable(self.exp.mono_train_src, batch_size=batch_size,
                                     batch_first=True,
                                     shuffle=True, copy_xy=True)
            mono_tgt = BatchIterable(self.exp.mono_train_tgt, batch_size=batch_size,
                                     batch_first=True,
                                     shuffle=True, copy_xy=True)
        else:
            mono_src = BatchIterable(self.exp.mono_valid_src, batch_size=batch_size,
                                     batch_first=True,
                                     shuffle=False, copy_xy=True)
            mono_tgt = BatchIterable(self.exp.mono_valid_tgt, batch_size=batch_size,
                                     batch_first=True,
                                     shuffle=False, copy_xy=True)

        return self._run_cycle(mono_src, mono_tgt, train_mode)

    def _run_cycle(self, mono_src, mono_tgt, train_mode):
        start = time.time()
        tot_src_loss = 0.0
        tot_src_cyc_loss = 0.0
        tot_tgt_loss = 0.0
        tot_tgt_cyc_loss = 0.0
        torch.set_grad_enabled(train_mode)
        self.model.train(train_mode)
        num_batches = max(mono_src.num_batches, mono_tgt.num_batches)
        # TODO: not sure if this is a good idea. check  the effect of unequal ratio of data
        data = itertools.zip_longest(mono_src, mono_tgt)
        tot_toks = 0

        num_exs = 0
        learn_rate = ""
        if train_mode:
            assert self.step_size % mono_src.batch_size == 0
            assert self.step_size % mono_tgt.batch_size == 0
            self.optimizer.zero_grad()

        with tqdm(data, total=num_batches, unit='batch', dynamic_ncols=True) as data_bar:
            for i, (src_batch, tgt_batch) in enumerate(data_bar):
                num_exs += max(len(src_batch) if src_batch else 0,
                               len(tgt_batch) if tgt_batch else 0)
                batch_losses = []
                if src_batch:
                    tot_toks += src_batch.y_toks * 2
                    src_loss_node = self._forward_batch(src_batch, path='E1D1')
                    tot_src_loss += src_loss_node.item()
                    batch_losses.append(src_loss_node)

                    src_cyc_loss_node = self._forward_batch(src_batch, path='E1D2E2D1')
                    tot_src_cyc_loss += src_cyc_loss_node.item()
                    batch_losses.append(src_cyc_loss_node)
                if tgt_batch:
                    tot_toks += tgt_batch.y_toks * 2
                    tgt_loss_node = self._forward_batch(tgt_batch, path='E2D2')
                    tot_tgt_loss += tgt_loss_node.item()
                    batch_losses.append(tgt_loss_node)

                    tgt_cyc_loss_node = self._forward_batch(tgt_batch, path='E2D1E1D2')
                    batch_losses.append(tgt_cyc_loss_node)
                    tot_tgt_cyc_loss += tgt_cyc_loss_node.item()

                tot_batch_loss_node = sum(batch_losses)
                tot_batch_loss = tot_batch_loss_node.item()
                if train_mode:
                    tot_batch_loss_node.backward()      # accumulate gradients
                    # take an optimizer's step
                    if num_exs % self.step_size == 0:
                        learn_rate = self.optimizer.step()
                        self.optimizer.zero_grad()
                        learn_rate = f'LR={learn_rate:g}'
                elapsed = time.time() - start
                bar_msg = f'Loss:{tot_batch_loss:.4f}, {int(tot_toks/elapsed)}toks/s {learn_rate}'
                data_bar.set_postfix_str(bar_msg, refresh=False)

        # average
        avg_src_loss = tot_src_loss / mono_src.num_batches
        avg_src_cyc_loss = tot_src_cyc_loss / mono_src.num_batches
        avg_tgt_loss = tot_tgt_loss / mono_tgt.num_batches
        avg_tgt_cyc_loss = tot_tgt_cyc_loss / mono_tgt.num_batches
        log.info(f'{"Training " if train_mode else "Validation"} Epoch\'s Losses: \n\t'
                 f' * Source-Source: Tot:{tot_src_loss:g} Avg:{avg_src_loss:g}\n\t'
                 f' * Source-Target-Source: Tot:{tot_src_cyc_loss:g} Avg:{avg_src_cyc_loss:g}\n\t'
                 f' * Target-Target: Tot:{tot_tgt_loss:g} Avg:{avg_tgt_loss:g} \n\t'
                 f' * Target-Source-Target: Tot:{tot_tgt_cyc_loss:g} Avg:{avg_tgt_cyc_loss:g}')
        return sum([avg_src_loss, avg_tgt_loss, avg_src_cyc_loss, avg_tgt_cyc_loss])

    def train(self, num_epochs: int, batch_size: int, **args):
        log.info(f'Going to train for {num_epochs} epochs; batch_size={batch_size}')
        keep_models = args.pop('keep_models', 4)
        if args.pop('resume_train'):
            num_epochs += self.start_epoch
        elif num_epochs <= self.start_epoch:
            raise Exception(f'The model was already trained to {self.start_epoch} epochs. '
                            f'Please increase epoch or clear the existing models')

        losses = []
        for ep in range(self.start_epoch, num_epochs):
            log.info(f"training epoch {ep+1} started...")
            train_loss = self._run_epoch(batch_size=batch_size, train_mode=True)
            log.info(f'Training epoch {ep+1} complete. Train Loss = {train_loss}')

            log.info(f"Validation epoch {ep+1} started...")
            val_loss = self._run_epoch(batch_size=batch_size, train_mode=False)
            log.info(f"Validation epoch {ep+1} complete. Validation Loss = {val_loss}")
            losses.append((ep, train_loss, val_loss))
            if keep_models > 0:
                # Move model to CPU before serializing
                # See https://discuss.pytorch.org/t/why-cuda-runs-out-of-memory-when-calling-torch-save/2219/5
                # otherwise torch causes OOM sometimes
                state = self.model.to(cpu_device).state_dict()
                self.exp.store_model(epoch=ep, model=state, train_score=train_loss,
                                     val_score=val_loss, keep=keep_models)
                self.model = self.model.to(device) # bring it back to GPU device
                del state
            gc.collect()
        summary = '\n'.join(f'{ep:02}\t{tl:.g}\t{vl:g}' for ep, tl, vl in losses)
        log.info(f"==Summary==:\nEpoch\t TrainLoss \t ValidationLoss \n {summary}")


def __test_binmt_model__():
    from rtg.module.decoder import Decoder

    vocab_size = 20
    exp = Experiment("tmp.work", config={'model_type': 'binmt'}, read_only=True)
    num_epoch = 100
    emb_size = 100
    model_dim = 100
    batch_size = 32

    src = tensor([[2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
                  [2, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4]])
    src_lens = tensor([src.size(1)] * src.size(0))

    for reverse in (False,):
        # train two models;
        #  first, just copy the numbers, i.e. y = x
        #  second, reverse the numbers y=(V + reserved - x)
        log.info(f"====== REVERSE={reverse}; VOCAB={vocab_size}======")
        model, args = BiNMT.make_model('DummyA', 'DummyB', vocab_size, vocab_size,
                                       emb_size=emb_size, hid_size=model_dim, n_layers=2)
        trainer = BiNmtTrainer(exp=exp, model=model, lr=0.01, warmup_steps=500, step_size=2*batch_size)

        decr = Decoder.new(exp, model, gen_args={'path': 'E1D1'})
        assert 2 == Batch.bos_val

        def print_res(res):
            for score, seq in res:
                log.info(f'{score:.4f} :: {seq}')

        for epoch in range(num_epoch):
            model.train()
            train_data = BatchIterable(vocab_size, batch_size, 50, seq_len=10, reverse=reverse,
                                       field=exp.tgt_vocab, batch_first=True)
            val_data = BatchIterable(vocab_size, batch_size, 5, reverse=reverse, batch_first=True,
                                     field=exp.tgt_vocab)
            train_loss = trainer._run_cycle(train_data, train_data, train_mode=True)
            val_loss = trainer._run_cycle(val_data, val_data, train_mode=False)
            log.info(
                f"Epoch {epoch}, training Loss: {train_loss:g} \t validation loss:{val_loss:g}")
            model.eval()
            res = decr.greedy_decode(src, src_lens, max_len=17)
            print_res(res)


if __name__ == '__main__':
    __test_binmt_model__()


class BiNMTGenerator(Seq2SeqGenerator):

    def __init__(self, model: BiNMT, field, x_seqs, x_lens, path):
        # pick a sub Seq2Seq model inside the BiNMT model as per the given path
        assert path
        super().__init__(model.paths[path], field, x_seqs, x_lens)
        self.path = path
        self.wrapper = model