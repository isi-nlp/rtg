import random
from typing import Optional, Callable

import torch
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm

from rtg import log, TranslationExperiment as Experiment
from rtg import my_tensor as tensor, device
from rtg.data.dataset import Batch, BatchIterable, padded_sequence_mask
from rtg.data.codec import Field
from rtg.module import NMTModel
from rtg.module.trainer import TrainerState, SteppedTrainer

PAD_IDX = Field.pad_idx  #


class Embedder(nn.Embedding):
    """
    This module takes words (word IDs, not the text ) and creates vectors.
    For the inverse operation see  `Generator` module
    """

    def __init__(self, name: str, vocab_size: int, emb_size: int,
                 weights: Optional[torch.Tensor] = None, freeze: bool = False, pad_idx=PAD_IDX):
        self.name = name
        self.vocab_size = vocab_size
        self.emb_size = emb_size
        super(Embedder, self).__init__(self.vocab_size, self.emb_size, padding_idx=pad_idx,
                                       _weight=weights)
        self.weight.requires_grad = not freeze


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

    def forward(self, x, log_probs=True):
        x_feats = self.proj(x)
        return (F.log_softmax if log_probs else F.softmax)(x_feats, dim=-1)


class SeqEncoder(nn.Module):

    def __init__(self, embedder: Embedder, hid_size: int, n_layers: int,
                 bidirectional: bool = True, dropout=0.5, ext_embedder: Embedder = None):
        super().__init__()
        self.emb: Embedder = embedder
        self.dropout = nn.Dropout(dropout)
        # Input size of RNN, which is same as embedding vector size
        self.emb_size = self.emb.emb_size
        # the output size of RNN, ie. hidden representation size
        self.hid_size = hid_size
        self.n_layers = n_layers
        self.bidirectional = bidirectional

        hid_size = self.hid_size
        if self.bidirectional:
            assert hid_size % 2 == 0
            hid_size = hid_size // 2
        self.rnn_node = nn.LSTM(self.emb_size, hid_size,
                                num_layers=self.n_layers,
                                bidirectional=self.bidirectional,
                                batch_first=True,
                                dropout=dropout if n_layers > 1 else 0)
        # if external embeddings are provided
        self.ext_embedder = ext_embedder
        # The output feature vectors vectors
        self.out_size = self.hid_size + (self.ext_embedder.emb_size if ext_embedder else 0)

    def forward(self, input_seqs: torch.Tensor, input_lengths, hidden=None, pre_embedded=False):
        assert len(input_seqs) == len(input_lengths)
        if pre_embedded:
            embedded = input_seqs
            batch_size, seq_len, emb_size = input_seqs.shape
            assert emb_size == self.emb_size
        else:
            batch_size, seq_len = input_seqs.shape
            embs = self.emb(input_seqs)
            embedded = embs.view(batch_size, seq_len, self.emb_size)

        embedded = self.dropout(embedded)
        packed = nn.utils.rnn.pack_padded_sequence(embedded, input_lengths, batch_first=True)
        outputs, hidden = self.rnn_node(packed, hidden)
        outputs, output_lengths = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True,
                                                                   padding_value=PAD_IDX)
        # Sum bidirectional outputs
        # outputs = outputs[:, :, :self.hid_size] + outputs[:, :, self.hid_size:]
        dec_state = self.to_dec_state(hidden)
        if self.ext_embedder is not None:
            ext_embs = self.ext_embedder(input_seqs).view(batch_size, seq_len,
                                                          self.ext_embedder.emb_size)
            ext_embs = self.dropout(ext_embs)
            outputs = torch.cat((outputs, ext_embs), dim=-1)
        return outputs, dec_state

    def to_dec_state(self, enc_state):
        # get the last layer's last time step output
        # lnhn is layer n hidden n which is last layer last hidden. similarly lncn
        hns, cns = enc_state
        if self.bidirectional:
            # cat bidirectional
            lnhn = hns.view(self.n_layers, 2, hns.shape[1], hns.shape[-1])[-1]
            lnhn = torch.cat([lnhn[0], lnhn[1]], dim=1)
            lncn = cns.view(self.n_layers, 2, cns.shape[1], cns.shape[-1])[-1]
            lncn = torch.cat([lncn[0], lncn[1]], dim=1)
        else:
            lnhn = hns.view(self.n_layers, hns.shape[1], hns.shape[-1])[-1]
            lncn = cns.view(self.n_layers, cns.shape[1], cns.shape[-1])[-1]

        # lnhn and lncn hold compact representation
        # duplicate for decoder layers
        return (lnhn.expand(self.n_layers, *lnhn.shape).contiguous(),
                lncn.expand(self.n_layers, *lncn.shape).contiguous())


class SeqDecoder(nn.Module):

    def __init__(self, prev_emb_node: Embedder, generator: Generator, n_layers: int, dropout=0.5):
        super(SeqDecoder, self).__init__()
        self.prev_emb = prev_emb_node
        self.dropout = nn.Dropout(dropout)
        self.generator: Generator = generator
        self.n_layers = n_layers
        self.emb_size = self.prev_emb.emb_size
        self.hid_size = self.generator.vec_size
        self.rnn_node = nn.LSTM(self.emb_size, self.hid_size, num_layers=self.n_layers,
                                bidirectional=False, batch_first=True,
                                dropout=dropout if n_layers > 1 else 0)

    def forward(self, enc_outs: Optional, prev_out, last_hidden, gen_probs=True, log_probs=True):
        # Note: we run this one step at a time

        # Get the embedding of the current input word (last output word)
        batch_size = prev_out.size(0)
        if enc_outs is not None:
            assert len(enc_outs) == batch_size
        # S=B x 1 x N
        embedded = self.prev_emb(prev_out).view(batch_size, 1, self.prev_emb.emb_size)
        embedded = self.dropout(embedded)
        # Get current hidden state from input word and last hidden state
        rnn_output, hidden = self.rnn_node(embedded, last_hidden)

        # [B x N ] <- [B x S=1 x N]
        rnn_output = rnn_output.squeeze(1)

        if gen_probs:
            # Finally predict next token
            next_word_distr = self.generator(rnn_output, log_probs=log_probs)
            # Return final output, hidden state, and attention weights (for visualization)
            return next_word_distr, hidden, None
        else:
            return rnn_output, hidden, None


class AttnModel(nn.Module):
    """
    Attention model
    """

    def __init__(self, inp_size, out_size=None, att_type='dot'):
        """
        :param inp_size: Input size on which the the attention
        :param out_size: Output of attention
        """
        super(AttnModel, self).__init__()
        self.inp_size = inp_size
        self.out_size = out_size if out_size is not None else inp_size
        if att_type == 'dot':
            assert self.inp_size == self.out_size
        elif att_type == 'general':
            self.attn_W = nn.Linear(self.inp_size, self.out_size)
        self.attn_type = att_type
        self.attn_func = {
            'dot': self.dot_attn,
            'general': self.general_attn
        }[self.attn_type]

    @staticmethod
    def dot_attn(this_rnn_out, encoder_outs):
        # this_rnn_out: [B, D]
        # encoder_out : [B, S, D]
        #    [B, D] --> [B, S, D] ;; repeat hidden sequence_len times
        this_run_out = this_rnn_out.unsqueeze(1).expand_as(encoder_outs)
        #  A batched dot product implementation using element wise product followed by sum
        #    [B, S]  <-- [B, S, D]
        # element wise multiply, then sum along the last dim (i.e. model_dim)
        weights = (encoder_outs * this_run_out).sum(dim=-1)
        return weights

    def general_attn(self, this_rnn_out, encoder_outs):
        # First map the encoder_outs to new vector space using attn_W
        mapped_enc_outs = self.attn_W(encoder_outs)
        # Then compute the dot
        return self.dot_attn(this_rnn_out, mapped_enc_outs)

    def forward(self, this_rnn_out, encoder_outs):
        assert encoder_outs.shape[-1] == self.inp_size
        assert this_rnn_out.shape[-1] == self.out_size

        weights = self.attn_func(this_rnn_out, encoder_outs)
        # Normalize energies to weights in range 0 to 1
        return F.softmax(weights, dim=1)


class AttnSeqDecoder(SeqDecoder):
    def __init__(self, prev_emb_node: Embedder, generator: Generator, n_layers: int,
                 ctx_size: Optional[int] = None,
                 dropout: float = 0.5, attention='dot'):
        super(AttnSeqDecoder, self).__init__(prev_emb_node, generator, n_layers, dropout=dropout)

        if attention and type(attention) is bool:
            # historical reasons, it was boolean in the beginning
            attention = 'dot'
        ctx_size = ctx_size if ctx_size else self.hid_size
        self.attn = AttnModel(inp_size=ctx_size, out_size=self.hid_size, att_type=attention)
        # Output from decoder rnn + ctx
        self.merge = nn.Linear(self.hid_size + ctx_size, self.hid_size)

    def forward(self, enc_outs, prev_out, last_hidden, gen_probs=True, log_probs=True):
        # Note: we run this one step at a time

        # Get the embedding of the current input word (last output word)
        batch_size = prev_out.size(0)
        embedded = self.prev_emb(prev_out)
        embedded = embedded.view(batch_size, 1, self.prev_emb.emb_size)
        embedded = self.dropout(embedded)
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
            output_probs = self.generator(concat_output, log_probs=log_probs)
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
        self.inp_size = dec.hid_size
        self.out_size = enc.hid_size

    def forward(self, enc_outs, enc_hids, max_len, bos_idx):
        batch_size = len(enc_outs)
        assert batch_size == enc_hids[0].shape[1] == enc_hids[1].shape[1]

        dec_inps = tensor([[bos_idx]] * batch_size, dtype=torch.long)
        dec_hids = enc_hids
        result = torch.zeros((batch_size, max_len, self.dec.hid_size), device=device)
        for t in range(max_len):
            dec_outs, dec_hids, _ = self.dec(enc_outs, dec_inps, dec_hids, gen_probs=False)
            result[:, t, :] = dec_outs

        # TODO: check how hidden state flows
        enc_outs, enc_hids = self.enc(result, [max_len] * batch_size, pre_embedded=True)
        return enc_outs, enc_hids


class RNNMT(NMTModel):

    def __init__(self, enc: SeqEncoder, dec: SeqDecoder, bridge: Seq2SeqBridge = None):
        super(RNNMT, self).__init__()
        self.enc: SeqEncoder = enc
        self.dec: SeqDecoder = dec
        if bridge:
            # enc --> bridge.dec --> bridge.enc --> dec
            assert enc.hid_size == bridge.inp_size
            assert bridge.out_size == dec.hid_size
        else:
            # enc --> dec
            assert enc.hid_size == dec.hid_size
        self.bridge = bridge

    def init_src_embedding(self, weights):
        log.info("Initializing source embeddings")
        assert weights.shape == self.enc.emb.weight.shape
        self.enc.emb.weight.data.copy_(weights.data)

    def init_tgt_embedding(self, weights, input=True, output=True):
        if input:
            log.info("Initializing target input embeddings")
            assert weights.shape == self.dec.prev_emb.weight.shape
            self.dec.prev_emb.weight.data.copy_(weights.data)
        if output:
            log.info("Initializing target output embeddings")
            assert weights.shape == self.dec.generator.proj.weight.shape
            self.dec.generator.proj.weight.data.copy_(weights.data)

    @property
    def model_dim(self):
        return self.enc.hid_size

    @property
    def model_type(self):
        return 'rnnmt'

    @property
    def vocab_size(self):
        return self.dec.generator.vocab_size

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

        dec_inps = tensor([[batch.bos_val]] * batch_size, dtype=torch.long)
        dec_hids = enc_hids
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
                   hid_size: int = 300, n_layers: int = 2, attention='general', dropout=0.33,
                   tied_emb: Optional[str] = 'three-way', exp: Experiment = None):
        args = {
            'src_lang': src_lang,
            'tgt_lang': tgt_lang,
            'src_vocab': src_vocab,
            'tgt_vocab': tgt_vocab,
            'emb_size': emb_size,
            'hid_size': hid_size,
            'n_layers': n_layers,
            'attention': attention,
            'dropout': dropout,
            'tied_emb': tied_emb
        }
        log.info(f"Make RNN NMT model, args= {args}")
        src_embedder = Embedder(src_lang, src_vocab, emb_size)
        tgt_embedder = Embedder(tgt_lang, tgt_vocab, emb_size)
        tgt_generator = Generator(tgt_lang, vec_size=hid_size, vocab_size=tgt_vocab)
        if tied_emb:
            assert src_vocab == tgt_vocab
            if tied_emb == 'three-way':
                log.info('Tying embedding three way : SrcIn == TgtIn == TgtOut')
                src_embedder.weight = tgt_embedder.weight
                tgt_generator.proj.weight = tgt_embedder.weight
            elif tied_emb == 'two-way':
                log.info('Tying embedding two way : SrcIn == TgtIn')
                src_embedder.weight = tgt_embedder.weight
            else:
                raise Exception('Invalid argument to tied_emb; Known: {three-way, two-way}')

        ext_embedder = None
        if exp:
            if exp.ext_emb_src_file.exists():
                log.info("Loading aligned embeddings.")
                aln_emb_weights = torch.load(str(exp.ext_emb_src_file))
                rows, cols = aln_emb_weights.shape
                log.info(f"Loaded aligned embeddings: shape={aln_emb_weights.shape}")
                assert rows == src_vocab, \
                    f'aln_emb_src vocabulary ({rows})' \
                    f' should be same as src_vocab ({src_vocab})'

                ext_embedder = Embedder(name=src_lang,
                                        vocab_size=rows,
                                        emb_size=cols,
                                        weights=aln_emb_weights,
                                        freeze=True)
                if attention != 'general':
                    log.warning("Using attention=general because it is necessary for"
                                " aligned embeddings")
                    attention = 'general'
                    args['attention'] = attention

        enc = SeqEncoder(src_embedder, hid_size, n_layers=n_layers, bidirectional=True,
                         dropout=dropout, ext_embedder=ext_embedder)
        if attention:
            log.info(f"Using attention={attention} models for decoding")
            dec = AttnSeqDecoder(tgt_embedder, tgt_generator,
                                 ctx_size=enc.out_size,
                                 n_layers=n_layers,
                                 dropout=dropout,
                                 attention=attention)
        else:
            log.info("NOT Using attention models for decoding")
            dec = SeqDecoder(tgt_embedder, tgt_generator, n_layers=n_layers, dropout=dropout)

        model = RNNMT(enc, dec)
        # Initialize parameters with Glorot / fan_avg.
        model.init_params()
        return model, args


def aeq(*items):
    for i in items[1:]:
        if items[0] != i:
            return False
    return True


class SimpleLossFunction:

    def __init__(self, optim):
        self.optim = optim

    def __call__(self, log_probs, batch: Batch, train_mode: bool) -> float:
        per_tok_loss = -log_probs

        tok_mask = padded_sequence_mask(batch.y_len, batch.max_y_len - 1)
        norm = batch.y_toks
        loss = (per_tok_loss * tok_mask.float()).sum().float() / norm
        if train_mode:
            loss.backward()
            self.optim.step()
            self.optim.zero_grad()
        return loss.item() * norm


class SteppedRNNMTTrainer(SteppedTrainer):

    def __init__(self, exp: Experiment,
                 model: Optional[RNNMT] = None,
                 optim: str = 'ADAM',
                 **optim_args):
        super().__init__(exp, model, model_factory=RNNMT.make_model, optim=optim, **optim_args)
        self.loss_func = SimpleLossFunction(optim=self.opt)

    def run_valid_epoch(self, data_iter: BatchIterable) -> float:
        state = TrainerState(self.model, -1)
        with tqdm(data_iter, total=data_iter.num_batches, unit='batch',
                  dynamic_ncols=True) as data_bar:
            for i, batch in enumerate(data_bar):
                batch = batch.to(device)
                # Step clear gradients
                self.model.zero_grad()
                # Step Run forward pass.
                outp_log_probs = self.model(batch)
                loss = self.loss_func(outp_log_probs, batch, train_mode=False)
                bar_msg, _ = state.step(batch.y_toks, loss)
                data_bar.set_postfix_str(bar_msg, refresh=False)
                del batch
        return state.running_loss()

    def train(self, steps: int, check_point: int, batch_size: int, fine_tune=False,
              check_pt_callback: Optional[Callable] = None, **args):
        log.info(f'Going to train for {steps} steps; batch_size={batch_size}; '
                 f'check point size:{check_point}; fine tune={fine_tune}')
        keep_models = args.get('keep_models', 4)  # keep last _ models and delete the old
        sort_by = args.get('sort_by', 'random')
        if steps <= self.start_step:
            raise Exception(f'The model was already trained to {self.start_step} steps. '
                            f'Please increase the steps or clear the existing models')
        train_data = self.exp.get_train_data(batch_size=batch_size, steps=steps - self.start_step,
                                             sort_by=sort_by, shuffle=True, batch_first=True,
                                             fine_tune=fine_tune)
        val_data = self.exp.get_val_data(batch_size, shuffle=False, batch_first=True,
                                         sort_desc=True)

        train_state = TrainerState(self.model, check_point=check_point)
        train_state.train_mode(True)
        unsaved_state = False
        with tqdm(train_data, initial=self.start_step, total=steps, unit='batch',
                  dynamic_ncols=True) as data_bar:
            for batch in data_bar:
                batch = batch.to(device)
                # Step clear gradients
                self.model.zero_grad()
                # Step Run forward pass.
                outp_log_probs = self.model(batch)

                loss = self.loss_func(outp_log_probs, batch, True)
                unsaved_state = True
                self.tbd.add_scalars('training', {'step_loss': loss,
                                                  'learn_rate': self.opt.curr_lr},
                                     self.opt.curr_step)
                bar_msg, is_check_pt = train_state.step(batch.y_toks, loss)
                bar_msg += f', LR={self.opt.curr_lr:g}'
                data_bar.set_postfix_str(bar_msg, refresh=False)

                del batch  # TODO: force free memory
                if is_check_pt:
                    train_loss = train_state.reset()
                    train_state.train_mode(False)
                    val_loss = self.run_valid_epoch(val_data)
                    self.make_check_point(train_loss, val_loss=val_loss, keep_models=keep_models)
                    if check_pt_callback:
                        check_pt_callback(model=self.model,
                                          step=self.opt.curr_step,
                                          train_loss=train_loss)
                    train_state.train_mode(True)
                    unsaved_state = False

        if unsaved_state:
            # End of training
            train_loss = train_state.reset()
            train_state.train_mode(False)
            val_loss = self.run_valid_epoch(val_data)
            self.make_check_point(train_loss, val_loss=val_loss, keep_models=keep_models)


def __test_seq2seq_model__():
    """
        batch_size = 4
        p = '/Users/tg/work/me/rtg/saral/runs/1S-rnn-basic'
        exp = Experiment(p)
        steps = 3000
        check_pt = 100
        trainer = SteppedRNNNMTTrainer(exp=exp, lr=0.01, warmup_steps=100)
        trainer.train(steps=steps, check_point=check_pt, batch_size=batch_size)
    """
    from rtg.data.dummy import DummyExperiment
    from rtg.module.decoder import Decoder

    vocab_size = 50
    batch_size = 30
    exp = DummyExperiment("tmp.work", config={'model_type': 'seq'
                                                            '2seq'},
                          read_only=True, vocab_size=vocab_size)
    emb_size = 100
    model_dim = 100
    steps = 3000
    check_pt = 100

    assert 2 == Batch.bos_val
    src = tensor([[4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
                  [13, 12, 11, 10, 9, 8, 7, 6, 5, 4]])
    src_lens = tensor([src.size(1)] * src.size(0))

    for reverse in (False,):
        # train two models;
        #  first, just copy the numbers, i.e. y = x
        #  second, reverse the numbers y=(V + reserved - x)
        log.info(f"====== REVERSE={reverse}; VOCAB={vocab_size}======")
        model, args = RNNMT.make_model('DummyA', 'DummyB', vocab_size, vocab_size, attention='dot',
                                       emb_size=emb_size, hid_size=model_dim, n_layers=1)
        trainer = SteppedRNNMTTrainer(exp=exp, model=model, lr=0.01, warmup_steps=100)
        decr = Decoder.new(exp, model)

        def check_pt_callback(**args):
            res = decr.greedy_decode(src, src_lens, max_len=17)
            for score, seq in res:
                log.info(f'{score:.4f} :: {seq}')

        trainer.train(steps=steps, check_point=check_pt, batch_size=batch_size,
                      check_pt_callback=check_pt_callback)


if __name__ == '__main__':
    __test_seq2seq_model__()
