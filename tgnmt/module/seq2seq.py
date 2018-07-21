import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

from tgnmt.dataprep import Batch
from tgnmt import TranslationExperiment as Experiment
from tgnmt import device, log
from tgnmt import my_tensor as tensor
from tgnmt.dataprep import BatchIterable, BOS_TOK, PAD_TOK, EOS_TOK
from tgnmt import profile
import gc


BOS_TOK_IDX = BOS_TOK[1]
EOS_TOK_IDX = EOS_TOK[1]
PAD_TOK_IDX = PAD_TOK[1]


class RNNEncoder(nn.Module):

    padding_idx = PAD_TOK_IDX

    def __init__(self, input_size, hidden_size, n_layers=2, dropout=0.4):
        super(RNNEncoder, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout = dropout

        self.embedding = nn.Embedding(input_size, hidden_size, padding_idx=self.padding_idx)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers, dropout=self.dropout, bidirectional=True)

    def forward(self, input_seqs, input_lengths, hidden=None):
        # Note: we run this all at once (over multiple batches of multiple sequences)
        embedded = self.embedding(input_seqs)
        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, input_lengths)
        outputs, hidden = self.gru(packed, hidden)
        outputs, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(outputs)  # unpack (back to padded)
        outputs = outputs[:, :, :self.hidden_size] + outputs[:, :, self.hidden_size:]  # Sum bidirectional outputs
        return outputs, hidden


class Attn(nn.Module):
    """
     FIXME: NOTE: this is very inefficient
    Attention model
    Taken from https://github.com/spro/practical-pytorch/blob/master/seq2seq-translation/seq2seq-translation-batched.ipynb

    """
    def __init__(self, method, hidden_size):
        super(Attn, self).__init__()

        self.method = method
        self.hidden_size = hidden_size

        if self.method == 'general':
            self.attn = nn.Linear(self.hidden_size, hidden_size)

        elif self.method == 'concat':
            self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
            self.v = nn.Parameter(torch.FloatTensor(1, hidden_size, device=device))

    @profile  # Attn
    def forward(self, hidden, encoder_outputs):
        max_len = encoder_outputs.size(0)
        this_batch_size = encoder_outputs.size(1)
        # Create variable to store attention energies
        attn_energies = torch.zeros(this_batch_size, max_len, device=device)  # B x S

        # For each batch of encoder outputs
        for b in range(this_batch_size):
            # Calculate energy for each encoder output
            for i in range(max_len):
                attn_energies[b, i] = self.score(hidden[b], encoder_outputs[i, b])

        # Normalize energies to weights in range 0 to 1, resize to 1 x B x S
        return F.softmax(attn_energies, dim=1).unsqueeze(1)

    def score(self, hidden, encoder_output):
        if self.method == 'dot':
            energy = hidden.dot(encoder_output)
            return energy

        elif self.method == 'general':
            energy = self.attn(encoder_output)
            energy = hidden.dot(energy)
            return energy

        elif self.method == 'concat':
            energy = self.attn(torch.cat((hidden, encoder_output), 1))
            energy = self.v.dot(energy)
            return energy


class GeneralAttn(nn.Module):  # General Attention optimized for batch
    """
    Attention mode
    """
    def __init__(self, hidden_size):
        super(GeneralAttn, self).__init__()

        self.hidden_size = hidden_size
        self.attn = nn.Linear(self.hidden_size, hidden_size)

    @profile  # General Attn
    def forward(self, hidden, encoder_outs):
        # hidden      : [B, D]
        # encoder_out : [S, B, D]
        #    [B, D] --> [S, B, D] ;; repeat hidden sequence_len times
        hid = hidden.unsqueeze(0).expand_as(encoder_outs)
        #  A batched dot product implementation using element wise product followed by sum
        #    [S, B, D]  --> [S, B]   ;; element wise multiply, then sum along the last dim (i.e. model_dim)
        weights = (encoder_outs * hid).sum(dim=-1)
        # [B, S] <-- [S, B]
        weights = weights.t()

        # Normalize energies to weights in range 0 to 1, resize to B x 1 x S
        return F.softmax(weights, dim=1).unsqueeze(1)


class AttnRNNDecoder(nn.Module):
    def __init__(self, attn_model, hidden_size, output_size, n_layers=2, dropout=0.1, padding_idx=Batch.pad_value):
        super(AttnRNNDecoder, self).__init__()

        # Keep for reference
        self.attn_model = attn_model
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout = dropout

        # Define layers
        self.embedding = nn.Embedding(output_size, hidden_size, padding_idx=padding_idx)
        self.embedding_dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers, dropout=dropout)
        self.concat = nn.Linear(hidden_size * 2, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)

        # Choose attention model
        if attn_model != 'none':
            if attn_model == 'general':
                self.attn = GeneralAttn(hidden_size)
            else:
                self.attn = Attn(attn_model, hidden_size)

    @profile            # AttnRNNDecoder
    def forward(self, input_seq, last_hidden, encoder_outputs):
        # Note: we run this one step at a time

        # Get the embedding of the current input word (last output word)
        batch_size = input_seq.size(0)
        embedded = self.embedding(input_seq)
        embedded = self.embedding_dropout(embedded)
        embedded = embedded.view(1, batch_size, self.hidden_size)  # S=1 x B x N

        # Get current hidden state from input word and last hidden state
        rnn_output, hidden = self.gru(embedded, last_hidden)
        # [B x N ] <- [S=1 x B x N]
        rnn_output = rnn_output.squeeze(0)

        # Calculate attention from current RNN state and all encoder outputs;
        # apply to encoder outputs to get weighted average
        attn_weights = self.attn(rnn_output, encoder_outputs)
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1))  # B x S=1 x N

        # Attentional vector using the RNN hidden state and context vector
        # concatenated together (Luong eq. 5)
        # rnn_output = rnn_output.squeeze(0)  # S=1 x B x N -> B x N
        context = context.squeeze(1)  # B x S=1 x N -> B x N
        concat_input = torch.cat((rnn_output, context), 1)
        concat_output = F.tanh(self.concat(concat_input))

        # Finally predict next token (Luong eq. 6, without softmax)
        output = self.out(concat_output)
        # Return final output, hidden state, and attention weights (for visualization)
        return output, hidden, attn_weights


class Seq2Seq(nn.Module):

    def __init__(self, src_vocab: int, tgt_vocab: int, model_dim=50):
        super(Seq2Seq, self).__init__()
        self.enc = RNNEncoder(src_vocab, model_dim, n_layers=1)
        self.dec = AttnRNNDecoder('general', model_dim, tgt_vocab, n_layers=1)

    @profile    # Seq2Seq
    def forward(self, batch: Batch, k=1):
        assert not batch.batch_first
        enc_outs, enc_hids = self.enc(batch.x_seqs, batch.x_len, None)

        batch_size = len(batch)
        dec_inps = tensor([BOS_TOK_IDX] * batch_size, dtype=torch.long)
        dec_hids = enc_hids[:self.dec.n_layers]
        """
        all_dec_outs = torch.zeros((batch.max_y_len, len(batch), self.dec.output_size), device=device)

        for t in range(batch.max_y_len):
            dec_outs, dec_hids, dec_attn = self.dec(dec_inps, dec_hids, enc_outs)
            all_dec_outs[t] = dec_outs
            dec_inps = batch.y_seqs[t]      # Next input is current target
            del dec_attn, dec_outs   # free memory
        return all_dec_outs
        """
        """
        Problem here is, the output vocabulary is usually too big.
        An output tensor of size [MaxSeqLen x BatchSize x VocabSize] is unrealistic for large BatchSize or MaxSeqLen
        So, a trick is used. 
        In reality, we don't really need [MaxSeqLen x BatchSize x VocabSize]
        In training:
            We already know which word we are expecting 
            we need probabilities of desired output word so as to reduce the loss 
            so a tensor of size [MaxSeqLen x BatchSize x 1] is much smaller    
        In production or eval:
            We need top 'k' words' probability. Greedy decoder would set k=1, beam decoder would set k= beam size
            so a tensor of size [MaxSeqLen x BatchSize x k] is much smaller
        """
        # return log probabilities of desired output
        outp_probs = torch.zeros((batch.max_y_len, batch_size), device=device)
        for t in range(1, batch.max_y_len):
            dec_outs, dec_hids, dec_attn = self.dec(dec_inps, dec_hids, enc_outs)
            decoder_lastt = dec_outs  # last time stamp of decoder
            word_probs = F.log_softmax(decoder_lastt, dim=-1)
            expct_word_idx = batch.y_seqs[t]   # expected output;; log probability for these indices should be high
            expct_word_log_probs = word_probs.gather(dim=1, index=expct_word_idx.view(batch_size, 1))
            outp_probs[t] = expct_word_log_probs.squeeze()
            dec_inps = expct_word_idx          # Next input is current target
            del dec_attn, dec_outs        # free memory
        return outp_probs

    @staticmethod
    def make_model(src_vocab: int, tgt_vocab: int, model_dim=50):
        args = {'src_vocab': src_vocab, 'tgt_vocab': tgt_vocab, 'model_dim': model_dim}
        return Seq2Seq(src_vocab, tgt_vocab, model_dim=model_dim), args


class Trainer:

    @profile
    def __init__(self, exp: Experiment, model=None, lr=0.0001):
        self.exp = exp
        self.start_epoch = 0
        if model is None:
            model = Seq2Seq(**exp.model_args).to(device)
            last_check_pt, last_epoch = self.exp.get_last_saved_model()
            if last_check_pt:
                log.info(f"Resuming training from epoch:{self.start_epoch}, model={last_check_pt}")
                self.start_epoch = last_epoch + 1
                model.load_state_dict(torch.load(last_check_pt))
        log.info(f"Moving model to device = {device}")
        self.model = model.to(device=device)
        self.model.train()
        del model           # this was on CPU, free that memory
        gc.collect()        # should the GC cleanup CPU buffers after moving to GPU ?
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

    @staticmethod
    def sequence_mask(lengths, max_len):
        batch_size = lengths.size(0)
        # create a row [0, 1, ... s] and duplicate this row batch_size times --> [B, S]
        seq_range_expand = torch.arange(0, max_len, dtype=torch.long, device=device).expand(batch_size, max_len)
        # make lengths vectors to [B x 1] and duplicate columns to [B, S]
        seq_length_expand = lengths.unsqueeze(1).expand_as(seq_range_expand)
        return seq_range_expand < seq_length_expand     # 0 if padding, 1 otherwise

    @classmethod
    def masked_cross_entropy(cls, logits, target, lengths):
        """
        Args:
            logits: a FloatTensor of size (batch, max_len, num_classes) which contains the
                unnormalized probability for each class.
            target: A Variable containing a LongTensor of size
                (batch, max_len) which contains the index of the true
                class for each corresponding step.
            lengths: a LongTensor of size (batch,)
                which contains the length of each data in a batch.
        Returns:
            loss: An average loss value masked by the length.
        """

        # [B * S,  V]  <-- [S, B, V]
        logits_flat = logits.view(-1, logits.size(-1))

        # [B * S, V]  <-- log_softmax along V
        log_probs_flat = F.log_softmax(logits_flat, dim=1)

        # [B * S, 1] <-- [B, S]
        target_flat = target.view(-1, 1)

        # [B * S, 1] <-- [B*S, V] by lookup on V using indices specified in target_flat
        losses_flat = -torch.gather(log_probs_flat, dim=1, index=target_flat)

        # [B, S] <-- [ B * S, 1]
        losses = losses_flat.view(*target.size())
        # [B, S]
        mask = cls.sequence_mask(lengths, max_len=target.size(1))
        losses = losses * mask.float()
        loss = losses.sum() / lengths.float().sum()
        return loss

    def train(self, num_epochs: int, batch_size: int, **args):
        log.info(f'Going to train for {num_epochs} epochs; batch_size={batch_size}')

        train_data = BatchIterable(self.exp.train_file, batch_size=batch_size, batch_first=False, shuffle=True)
        # val_data = BatchIterable(self.exp.valid_file, batch_size=batch_size, in_mem=True, batch_first=False)
        keep_models = args.get('keep_models', 4)
        if args.get('resume_train'):
            num_epochs += self.start_epoch
        elif num_epochs <= self.start_epoch:
            raise Exception(f'The model was already trained to {self.start_epoch} epochs. '
                            f'Please increase epoch or clear the existing models')
        for ep in range(self.start_epoch, num_epochs):
            tot_loss = self.run_epoch(train_data)
            log.info(f'Epoch {ep+1} complete.. Training loss in this epoch {tot_loss}...')
            if keep_models > 0:
                self.exp.store_model(epoch=ep, model=self.model.state_dict(), score=tot_loss, keep=keep_models)

    def run_epoch(self, train_data):
        tot_loss = 0.0
        for i, batch in tqdm(enumerate(train_data), total=train_data.num_batches):
            # Step clear gradients
            self.model.zero_grad()

            # Step Run forward pass.
            # dec_outs = self.model(batch)
            outp_log_probs = self.model(batch)
            per_tok_loss = -outp_log_probs.t()
            tok_mask = self.sequence_mask(batch.y_len, batch.max_y_len)
            loss = (per_tok_loss * tok_mask.float()).sum().float() / batch.y_toks
            """
            loss = self.masked_cross_entropy(
                dec_outs.t().contiguous(),  # -> batch x seq
                batch.y_seqs.t().contiguous(),   # -> batch x seq_len
                batch.y_len
            )
            """
            tot_loss += loss.item()
            loss.backward()
            self.optimizer.step()
            del batch
            gc.collect()

        return tot_loss


if __name__ == '__main__':
    from tgnmt.dummy import BatchIterable
    from tgnmt.module.decoder import Decoder
    vocab_size = 25
    exp = Experiment("work", config={'model_type': 'rnn'}, read_only=True)
    num_epoch = 20
    test_x_seqs = tensor([Batch.bos_val, 4, 5, 6, 7, 8, 9, 10, 11]).view(1, -1)
    test_x_lens = tensor([test_x_seqs.size(1)])

    for reverse in (False, True):
        # train two models;
        #  first, just copy the numbers, i.e. y = x
        #  second, reverse the numbers y=(V + reserved - x)
        log.info(f"====== REVERSE={reverse}; VOCAB={vocab_size}======")
        model = Seq2Seq.make_model(vocab_size, vocab_size)[0]
        trainer = Trainer(exp, model=model)
        decoder = Decoder.new(exp, model)
        for ep in range(num_epoch):
            log.info(f"Running epoch {ep+1}")
            data = BatchIterable(vocab_size, batch_size=30, n_batches=50, reverse=reverse)
            model.train()
            loss = trainer.run_epoch(train_data=data)
            log.info(f"Epoch {ep+1} finish. Loss = {loss:.4f}")
            model.eval()
            out = decoder.greedy_decode(x_seqs=test_x_seqs, x_lens=test_x_lens, max_len=9)[0]
            log.info(f"Prediction: score:{out[0]:.4f} :: seq: {out[1].data}")
