import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from typing import List


from tgnmt.dataprep import Batch
from tgnmt import TranslationExperiment as Experiment
from tgnmt import device, log
from tgnmt import my_tensor as tensor
from tgnmt.dataprep import BatchIterable, BOS_TOK

BOS_TOK_IDX = BOS_TOK[1]


class RNNEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers=2, dropout=0.4):
        super(RNNEncoder, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout = dropout

        self.embedding = nn.Embedding(input_size, hidden_size)
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


class AttnRNNDecoder(nn.Module):
    def __init__(self, attn_model, hidden_size, output_size, n_layers=2, dropout=0.1):
        super(AttnRNNDecoder, self).__init__()

        # Keep for reference
        self.attn_model = attn_model
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout = dropout

        # Define layers
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.embedding_dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers, dropout=dropout)
        self.concat = nn.Linear(hidden_size * 2, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)

        # Choose attention model
        if attn_model != 'none':
            self.attn = Attn(attn_model, hidden_size)

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

    def forward(self, batch: Batch):
        assert not batch.batch_first
        enc_outs, enc_hids = self.enc(batch.x_seqs, batch.x_len, None)

        dec_inps = tensor([BOS_TOK_IDX] * len(batch), dtype=torch.long)
        dec_hids = enc_hids[:self.dec.n_layers]
        all_dec_outs = torch.zeros(batch.max_y_len, len(batch), self.dec.output_size, device=device)
        for t in range(batch.max_y_len):
            dec_outs, dec_hids, dec_attn = self.dec(dec_inps, dec_hids, enc_outs)
            all_dec_outs[t] = dec_outs
            dec_inps = batch.y_seqs[t]  # Next input is current target
        return all_dec_outs

    @staticmethod
    def make_model(src_vocab: int, tgt_vocab: int, model_dim=50):
        return Seq2Seq(src_vocab, tgt_vocab, model_dim=model_dim)


class Trainer:

    def __init__(self, exp: Experiment, lr=0.0001):
        self.exp = exp
        last_model, last_epoch = self.exp.get_last_saved_model()
        if last_model:
            self.model = torch.load(last_model)
            self.start_epoch = last_epoch + 1
            log.info(f"Resuming training from epoch:{self.start_epoch}, model={last_model}")
            assert type(self.model) is Seq2Seq
        else:
            self.model = Seq2Seq(exp.src_vocab.size() + 1, exp.tgt_vocab.size() + 1)
            self.start_epoch = 0
        self.model = self.model.to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

    @staticmethod
    def sequence_mask(lengths, max_len):
        batch_size = lengths.size(0)
        # create a row [0, 1, ... s] and duplicate this row batch_size times --> [B, S]
        seq_range_expand = torch.arange(0, max_len, dtype=torch.long, device=device).expand(batch_size, max_len)
        # make lengths vectors to [B x 1] and duplicate columns to [B, S]
        seq_length_expand = lengths.unsqueeze(1).expand_as(seq_range_expand)
        return seq_range_expand < seq_length_expand # 0 if padding, 1 otherwise

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

        train_data = BatchIterable(self.exp.train_file, batch_size=batch_size, in_mem=True, batch_first=False)
        # val_data = BatchIterable(self.exp.valid_file, batch_size=batch_size, in_mem=True, batch_first=False)
        keep_models = args.get('keep_models', 4)
        if args.get('resume_train'):
            num_epochs += self.start_epoch
        elif num_epochs <= self.start_epoch:
            raise Exception(f'The model was already trained to {self.start_epoch} epochs. '
                            f'Please increase epoch or clear the existing models')
        for ep in range(self.start_epoch, num_epochs):
            tot_loss = 0.0
            for i, batch in tqdm(enumerate(train_data)):
                # Step clear gradients
                self.model.zero_grad()
                # Step Run forward pass.

                dec_outs = self.model(batch)

                loss = self.masked_cross_entropy(
                    dec_outs.t().contiguous(),  # -> batch x seq
                    batch.y_seqs.t().contiguous(),  # -> batch x seq_len
                    batch.y_len
                )
                tot_loss += loss.item()
                loss.backward()
                self.optimizer.step()

            log.info(f'Epoch {ep+1} complete.. Training loss in this epoch {tot_loss}...')
            self.exp.store_model(epoch=ep, model=self.model, score=tot_loss, keep=keep_models)


class GreedyDecoder:

    def __init__(self, exp: Experiment):
        self.exp = exp
        args = exp.get_model_args()
        self.model = Seq2Seq.make_model(**args)

        last_check_pt, _ = exp.get_last_saved_model()
        log.debug(f'Restoring from {last_check_pt}')
        self.model.load_state_dict(torch.load(last_check_pt))
        self.model.eval()

    def greedy_decode(self, seq: List[int], max_out_len=200):
        # [S, 1] <-- [S]
        x_seqs = tensor(seq, dtype=torch.long).view(-1, 1)
        # [S]
        x_lens = tensor([len(seq)], dtype=torch.long)
        # [S, B=1, d], [S, B=1, d] <-- [S, 1], [S]
        enc_outs, enc_hids = self.model.enc(x_seqs, x_lens, None)
        # [1]
        dec_inps = tensor([BOS_TOK_IDX], dtype=torch.long)
        # [S=n, B=1, d]
        dec_hids = enc_hids[:self.model.dec.n_layers]
        # [S=m]
        final_dec_outs = torch.zeros(max_out_len, dtype=torch.long, device=device)
        for t in range(max_out_len):

            dec_outs, dec_hids, dec_attn = self.model.dec(dec_inps, dec_hids, enc_outs)
            word_prob, word_idx = F.log_softmax(dec_outs, dim=1).view(-1).max(0)
            final_dec_outs[t] = word_idx
            dec_inps[0] = word_idx  # Next input is current output
        return final_dec_outs

    def decode_file(self, inp, out):
        for i, line in enumerate(inp):
            in_toks = line.strip().split()
            log.info(f" Input: {i}: {' '.join(in_toks)}")
            in_seq = self.exp.src_vocab.seq2idx(in_toks)
            out_seq = self.greedy_decode(in_seq)
            out_toks = self.exp.tgt_vocab.idx2seq(out_seq)
            out_line = ' '.join(out_toks)
            log.info(f"Output: {i}: {out_line}")
            out.write(f'{out_line}\n')
