import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from tgnmt import device
from tgnmt.dataprep import Batch


class LSTMEncoder(nn.Module):
    """
    Encoder module for encoding sequences
    """

    def __init__(self, vocab_size, emb_dim=100, hid_dim=100, pad_idx=Batch.pad_value, dropout=0.4, last_step_only=True,
                 num_layers=2, num_directions=2):
        """
        :param vocab_size: size of vocabulary i.e. maximum index in the input sequence
        :param emb_dim: embedding dimension (of word vectors)
        :param hid_dim: dimension of RNN units
        :param pad_idx: input index that is used for padding
        :param dropout: dropout rate for RNN
        :param last_step_only: return the last step output only; default returns all the time steps
        :param num_layers: number of layers in RNN
        :param num_directions: bidirectional = 2, uni directional = 1
        """
        super(LSTMEncoder, self).__init__()
        self.hid_dim = hid_dim
        self.embeddings = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        self.num_layers = num_layers
        assert 0 < num_directions < 2
        self.num_dirs = num_directions
        self.last_step_only = last_step_only
        self.rnn = nn.LSTM(emb_dim, hid_dim, batch_first=True, bidirectional=self.num_dirs == 2,
                           num_layers=self.num_layers, dropout=dropout)

    def init_hidden(self, batch_size=1):
        return (torch.zeros(self.num_layers * self.num_dirs, batch_size, self.hid_dim, device=device),
                torch.zeros(self.num_layers * self.num_dirs, batch_size, self.hid_dim, device=device))

    def forward(self, batch: Batch):
        batch_size, seq_len = batch.x_seqs.size()  # seq_len is padded sequence length
        hidden = self.init_hidden(batch_size=batch_size)
        embeds = self.embeddings(batch.x_seqs).view(batch_size, seq_len, -1)  # [batch x max_len x emb_dim]
        packed_input = pack_padded_sequence(embeds, batch.x_len, batch_first=True)
        packed_output, _ = self.rnn(packed_input, hidden)
        rnn_out, _ = pad_packed_sequence(packed_output, batch_first=True, total_length=batch.max_x_len)
        if self.last_step_only:
            # last_time_step = rnn_out[:, -1, :]  # ERROR: second dimension should be picked according to batch.x_len
            # Thanks to Nelson: https://blog.nelsonliu.me/2018/01/24/extracting-last-timestep-outputs-from-pytorch-rnns/
            b_dim, t_dim, f_dim = 0, 1, 2  # batch, time, features
            # subtract 1 from len --> [Batch x 1] -->  expand [Batch x Feats] --> add a dim [Batch x 1 x Feats]
            time_idx = (batch.x_len - 1).view(-1, 1).expand(-1, rnn_out.size(f_dim)).unsqueeze(t_dim)
            # pick values --> remove time dimension
            last_step_out = rnn_out.gather(t_dim, time_idx).squeeze(t_dim)
            return last_step_out
        else:
            return rnn_out
