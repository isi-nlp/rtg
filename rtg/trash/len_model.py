# NOTE: this is obsolete ; TG wrote this when he was learning to implement and train networks

import torch.optim as optim
import tqdm
from tqdm import tqdm

from rtg import TranslationExperiment as Experiment
from rtg import log
from rtg import my_tensor as tensor
from rtg.data.dataset import BatchIterable

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from rtg import device
from rtg.data.dataset import Batch, Field


class LSTMEncoder(nn.Module):
    """
    Encoder module for encoding sequences
    """

    def __init__(self, vocab_size, emb_dim=100, hid_dim=100, pad_idx=Field.pad_idx, dropout=0.4, last_step_only=True,
                 num_layers=2, num_directions=2):
        """
        :param vocab_size: size of vocabulary i.e. maximum index in the input sequence
        :param emb_dim: embedding dimension (of word vectors)
        :param hid_dim: dimension of RNN units
        :param pad_idx: input index that is used for padding
        :param dropout: dropout rate for RNN
        :param last_step_only: return the last step output only; default returns all the time steps
        :param num_layers: number of n_layers in RNN
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


class LengthModel(nn.Module):
    """
    Model to predict the length of outputs
    Why predict length?
    Maybe for fun! If a model cant predict output length, can it predict output correctly?
    maybe for learning step-by-step, i.e. practice easier or smaller problems before the bigger!
    Predicting length is easier problem than predicting the output itself.
    """

    def __init__(self, vocab_size, emb_dim=100, hid_dim=100, dropout=0.4, num_dirs=2):
        super(LengthModel, self).__init__()
        self.enc = LSTMEncoder(vocab_size, emb_dim, hid_dim, dropout=dropout, last_step_only=True,
                               num_directions=num_dirs)
        self.out = nn.Linear(num_dirs * hid_dim, 1)

    def forward(self, batch: Batch):
        last_step_out = self.enc(batch)
        out = self.out(last_step_out)
        return out.squeeze(1)       # make a vector


class Trainer:

    def __init__(self, exp: Experiment):
        self.exp = exp

        last_model, last_epoch = self.exp.get_last_saved_model()
        if last_model:
            self.model = torch.load(last_model)
            self.start_epoch = last_epoch + 1
            log.info(f"Resuming training from epoch:{self.start_epoch}, model={last_model}")
        else:
            self.model = LengthModel(vocab_size=exp.src_vocab.size() + 1).to(device)
            self.start_epoch = 0
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.05)
        self.loss_func = nn.MSELoss()

    def evaluate(self, data) -> float:
        tot_loss = 0.0
        for i, batch in tqdm(enumerate(data)):
            # Step clear gradients
            self.model.zero_grad()
            # Step Run forward pass.
            pred_len = self.model(batch)
            # Step. Compute the loss, gradients, and update the parameters by

            #  calling optimizer.step()
            loss = self.loss_func(pred_len, tensor(batch.y_len.data, dtype=torch.float))
            tot_loss += loss
        return tot_loss

    def train(self, num_epochs: int, batch_size: int, **args):
        log.info(f'Going to train for {num_epochs} epochs; batch_size={batch_size}')

        train_data = BatchIterable(self.exp.train_file, batch_size=batch_size, in_mem=True,
                                   field=self.exp.tgt_vocab)
        val_data = BatchIterable(self.exp.valid_file, batch_size=batch_size, in_mem=True,
                                 field=self.exp.tgt_vocab)
        keep_models = args.get('keep_models', 4)
        if num_epochs <= self.start_epoch:
            raise Exception(f'The model was already trained to {self.start_epoch} epochs. '
                            f'Please increase epoch or clear the existing models')
        for ep in range(self.start_epoch, num_epochs):
            for i, batch in tqdm(enumerate(train_data)):
                # Step clear gradients
                self.model.zero_grad()
                # Step Run forward pass.

                pred_len = self.model(batch)
                # Step. Compute the loss, gradients, and update the parameters by

                #  calling optimizer.step()
                loss = self.loss_func(pred_len, tensor(batch.y_len.data, dtype=torch.float))
                loss.backward()
                self.optimizer.step()

            log.info(f'Epoch {ep+1} complete.. validating...')
            score = self.evaluate(val_data)
            self.exp.store_model(epoch=ep, model=self.model, score=score, keep=keep_models)

