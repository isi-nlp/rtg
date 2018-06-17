import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
from tqdm import tqdm

from . import TranslationExperiment as Experiment
from . import log, device, my_tensor as tensor
from .dataprep import BatchIterable
from .len_model import LengthModel

tqdm.monitor_interval = 0


class Trainer:

    def __init__(self, exp: Experiment):
        self.exp = exp

        last_model, last_epoch = self.exp.get_last_saved_model()
        if last_model:
            self.model = torch.load(last_model)
            self.start_epoch = last_epoch + 1
            log.info(f"Resuming training from epoch:{self.start_epoch}, model={last_model}")
        else:
            self.model = LengthModel(vocab_size=exp.src_field.size() + 1).to(device)
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

        train_data = BatchIterable(self.exp.train_file, batch_size=batch_size, in_mem=True)
        val_data = BatchIterable(self.exp.valid_file, batch_size=batch_size, in_mem=True)
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

