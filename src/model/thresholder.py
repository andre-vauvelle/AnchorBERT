import numpy as np
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, ConcatDataset
import re
from tqdm import tqdm


class ThresholdModel(pl.LightningModule):
    """
    Very simple thresholding method for phenotyping from Sinnott et al 2018
    """

    def __init__(self, threshold, token, token2idx, token_type='phecode', batch_size=100_000, num_workers=0,
                 label_noise=False):
        """

        :param threshold:
        :param token:
        :param token2idx: if none then assumed token is already idx
        :param token_type:
        :param batch_size:
        :param num_workers:
        """
        super().__init__()
        self.threshold = threshold
        if token2idx is None:
            self.token_idxs = token
        else:
            keys = [k for k in token2idx.keys() if re.match(token, k) is not None]
            self.token_idxs = [token2idx[k] for k in keys]
        self.token_type = token_type
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.label_noise = label_noise

    def forward(self, x):
        counts = torch.stack([x == y for y in self.token_idxs]).sum(0).sum(axis=1)
        return (counts >= self.threshold).long()

    def training_step(self, batch, batch_idx):
        pass

    def predict(self, dataset, val_dataset=None):
        if val_dataset is not None:
            dataset = ConcatDataset((dataset, val_dataset))

        predictions = torch.zeros(len(dataset))
        dataloader = DataLoader(dataset=dataset, batch_size=self.batch_size, shuffle=False,
                                num_workers=0)
        for i, batch in tqdm(enumerate(dataloader)):
            token_idx, age_idx, position, segment, phecode_idx, label = batch
            if self.token_type == 'phecode':
                x = phecode_idx
            else:
                x = token_idx
            output = self(x)
            if self.label_noise:
                output = label.flatten()
            predictions[i * self.batch_size:(i + 1) * self.batch_size] = output

        metrics = {
        }

        return predictions, metrics
