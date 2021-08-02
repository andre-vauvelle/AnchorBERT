import os

import numpy as np
import torch
import pytorch_lightning as pl
from sklearn import preprocessing
from torch.utils.data import DataLoader
import torchmetrics
import re
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression

import matplotlib.pyplot as plt

# from pl_bolts.models.regression import LogisticRegression
from definitions import MODEL_DIR
from omni.common import save_pickle


class LogisticRegressionModel(pl.LightningModule):
    """
    Anchor variable model from Halpern et al., https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4419996/
    Uses anchor variable A, which is trial_n postive anchor for trial_n latent variable Y_i if it is an anchor for Y_i and
    P(Y_i=1|A=1)=1

    Method:
    1. Learn calibrated classifier to predict P(A|\tilde{X}), where \tilde{X} is censored from A
    2. Using validation set compute C = 1/|\mathcal{P}| \sum P(A=1|\tilde{X}), where \mathcal{P} is the validation data
    set where A=1
    3. For trial_n previously unseen patient t, predict:
        - P(A=1|\tilde{X})/C    if A(t) = 0
        - 1                     if A(t) = 1
    """

    def __init__(self, token, token2idx, token_type='phecode', batch_size=1_000, num_workers=0, verbose=False,
                 symbol_idxs=[0, 1, 2], emission_size=5,
                 checkpoint_path=os.path.join(MODEL_DIR, 'sklearn_regression', 'default.pkl'),
                 C=1):
        super().__init__()
        if token2idx is None:
            self.token_idxs = token
            self.symbol_idxs = symbol_idxs
            self.total_input_dim = emission_size + len(symbol_idxs)
            self.features = emission_size - len(token)
        else:
            keys = [k for k in token2idx.keys() if re.match(token, k) is not None]
            self.token_idxs = [token2idx[k] for k in keys]
            self.symbol_idxs = [token2idx[k] for k in
                                ['PAD', 'MASK', 'CLS', 'UNK']]  # not including SEP as we can use as feature
            self.total_input_dim = len(token2idx.keys())
            self.features = self.total_input_dim - len(self.token_idxs) - len(self.symbol_idxs)

        self.token2idx = token2idx
        self.token_type = token_type
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.model = LogisticRegression(penalty='l2', C=C, solver='lbfgs', max_iter=1000,
                                        verbose=verbose)
        self.checkpoint_path = checkpoint_path

    def forward(self, x_censored):
        predictions = self.model.predict_proba(x_censored)[:, 1]
        return predictions

    def training_step(self, batch, batch_idx):
        pass

    def get_anchors_censored(self, x):
        """Gets the counts of non-anchor terms for x_censored and trial_n binary variable for the anchor, y"""
        x_one_hot = torch.nn.functional.one_hot(x.to(torch.int64), self.total_input_dim)
        feature_idx = list(set(range(self.total_input_dim)) - set(self.token_idxs) - set(self.symbol_idxs))
        x_one_hot_censored = x_one_hot[:, :, feature_idx]
        x_censored = x_one_hot_censored.sum(axis=1)
        y_anchor = (x_one_hot[:, :, self.token_idxs] > 0).any(axis=1)
        y_anchor = y_anchor.any(-1).long()  # incase of multiple token_idxs
        return x_censored, y_anchor

    def sklearn_dataload(self, dataloader):
        x_censored = torch.zeros((len(dataloader.dataset), self.features))
        y_anchor = torch.zeros(len(dataloader.dataset))
        for i, batch in tqdm(enumerate(dataloader), desc='anchor sklearn dataload'):
            token_idx, age_idx, position, segment, phecode_idx = batch
            idxs = phecode_idx if self.token_type == 'phecode' else token_idx
            x_censored_batch, y_anchor_batch = self.get_anchors_censored(idxs)
            x_censored[i * dataloader.batch_size:(i + 1) * dataloader.batch_size] = x_censored_batch
            y_anchor[i * dataloader.batch_size:(i + 1) * dataloader.batch_size] = y_anchor_batch

        return x_censored, y_anchor

    def predict_sklearn(self, dataloader):
        x_censored, y_anchor = self.sklearn_dataload(dataloader)
        scaler = preprocessing.StandardScaler().fit(x_censored)
        x_censored_norm = scaler.transform(x_censored)
        self.model = self.model.fit(x_censored_norm, y_anchor)
        predictions = self(x_censored_norm)
        predictions = torch.from_numpy(predictions).float()
        return predictions, y_anchor

    def predict(self, dataset, val_dataset=None):
        dataloader = DataLoader(dataset=dataset, batch_size=self.batch_size, shuffle=False,
                                num_workers=self.num_workers) # no shuffle needed as already done in preprocessing pipeline

        predictions, y_anchor = self.predict_sklearn(dataloader)

        if self.checkpoint_path is not None:
            save_pickle(self.model, filename=self.checkpoint_path)

        # Metrics
        print("average_precision: {}".format(torchmetrics.functional.average_precision(predictions, y_anchor, pos_label=1)))
        print("auroc: {}".format(torchmetrics.functional.auroc(predictions, y_anchor.int(), pos_label=1)))
        if val_dataset is not None:
            val_dataloader = DataLoader(dataset=val_dataset, batch_size=self.batch_size, shuffle=False,
                                        num_workers=self.num_workers)
            val_x_censored, val_y_anchor = self.sklearn_dataload(val_dataloader)
            scaler = preprocessing.StandardScaler().fit(val_x_censored)
            val_x_censored_norm = scaler.transform(val_x_censored)
            val_predictions = self(val_x_censored_norm)
            val_predictions = torch.from_numpy(val_predictions).float()
            print("val_average_precision: {}".format(torchmetrics.functional.average_precision(val_predictions, val_y_anchor, pos_label=1)))
            print("val_auroc: {}".format(torchmetrics.functional.auroc(val_predictions, val_y_anchor.int(), pos_label=1)))

        return predictions

# if __name__ == '__main__':
#     plt.boxplot(x_censored_norm[np.array(y_anchor.to(bool))])
