import os

import numpy as np
import sklearn
import torch
import pytorch_lightning as pl
from sklearn import preprocessing, model_selection
from torch.utils.data import DataLoader, ConcatDataset
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
                 C=1, param_grid=None):
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
        self.model = LogisticRegression(penalty='l2', C=C, solver='lbfgs', max_iter=1000, verbose=verbose)
        self.param_grid = param_grid
        self.checkpoint_path = checkpoint_path
        self.verbose = verbose

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
        """
        gets data for sklearn model from pytorch dataloader
        :param dataloader:
        :return: x_censored: a numpy array of code couts, y_anchor: binary indicator of phecode presence, label: a noised version of y_anchor
        """
        x_censored = torch.zeros((len(dataloader.dataset), self.features))
        y_anchor = torch.zeros(len(dataloader.dataset))
        label = torch.zeros(len(dataloader.dataset))
        for i, batch in tqdm(enumerate(dataloader), desc='anchor sklearn dataload'):
            token_idx, age_idx, position, segment, phecode_idx, label_batch = batch
            idxs = phecode_idx if self.token_type == 'phecode' else token_idx
            x_censored_batch, y_anchor_batch = self.get_anchors_censored(idxs)
            x_censored[i * dataloader.batch_size:(i + 1) * dataloader.batch_size] = x_censored_batch
            label[i * dataloader.batch_size:(i + 1) * dataloader.batch_size] = label_batch.flatten()
            y_anchor[i * dataloader.batch_size:(i + 1) * dataloader.batch_size] = y_anchor_batch.flatten()

        return x_censored, y_anchor, label

    def fit_sklearn(self, dataloader):

        x_censored, y_anchor, label = self.sklearn_dataload(dataloader)
        scaler = preprocessing.StandardScaler().fit(x_censored)
        x_censored_norm = scaler.transform(x_censored)

        if self.param_grid is not None:
            train_samples = dataloader.dataset.datasets[0]
            test_samples = dataloader.dataset.datasets[1]

            train_indices = np.arange(len(train_samples))
            test_indices = np.arange(len(train_samples), len(train_samples) + len(test_samples))
            cv = [(train_indices, test_indices)]
            cv_model = sklearn.model_selection.GridSearchCV(estimator=self.model, param_grid=self.param_grid,
                                                            refit=False,
                                                            cv=cv,
                                                            scoring='average_precision',
                                                            n_jobs=1 if self.num_workers == 0 else self.num_workers)
            cv_model = cv_model.fit(x_censored_norm, label)
            self.model = LogisticRegression(penalty='l2', C=cv_model.best_params_['C'], solver='lbfgs', max_iter=1000,
                                            verbose=self.verbose)
        self.model.fit(x_censored_norm, label)

    def predict_sklearn(self, dataloader):
        x_censored, y_anchor, label = self.sklearn_dataload(dataloader)
        scaler = preprocessing.StandardScaler().fit(x_censored)
        x_censored_norm = scaler.transform(x_censored)
        predictions = self(x_censored_norm)
        predictions = torch.from_numpy(predictions).float()
        return predictions, y_anchor, label

    def predict(self, dataset, val_dataset=None, test_dataset=None):

        dataloader = DataLoader(dataset=dataset, batch_size=self.batch_size, shuffle=False,
                                num_workers=self.num_workers)  # no shuffle needed as already done in preprocessing pipeline

        if val_dataset is not None:
            full_dataset = ConcatDataset((dataset, val_dataset))
            full_dataloader = DataLoader(dataset=full_dataset, batch_size=self.batch_size, shuffle=False,
                                         num_workers=self.num_workers)  # no shuffle needed as already done in preprocessing pipeline
            self.fit_sklearn(full_dataloader)
        else:
            self.fit_sklearn(dataloader)

        predictions, y_anchor, label = self.predict_sklearn(dataloader)

        if self.checkpoint_path is not None:
            save_pickle(self.model, filename=self.checkpoint_path)

        # Metrics
        auprc = torchmetrics.functional.average_precision(predictions, y_anchor, pos_label=1)
        print("average_precision: {}".format(auprc))
        auroc = torchmetrics.functional.auroc(predictions, y_anchor.int(), pos_label=1)
        print("auroc: {}".format(auroc))
        if val_dataset is not None:
            val_dataloader = DataLoader(dataset=val_dataset, batch_size=self.batch_size, shuffle=False,
                                        num_workers=self.num_workers)
            val_x_censored, val_y_anchor, val_label = self.sklearn_dataload(val_dataloader)
            scaler = preprocessing.StandardScaler().fit(val_x_censored)
            val_x_censored_norm = scaler.transform(val_x_censored)
            val_predictions = self(val_x_censored_norm)
            val_predictions = torch.from_numpy(val_predictions).float()
            val_auprc = torchmetrics.functional.average_precision(val_predictions, val_y_anchor, pos_label=1)
            print("val_average_precision: {}".format(val_auprc))
            val_auroc = torchmetrics.functional.auroc(val_predictions, val_y_anchor.int(), pos_label=1)
            print("val_auroc: {}".format(val_auroc))
            predictions = torch.cat((predictions, val_predictions), dim=0)
        else:
            val_auroc = 0
            val_auprc = 0
        if test_dataset is not None:
            test_dataloader = DataLoader(dataset=test_dataset, batch_size=self.batch_size, shuffle=False,
                                         num_workers=self.num_workers)
            test_x_censored, test_y_anchor, test_label = self.sklearn_dataload(test_dataloader)
            scaler = preprocessing.StandardScaler().fit(test_x_censored)
            test_x_censored_norm = scaler.transform(test_x_censored)
            test_predictions = self(test_x_censored_norm)
            test_predictions = torch.from_numpy(test_predictions).float()
            test_auprc = torchmetrics.functional.average_precision(test_predictions, test_y_anchor, pos_label=1)
            print("test_average_precision: {}".format(test_auprc))
            test_auroc = torchmetrics.functional.auroc(test_predictions, test_y_anchor.int(), pos_label=1)
            print("test_auroc: {}".format(test_auroc))
            predictions = torch.cat((predictions, test_predictions), dim=0)
        else:
            test_auroc = 0
            test_auprc = 0

        metrics = {
            "average_precision": auprc,
            "auroc": auroc,
            "val_average_precision": val_auprc,
            "val_auroc": val_auroc,
            "test_average_precision": test_auprc,
            "test_auroc": test_auroc,
            "C": self.model.get_params()['C']
        }

        return predictions, metrics

# if __name__ == '__main__':
#     plt.boxplot(x_censored_norm[np.array(y_anchor.to(bool))])
