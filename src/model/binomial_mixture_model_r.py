import subprocess
import tempfile
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from model.utils import flatten_generated_data

import re
import pytorch_lightning as pl


class BinomialMixtureModelR(pl.LightningModule):
    """
    Binomial Mixture Model From R PheProb package
    """

    def __init__(self, token, token2idx, token_type='phecode', verbose=False, num_workers=0, sep_symbol_idx=1):
        super().__init__()
        self.verbose = verbose
        if token2idx is None:
            self.token_idxs = token
            self.symbol_idxs = [sep_symbol_idx]
        else:
            keys = [k for k in token2idx.keys() if re.match(token, k) is not None]
            self.token_idxs = [token2idx[k] for k in keys]
            self.symbol_idxs = [token2idx[k] for k in ['PAD', 'CLS', 'UNK', 'SEP']]
        self.token2idx = token2idx
        self.token_type = token_type
        self.batch_size = 50_000
        self.num_workers = num_workers

    def forward(self, x):
        counts = torch.stack([x == y for y in self.token_idxs]).sum(0).sum(axis=1)
        total_codes = torch.stack([y != x for y in self.symbol_idxs]).sum(0).sum(axis=1)

        df = pd.DataFrame({'S': counts, 'C': total_codes})

        with tempfile.NamedTemporaryFile() as temp_input, tempfile.NamedTemporaryFile() as temp_output:
            df.to_csv(temp_input.name, index=False)
            verbose_str = '> /dev/null' if not self.verbose else ''
            eval_string = """
source /share/apps/source_files/R/R-4.0.3.source \n \
R -e \'\
library(pheprob)\n\
df <- read.csv("{}")\n\
predictions <- pheprob.pred(df$S, df$C, yes.concom = TRUE, S.new=NULL, C.new=NULL)\n\
write.csv(predictions, "{}")\'{}""".format(temp_input.name, temp_output.name, verbose_str)
            subprocess.call(eval_string, shell=True)
            predictions = pd.read_csv(temp_output.name)
        return torch.from_numpy(predictions.loc[:, 'pheprob.pred'].values)

    def predict(self, dataset, val_dataset=None):
        dataloader = DataLoader(dataset=dataset, batch_size=self.batch_size, shuffle=False,
                                num_workers=self.num_workers)
        x = torch.zeros((len(dataset), dataset.max_len))
        for i, batch in tqdm(enumerate(dataloader)):
            token_idx, age_idx, position, segment, phecode_idx = batch
            if self.token_type == 'phecode':
                x[i * dataloader.batch_size:(i + 1) * dataloader.batch_size] = phecode_idx
            else:
                x[i * dataloader.batch_size:(i + 1) * dataloader.batch_size] = token_idx
        predictions = self(x)
        return predictions
