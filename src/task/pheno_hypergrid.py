import os

import numpy as np

import hyperopt
import torch
from hyperopt import fmin, hp
from hyperopt.mongoexp import MongoTrials
from hyperopt.pyll import scope

from definitions import MODEL_DIR
from omni.common import load_pickle
from task.pheno import hyperopt_objective

import warnings
from sklearn.model_selection import ParameterGrid

import pandas as pd

DEBUG_STRING = 'debug' if __debug__ else ''

if __name__ == '__main__':
    # target_token = '714.0|714.1'
    target_token = '714.0|714.1'

    params = {
        'lr': [1e-3, 1e-4, 1e-5],
        'hidden_size': [240, 360],  # hidden size must be multiple of num_attention_heads
        'num_hidden_layers': [6, 10],
        'num_attention_heads': [12],  # number of attention heads
        'intermediate_size': [128, 256]
    }

    param_grid = ParameterGrid(params)

    store = []
    for s in param_grid:
        store.append(s)

    df = pd.DataFrame(store)

    df = df.sample(frac=1).reset_index(drop=True)
    df.to_csv('/home/vauvelle/pycharm-sftp/pheprob/src/jobs/configs/anchorbert_paramgrid-{}.csv'.format(df.shape[0]),
              index=False)
