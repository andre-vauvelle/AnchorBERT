import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import torch

import torch.functional as f

# import seaborn as sns
# code = '411.2'
from definitions import RESULTS_DIR
from model.metrics import inverse_normal_rank

# name = 'bert_250.2.tsv'
# name = 'all_250.2.tsv'
# df = pd.read_csv('/SAN/ihibiobank/denaxaslab/andre/UKBB/data/processed/pheprob/{}'.format(name), sep='\t')
# df_a = pd.read_csv('/SAN/ihibiobank/denaxaslab/andre/UKBB/data/processed/pheprob/all_anchor_714.0|714.1.tsv', sep='\t')
# df = pd.read_csv('/SAN/ihibiobank/denaxaslab/andre/UKBB/data/processed/pheprob/bert_threshold1_250.2.tsv', sep='\t')

# import numpy as np


def get_diagram_data(y, p, n_bins):
    n_bins = float(n_bins)  # a float to take care of division

    # we'll append because some bins might be empty
    mean_predicted_values = np.empty((0,))
    true_fractions = np.zeros((0,))

    for b in range(1, int(n_bins) + 1):
        i = np.logical_and(p <= b / n_bins, p > (b - 1) / n_bins)  # indexes for p in the current bin

        # skip bin if empty
        if np.sum(i) == 0:
            continue

        mean_predicted_value = np.mean(p[i])
        # print "***", np.sum( y[i] ), np.sum( i )
        true_fraction = np.sum(y[i]) / np.sum(i)  # y are 0/1; i are logical and evaluate to 0/1

        mean_predicted_values = np.hstack((mean_predicted_values, mean_predicted_value))
        true_fractions = np.hstack((true_fractions, true_fraction))

    return mean_predicted_values, true_fractions


def plot_calibration(predictions, y, n_bins=10, save=True, save_path=None):
    if save_path is None:
        save_path = os.path.join(os.path.join(RESULTS_DIR, 'pheno_results', 'calibration', 'default.png'))
    df = pd.DataFrame({'predictions': predictions.numpy(), 'y_anchor': y.numpy()})
    df.predictions.hist()
    plt.show()

    mean_predicted_values, true_fractions = get_diagram_data(df.y_anchor, df.predictions, n_bins=n_bins)
    fig, ax = plt.subplots()
    ax.plot(np.arange(0, 1.1, step=0.1), np.arange(0, 1.1, step=0.1))
    ax.plot(mean_predicted_values, true_fractions)
    ax.set_xlabel('Mean prediction values')
    ax.set_ylabel('True Fractions')
    plt.show()
    if save:
        plt.savefig(save_path)
