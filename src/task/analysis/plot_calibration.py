import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

import sklearn
import torch

import torch.functional as f

# import seaborn as sns
# code = '411.2'
import torchmetrics

from definitions import RESULTS_DIR
from model.metrics import apply_inverse_normal_rank


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


def plot_calibration(predictions, y, n_bins=10, save=True, save_path=None, title=''):
    if save_path is None:
        save_path = os.path.join(os.path.join(RESULTS_DIR, 'pheno_results', 'calibration', 'default.png'))
    df = pd.DataFrame({'predictions': predictions, 'y_anchor': y})
    df.predictions.hist()
    ax = plt.gca()
    ax.set_title(title)
    plt.show()

    mean_predicted_values, true_fractions = get_diagram_data(df.y_anchor, df.predictions, n_bins=n_bins)
    fig, ax = plt.subplots()
    ax.plot(np.arange(0, 1.1, step=0.1), np.arange(0, 1.1, step=0.1))
    ax.plot(mean_predicted_values, true_fractions)
    ax.set_xlabel('Mean prediction values')
    ax.set_ylabel('True Fractions')
    ax.set_title(title)
    plt.show()
    if save:
        plt.savefig(save_path)


if __name__ == '__main__':
    data_dict = {
        "411.2": {
            "data_path": '/SAN/ihibiobank/denaxaslab/andre/UKBB/data/processed/phenotypes/411.2.tsv',
            "name": 'Myocardial Infarction'},
        "428.2": {
            "data_path": '/SAN/ihibiobank/denaxaslab/andre/UKBB/data/processed/phenotypes/428.2.tsv',
            "name": 'Heart Failure'},
        "250.2": {
            "data_path": '/SAN/ihibiobank/denaxaslab/andre/UKBB/data/processed/phenotypes/bert_binomial_r_logreg_threshold1_threshold2_threshold3_250.2.tsv',
            "name": 'Type 2 Diabetes'},
        "714": {
            "data_path": '/SAN/ihibiobank/denaxaslab/andre/UKBB/data/processed/phenotypes/714.0|714.1.tsv',
            "name": 'Rheumatoid Arthritis'
        }}

    code = '714'
    df = pd.read_csv(data_dict[code]['data_path'], sep='\t')
    phenotype_name = data_dict[code]['name']

    plot_calibration(df.bert, df.threshold1, save=False, n_bins=100, title='bert {}'.format(phenotype_name))
    plot_calibration(df.logreg, df.threshold1, save=False, n_bins=100, title='logreg {}'.format(phenotype_name))

    plot_calibration(df.bert, df.threshold1, save=False, n_bins=10, title='bert {}'.format(phenotype_name))
    plot_calibration(df.logreg, df.threshold1, save=False, n_bins=10, title='logreg {}'.format(phenotype_name))

    tn, fp, fn, tp = sklearn.metrics.confusion_matrix(df.threshold1, df.bert.round()).ravel()
    print('BERT   tn', tn, 'fp', fp, 'fn', fn, 'tp', tp, 'fdr', fp / (fp + tp), 'precision', tp / (tp + fp), 'npv',
          tn / (tn + fn),
          'fpr', fp / (fp + tn), 'fnr', fn / (fn + tp), 'for', fn / (fn + tn), 'specificity', tn / (tn + fp),
          'sensitivity',
          tp / (tp + fn))
    tn, fp, fn, tp = sklearn.metrics.confusion_matrix(df.threshold1, df.logreg.round()).ravel()
    print('LOGREG tn', tn, 'fp', fp, 'fn', fn, 'tp', tp, 'fdr', fp / (fp + tp), 'precision', tp / (tp + fp), 'npv',
          tn / (tn + fn),
          'fpr', fp / (fp + tn), 'fnr', fn / (fn + tp), 'for', fn / (fn + tn), 'specificity', tn / (tn + fp),
          'sensitivity',
          tp / (tp + fn))

    # auprc
    data = {}

    y_true = df.threshold1

    y_pred = df.bert
    average_precision = torchmetrics.functional.average_precision(preds=torch.Tensor(df.bert),
                                                                  target=torch.Tensor(df.threshold1).int(),
                                                                  pos_label=1)
    precision, recall, _ = sklearn.metrics.precision_recall_curve(y_true, y_pred, pos_label=1)
    data.update({'bert': {
        'precision': precision,
        'recall': recall,
        'roc_auc': average_precision}})

    y_pred = df.logreg
    average_precision = torchmetrics.functional.average_precision(preds=torch.Tensor(df.logreg),
                                                                  target=torch.Tensor(df.threshold1).int(),
                                                                  pos_label=1)
    precision, recall, _ = sklearn.metrics.precision_recall_curve(y_true, y_pred, pos_label=1)
    data.update({'logreg': {
        'precision': precision,
        'recall': recall,
        'roc_auc': average_precision}})

    plt.figure()
    lw = 2
    plt.plot(data['bert']['precision'], data['bert']['recall'], color='c',
             lw=lw, label='bert (average precision= %0.2f)' % data['bert']['roc_auc'])
    plt.plot(data['logreg']['precision'], data['logreg']['recall'], color='r',
             lw=lw, label='logreg (average precision= %0.2f)' % data['logreg']['roc_auc'])
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('recall')
    plt.ylabel('precision')
    plt.title('Precision Recall Curve {}'.format(phenotype_name))
    plt.legend(loc="lower left")
    plt.show()

    # auroc
    data = {}

    y_true = df.threshold1

    x_offset = 0.05
    y_offset = -0.05
    arrowprops = {'arrowstyle': "->"}

    y_pred = df.bert
    fpr, tpr, thresholds = sklearn.metrics.roc_curve(y_true, y_pred)
    roc_auc = sklearn.metrics.auc(fpr, tpr)
    data.update({'bert': {
        'fpr': fpr,
        'tpr': tpr,
        'roc_auc': roc_auc,
        'thresholds': thresholds,
        '0.5_t': thresholds[thresholds > 0.5].min(),
        '0.5': (fpr[thresholds > 0.5].max(), tpr[thresholds > 0.5].max()),
        '0.5txt': (fpr[thresholds > 0.5].max() + x_offset, tpr[thresholds > 0.5].max() + y_offset)
    }})

    y_pred = df.logreg
    fpr, tpr, thresholds = sklearn.metrics.roc_curve(y_true, y_pred)
    roc_auc = sklearn.metrics.auc(fpr, tpr)
    data.update({'logreg': {
        'fpr': fpr,
        'tpr': tpr,
        'roc_auc': roc_auc,
        'thresholds': thresholds,
        '0.5_t': thresholds[thresholds > 0.5].min(),
        '0.5': (fpr[thresholds > 0.5].max(), tpr[thresholds > 0.5].max()),
        '0.5txt': (fpr[thresholds > 0.5].max() + x_offset, tpr[thresholds > 0.5].max() + y_offset)
    }})

    plt.figure(figsize=(6, 6))
    lw = 2

    plt.plot(data['bert']['fpr'], data['bert']['tpr'], color='c',
             lw=lw, label='bert roc curve (area = %0.2f)' % data['bert']['roc_auc'])
    plt.annotate("{0:.1f} bert Threshold".format(data['bert']['0.5_t']), data['bert']['0.5'], data['bert']['0.5txt'],
                 arrowprops=arrowprops)
    plt.plot(data['logreg']['fpr'], data['logreg']['tpr'], color='r',
             lw=lw, label='logreg roc curve (area = %0.2f)' % data['logreg']['roc_auc'])
    plt.annotate("{0:.1f} logreg Threshold".format(data['logreg']['0.5_t']), data['logreg']['0.5'],
                 data['logreg']['0.5txt'], arrowprops=arrowprops)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('false positive rate')
    plt.ylabel('true positive rate')
    plt.title('ROC {}'.format(phenotype_name))
    plt.legend(loc="lower right")
    plt.gcf().set_dpi(200)
    plt.show()
