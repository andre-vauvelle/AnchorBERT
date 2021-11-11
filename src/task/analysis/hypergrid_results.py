# Get results of hyperparameter optimizatiom from mongo db
import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pymongo
from pymongo import MongoClient
import pprint
import argparse

from definitions import RESULTS_DIR

plt.rcParams['figure.figsize'] = [9.0, 8.0]
plt.rcParams['figure.dpi'] = 100
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
plt.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
plt.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'
plt.rcParams['font.size'] = 10

args



if __name__ == '__main__':
    client = MongoClient('bigtop', 27017)
    db = client.experiments

    data = {"411.2":
                {"name": "Myocardial Infarction"},
            "250.2":
                {"name": "Type 2 Diabetes"},
            "428.2":
                {"name": "Heart Failure"},
            "290.1":
                {"name": "Dementia"},
            "714.0|714.1":
                {"name": "Rheumatoid Arthritis"}
            }
    noise_type = 'case_noise'
    # noise_type='control_noise'

    query_string = {"config.global_params.control_noise": {"$eq": 0},
                    "config.global_params.case_noise": {"$eq": 0},
                    "config.file_config.phenofile_name": {"$not": {"$regex": ".*debug"}},
                    "result": {"$ne": "null"}}

    projections_string = {"result": 1, "config": 1, "start_time": 1}

    results = list(db.runs.find(query_string, projections_string))

    store = []
    for r in results:
        try:
            store.append({
                "_id": r['_id'],
                "token": str(r['config']['target_token']),
                "start_time": r['start_time'],
                'lr': r['config']['bert_config']['optim_config']['lr'],
                'hidden_size': r['config']['bert_config']['model_config']['hidden_size'],
                'num_hidden_layers': r['config']['bert_config']['model_config']['num_hidden_layers'],
                'num_attention_heads': r['config']['bert_config']['model_config']['num_attention_heads'],
                'intermediate_size': r['config']['bert_config']['model_config']['intermediate_size'],
                "val_loss": r['result']['val_loss'],
                "val_AveragePrecision": r['result']['val_AveragePrecision'],
                "val_AUROC": r['result']['val_AUROC'],
                "test_AveragePrecision": r['result']['test_AveragePrecision'],
                "test_AUROC": r['result']['test_AUROC']
            })
        except:
            pass

    df = pd.DataFrame(store)
    df.sort_values(by=['val_AveragePrecision', '_id', 'token'], inplace=True)

    hparams = ['lr', 'intermediate_size', 'hidden_size', 'num_hidden_layers', 'intermediate_size']
    # explore hyperparameters
    for h in hparams:
        df.boxplot(column=['val_AveragePrecision'],
                   by=h)
        plt.show()

    groups = list(df.groupby('token'))
    for t, df_g in groups:
        name = data[str(t)]['name']
        noise = df_g[noise_type]

        plt.plot(noise, df_g.logreg_valid_ap, label='LR', marker='x', linestyle='-.')
        plt.plot(noise, df_g.bert_valid_ap, label='BERT', marker='o')

        # plt.plot(noise, df_g.logreg_valid_auroc, label='LR', marker='x', linestyle='-.')
        # plt.plot(noise, df_g.bert_valid_auroc, label='BERT', marker='o')

        plt.ylabel('AUPRC')
        plt.xlabel('Proportion of corrupted controls')
        plt.xticks(np.arange(0, 1, 0.1))
        plt.title(name)
        plt.legend(frameon=False, prop={'size': 18})
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, 'noise_results_plots', '{}_{}.png'.format(noise_type, t)))
        plt.show()
