import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pymongo
from pymongo import MongoClient
import pprint

from definitions import RESULTS_DIR

plt.rcParams['figure.figsize'] = [9.0, 8.0]
plt.rcParams['figure.dpi'] = 200
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
plt.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
plt.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'
plt.rcParams['font.size'] = 22


if __name__ == '__main__':
    client = MongoClient('bigtop', 27017)
    db = client.experiments

    token = '411.2'
    token = '250.2'
    token = '428.2'
    # noise_type = 'control_noise'
    # noise_type = 'case_noise'
    # noise_type = 'both_noise'
    # metric_col = 'val_average_precision'
    metric_col = 'val_auroc'

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
    phenotype_name = data[token]['name']

    query_string = {"config.global_params.control_noise": {"$gte": 0},
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
                "control_noise": r['config']['global_params']['control_noise'],
                "case_noise": r['config']['global_params']['case_noise'],
                "bert_train_auroc": r["result"]['bert']['auroc'],
                "bert_train_ap": r["result"]['bert']['average_precision'],
                "bert_valid_auroc": r["result"]['bert']['val_auroc'],
                "bert_valid_ap": r["result"]['bert']['val_average_precision'],
                "logreg_train_auroc": r["result"]['logreg']['auroc'],
                "logreg_train_ap": r["result"]['logreg']['average_precision'],
                "logreg_valid_auroc": r["result"]['logreg']['val_auroc'],
                "logreg_valid_ap": r["result"]['logreg']['val_average_precision'],
            })
        except:
            pass