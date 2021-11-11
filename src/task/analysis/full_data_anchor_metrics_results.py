# Get anchor metrics from sacred mongo db for Table 3
import pandas as pd
import matplotlib.pyplot as plt
import pymongo
from pymongo import MongoClient
import pprint

if __name__ == '__main__':
    client = MongoClient('bigtop', 27017)
    db = client.experiments

    query_string = {"config.global_params.control_noise": {"$eq": 0},
                    "config.global_params.case_noise": {"$eq": 0},
                    "config.file_config.phenofile_name": {"$not": {"$regex": ".*debug"}},
                    "result": {"$ne": "null"}}

    projections_string = {"result": 1, "config": 1, "start_time": 1}

    results = list(db.runs.find(query_string, projections_string))

    store = []
    for r in results:
        store.append({
            "_id": r['_id'],
            "token": r['config']['target_token'],
            "start_time": r['start_time'],
            "bert_train_auroc": r["result"]['bert']['train_AUROC'],
            "bert_train_ap": r["result"]['bert']['train_AveragePrecision'],
            "bert_valid_auroc": r["result"]['bert']['val_AUROC'],
            "bert_valid_ap": r["result"]['bert']['val_AveragePrecision'],
            "bert_test_auroc": r["result"]['bert']['test_AUROC'],
            "bert_test_ap": r["result"]['bert']['test_AveragePrecision'],
            "logreg_train_auroc": r["result"]['logreg']['auroc'],
            "logreg_train_ap": r["result"]['logreg']['average_precision'],
            "logreg_valid_auroc": r["result"]['logreg']['val_auroc'],
            "logreg_valid_ap": r["result"]['logreg']['val_average_precision'],
            "logreg_test_auroc": r["result"]['logreg']['test_auroc'],
            "logreg_test_ap": r["result"]['logreg']['test_average_precision'],
        })

    df = pd.DataFrame(store)
    df = df.iloc[1:, ]
    df.sort_values(by='bert_valid_ap', inplace=True, ascending=False)
    table = df.loc[:, ['token', 'bert_test_auroc', 'bert_test_ap', 'logreg_test_auroc', 'logreg_test_ap']].groupby(
        by=['token']).agg({
        'bert_test_auroc': ['mean', 'std'],
        'bert_test_ap': ['mean', 'std'],
        'logreg_test_auroc': ['mean', 'std'],
        'logreg_test_ap': ['mean', 'std'],
    })

    print(table)
