import pandas as pd
import matplotlib.pyplot as plt
import pymongo
from pymongo import MongoClient
import pprint

if __name__ == '__main__':
    client = MongoClient('bigtop', 27017)
    db = client.experiments

    # token = '411.2'
    # token = '250.2'
    token = '428.2'
    noise_type = 'control_noise'
    # noise_type = 'case_noise'
    # noise_type = 'both_noise'
    # metric_col = 'val_average_precision'
    metric_col = 'val_auroc'

    data = {"411.2":
                {"name": "Myocardial Infarction"},
            "250.2":
                {"name": "Type 2 Diabetes"},
            "428.2":
                {"name": "Heart Failure"}
            }
    phenotype_name = data[token]['name']

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
                "token": r['config']['target_token'],
                "start_time": r['start_time'],
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

    df = pd.DataFrame(store)
    df = df.iloc[1:, ]
    df.sort_values(by='bert_valid_ap', inplace=True, ascending=False)

    df.loc[:, ['logreg_train_auroc', 'logreg_train_ap', 'logreg_valid_auroc', 'logreg_valid_ap']]
    df.loc[:, ['bert_train_auroc', 'bert_train_ap', 'bert_valid_auroc', 'bert_valid_ap']]

    df = pd.read_csv('/SAN/ihibiobank/denaxaslab/andre/UKBB/data/processed/phenotypes/250.2_ca0_co0.tsv', sep='\t')
    df.threshold1.sum()
    round(df.threshold1.sum() / df.shape[0], 4)
    df = pd.read_csv('/SAN/ihibiobank/denaxaslab/andre/UKBB/data/processed/phenotypes/411.2_ca0_co0.tsv', sep='\t')
    df.threshold1.sum()
    round(df.threshold1.sum() / df.shape[0], 4)
    df = pd.read_csv('/SAN/ihibiobank/denaxaslab/andre/UKBB/data/processed/phenotypes/428.2_ca0_co0.tsv', sep='\t')
    df.threshold1.sum()
    round(df.threshold1.sum() / df.shape[0], 4)
    df = pd.read_csv('/SAN/ihibiobank/denaxaslab/andre/UKBB/data/processed/phenotypes/714.0|714.1_ca0_co0.tsv',
                     sep='\t')
    df.threshold1.sum()
    round(df.threshold1.sum() / df.shape[0], 4)
    df = pd.read_csv('/SAN/ihibiobank/denaxaslab/andre/UKBB/data/processed/phenotypes/290.1_ca0_co0.tsv',
                     sep='\t')
    df.threshold1.sum()
    round(df.threshold1.sum() / df.shape[0], 4)
