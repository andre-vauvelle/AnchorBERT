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

    query_string = {}

    if noise_type == 'control_noise':
        query_string = {"config.global_params.control_noise": {"$in": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]},
                        "config.global_params.case_noise": {"$eq": 0},
                        "result": {"$ne": "null"}}
    elif noise_type == 'case_noise':
        query_string = {
            "config.global_params.case_noise": {"$in": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]},
            "config.global_params.control_noise": {"$eq": 0},
            "result": {"$ne": "null"}}
    elif noise_type == 'both_noise':
        query_string = {
            "config.global_params.case_noise": {"$in": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]},
            "config.global_params.control_noise": {"$in": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]},
            "result": {"$ne": "null"}}

    projections_string = {"result": 1, "config": 1, "start_time": 1}

    results = db.runs.find(query_string, projections_string)
    store = []
    for r in results:
        params = r['config']['global_params']
        result = r['result']

        if result is None:
            continue
        bert_result = result['bert']
        logreg_result = result['logreg']

        bert_result.update(params)
        logreg_result.update(params)
        bert_result.update({"id": r['_id']})
        logreg_result.update({"id": r['_id']})
        bert_result.update({"start_time": r['start_time']})
        logreg_result.update({"start_time": r['start_time']})
        bert_result.update({"target_token": r['config']['target_token']})
        logreg_result.update({"target_token": r['config']['target_token']})

        bert_result.update({"method": 'bert'})
        logreg_result.update({"method": 'logreg'})

        store.append(bert_result)
        store.append(logreg_result)

    df_results = pd.DataFrame(store)
    df_results = df_results[df_results.target_token == token]
    df_results = df_results.sort_values('start_time').drop_duplicates(
        ['control_noise', 'case_noise', 'method'], keep='last')
    # # df_results = df_results.iloc[6:, :]
    # df_results = df_results[df_results.val_average_precision.apply(lambda x: isinstance(x, float))]
    # df_results = df_results[df_results.val_average_precision > 0.3]
    # df_results = df_results[df_results.id >= 445]

    for method, df in df_results.groupby('method'):
        noise_type_col = noise_type if noise_type != 'both_noise' else 'control_noise'
        df.sort_values(noise_type_col, inplace=True, ascending=False)
        plt.plot(df[noise_type_col], df[metric_col], label=method)

    plt.legend()
    plt.ylabel(metric_col)
    plt.xlabel('{}'.format(noise_type))
    plt.title('{}'.format(phenotype_name))
    plt.show()
