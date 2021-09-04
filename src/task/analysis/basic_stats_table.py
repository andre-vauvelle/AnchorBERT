import pandas as pd
import numpy as np
import os

from definitions import DATA_DIR
from omni.common import load_pickle

global_params = {
    'use_code': 'code',  # 'phecode'
    'with_codes': 'all',
    'max_len_seq': 256,
    'inverse_normal_rank_cols': ['logreg_anchor', 'bert_anchor'],  # None to activate for all cols
    'anchor_cols': ['logreg', 'bert'],
    'case_noise': 0,
    'control_noise': 0,
}
file_config = {
    'phe_vocab': os.path.join(DATA_DIR, 'processed', global_params['with_codes'], 'phecode_vocab.pkl'),
    # vocabulary idx2token, token2idx
    'code_vocab': os.path.join(DATA_DIR, 'processed', global_params['with_codes'], 'code_vocab.pkl'),
    # vocabulary idx2token, token2idx
    'age_vocab': os.path.join(DATA_DIR, 'processed', global_params['with_codes'], 'age_vocab.pkl'),
    # vocabulary idx2token, token2idx
    'train_data': os.path.join(DATA_DIR, 'processed', global_params['with_codes'], 'MLM', 'phe_train.parquet'),
    'val_data': os.path.join(DATA_DIR, 'processed', global_params['with_codes'], 'MLM', 'phe_val.parquet'),
    'test_data': os.path.join(DATA_DIR, 'processed', global_params['with_codes'], 'MLM', 'phe_test.parquet'),
}
code_vocab = load_pickle(file_config['code_vocab'])
age_vocab = load_pickle(file_config['age_vocab'])
phe_vocab = load_pickle(file_config['phe_vocab'])

train_data = pd.read_parquet(file_config['train_data'])
val_data = pd.read_parquet(file_config['val_data'])
test_data = pd.read_parquet(file_config['test_data'])

data = pd.concat([train_data, val_data, test_data], axis=0)

events = data.phecode.explode()
events_nosymb = events[~events.isin(['MASK', 'SEP', 'PAD', 'CLS', 'UNK'])]
events_nomask = events[events != 'MASK']
events_sep = events[events == "SEP"]


def drop_consecutive_duplicates(a):
    ar = a.values
    return a[np.concatenate(([True], ar[:-1] != ar[1:]))]


total_patients = data.shape[0]
total_visits = (drop_consecutive_duplicates(events_nomask) == 'SEP').sum()
total_events = events_nosymb.shape[0]
total_unique_codes = len(code_vocab['token2idx'])
total_unique_events = events.unique().shape[0]
vists_per_patient = (events == 'SEP').sum() / total_patients
average_events_per_patient = total_events / total_patients
events_per_visit = total_events / (events == 'SEP').sum()
# max_number_of_events = events_nosymb.groupby(level=0).count().max()

# check
assert round(events_per_visit * vists_per_patient * total_patients) == total_events
assert average_events_per_patient * total_patients == total_events

print(
    total_patients,
    total_visits,
    vists_per_patient,
    average_events_per_patient,

    total_events,
    events_per_visit,
    total_unique_codes,
    total_unique_events
)
