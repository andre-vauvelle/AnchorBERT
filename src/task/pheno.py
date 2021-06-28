import os

import pandas as pd
import torch
from torch.utils.data import DataLoader

from data.datasets import PheDataset
from model.anchor_model import AnchorModel
from model.bert.main import BERTAnchorModel
from model.binomial_mixture_model_r import BinomialMixtureModelR
from model.metrics import inverse_normal_rank
from model.thresholder import ThresholdModel
from omni.common import load_pickle

import pytorch_lightning as pl

from definitions import DATA_DIR, MODEL_DIR, MONGO_STR, MONGO_DB

from sacred import Experiment
from sacred.observers import MongoObserver
from sacred.utils import apply_backspaces_and_linefeeds

base = os.path.basename(__file__)
experiment_name = os.path.splitext(base)[0]
ex = Experiment(experiment_name)
ex.observers.append(MongoObserver(url=MONGO_STR, db_name=MONGO_DB))
ex.captured_out_filter = apply_backspaces_and_linefeeds

# Hyperlipidemia 272.1, E78.5, E78.4
# target_token = '272.1'

# Hypothyroidism all 244.x
# target_token = '244..'

# Rheumatoid arthritis 714 714.1 excluding juvenile arthritis
# target_token = '714.0|714.1'

# Diabetes 250.2

# Dementias 290.1

SYMBOLS = ['PAD',
           'MASK',
           'SEP',
           'CLS',
           'UNK', ]

# DEBUG = __debug__
DEBUG = True

N_WORKERS = 0 if DEBUG else os.cpu_count()


@ex.config
def config():
    target_token = '250.2'
    global_params = {
        'use_code': 'code',  # 'phecode'
        'with_codes': 'all',
        'max_len_seq': 256,
        'inverse_normal_rank_cols': []  # None to activate for all cols
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
        # formatted data
        'model_path': os.path.join(MODEL_DIR, global_params['with_codes'], global_params['use_code']),
        # where to save model
    }
    bert_config = {
        "optim_config": {
            'lr': 1e-4,
            'warmup_proportion': 0.1,
            'weight_decay': 0.01
        },
        "train_params": {
            'epochs': 5,
            'batch_size': 64,
            'accumulate_grad_batches': 4,
            'effective_batch_size': 256,
            'gpus': -1 if torch.cuda.is_available() else 0,
            'auto_scale_batch_size': False,
            'auto_lr_find': False,
            'val_check_interval': 0.2
        },
        "model_config": {
            # 'pretrained': os.path.join(MODEL_DIR, '60train_model_min_prop'),
            # 'pretrained': None,
            # if None then train from scratch
            'vocab_size': len(load_pickle(file_config['phe_vocab'])['token2idx'].keys()),
            # number of disease + symbols for word embedding
            'seg_vocab_size': 2,  # number of vocab for seg embedding
            'age_vocab_size': None,  # len(load_pickle(file_config['age_vocab'])['token2idx'].keys()),
            # number of vocab for age embedding
            'max_position_embedding': global_params['max_len_seq'],  # maximum number of tokens
            'hidden_size': 288,  # word embedding and seg embedding hidden size
            'hidden_dropout_prob': 0.2,  # dropout rate
            'num_hidden_layers': 6,  # number of multi-head attention layers required
            'num_attention_heads': 12,  # number of attention heads
            'attention_probs_dropout_prob': 0.22,  # multi-head attention dropout rate
            'intermediate_size': 512,
            # the size of the "intermediate" layer in the transformer encoder
            'hidden_act': 'gelu',
            # The non-linear activation function in the encoder and the pooler "gelu", 'relu', 'swish' are supported
            'initializer_range': 0.02,  # parameter weight initializer range
        },
        "feature_dict": {
            'age': False,
            'seg': True,
            'position': True,
            'word': True
        }
    }


def save_phenofile(dataset, prediction_store, filename, inverse_normal_rank_cols=None):
    """
    Saved file according to plink2 phenofile format https://www.cog-genomics.org/plink/2.0/input#pheno
    :param inverse_normal_rank_cols:
    :param dataset: trial_n dataframe with eid in
    :param predictions: predictions must in the same order and unshuffled
    :param filename: baseline_path without file extention
    :return:
    """
    if inverse_normal_rank_cols is None:
        inverse_normal_rank_cols = ['anchor', 'binomial_r', 'bert']

    pheno_columns = list(prediction_store.keys())
    prediction_store.update({'eid': dataset.eid.values})
    predictions_df = pd.DataFrame(prediction_store)

    inverse_normal_rank_cols = [col for col in inverse_normal_rank_cols if col in predictions_df.columns]
    for col in inverse_normal_rank_cols:
        predictions_df.loc[:, col] = inverse_normal_rank(predictions_df.loc[:, col])

    bridge_file = os.path.join(DATA_DIR, 'external', 'ukb12113_ukb58356_bridgefile_tabdelim.txt')
    bridge = pd.read_csv(bridge_file, delimiter='\t')

    phenofile = pd.merge(predictions_df, bridge, left_on='eid', right_on='ID_1_58356', how='inner')

    phenofile = phenofile.loc[:, ['ID_1_12113'] + pheno_columns]
    phenofile.columns = ['IID'] + pheno_columns
    phenofile.to_csv(filename, sep='\t', index=False)
    print("pheno file saved at: {}".format(filename))


def gen_covariates_file(filepath=os.path.join(DATA_DIR, 'processed', 'pheprob', 'covariates.tsv')):
    if not os.path.exists(filepath):
        baseline_path = '/SAN/ihibiobank/denaxaslab/UKB_application_58356/raw_data/baseline/58356.csv'
        chunksize = 10_000
        estimated_total_rows = 500_000
        eid = [0]
        pca = [9900, 9901, 9902, 9903, 9904, 9905, 9906, 9907, 9908, 9909]
        sex_yob = [22, 23]
        reader = pd.read_csv(baseline_path, sep=',', chunksize=chunksize, low_memory=False, encoding="ISO-8859-1")
        chunk_store = []
        for i, chunk in enumerate(reader):
            print('Total chunks read {}, estimate remaining {}'.format(i, (estimated_total_rows / chunksize) - i))
            chunk_store.append(chunk.iloc[:, eid + sex_yob + pca])

        base = pd.concat(chunk_store, axis=0)

        bridge_file = os.path.join(DATA_DIR, 'external', 'ukb12113_ukb58356_bridgefile_tabdelim.txt')
        bridge = pd.read_csv(bridge_file, delimiter='\t')

        covariates = pd.merge(base, bridge, left_on='eid', right_on='ID_1_58356', how='inner')

        covariates = covariates.loc[:, ['ID_1_12113'] + list(base.columns[1:])]
        covariates.columns = ['IID', 'sex', 'age', 'pca1', 'pca2', 'pca3', 'pca4', 'pca5', 'pca6', 'pca7', 'pca8',
                              'pca9', 'pca10']

        # TODO: replace with age of onset?
        covariates.age = 2018 - covariates.age
        print("Saved covariates at: {}".format(filepath))
        covariates.to_csv(filepath, sep='\t', index=False)
    else:
        print("Covariates already saved at: {}".format(filepath))


@ex.automain
def main(target_token, global_params, file_config, bert_config):
    code_vocab = load_pickle(file_config['code_vocab'])
    age_vocab = load_pickle(file_config['age_vocab'])
    phe_vocab = load_pickle(file_config['phe_vocab'])

    model_token2idx = phe_vocab['token2idx']

    train_data = pd.read_parquet(file_config['train_data'])
    val_data = pd.read_parquet(file_config['val_data'])
    train_data = train_data.head(10_000) if DEBUG else train_data
    val_data = val_data.head(1_000) if DEBUG else val_data

    train_dataset = PheDataset(train_data, code_vocab['token2idx'], age_vocab['token2idx'],
                               max_len=global_params['max_len_seq'],
                               phe2idx=phe_vocab['token2idx'])
    val_dataset = PheDataset(val_data, code_vocab['token2idx'], age_vocab['token2idx'],
                             max_len=global_params['max_len_seq'],
                             phe2idx=phe_vocab['token2idx'])

    estimators = [
        {'name': 'bert',
         # 'model': BERTAnchorModel(token=target_token, token2idx=model_token2idx,
         #                          bert_config=bert_config, file_config=file_config,
         #                          num_workers=N_WORKERS,
         #                          checkpoint_path=os.path.join(MODEL_DIR, 'lightning',
         #                                                       'BERTAnchorModel_{}.ckpt'.format(target_token)))},
         'model': BERTAnchorModel.load_from_checkpoint(
             token=target_token, token2idx=model_token2idx, bert_config=bert_config, file_config=file_config,
             checkpoint_path=os.path.join(MODEL_DIR, 'lightning', 'BERTAnchorModel_{}.ckpt'.format(target_token)),
             do_training=False, num_workers=N_WORKERS)
         },
        {'name': 'binomial_r',
         'model': BinomialMixtureModelR(token=target_token, token2idx=model_token2idx,
                                        num_workers=N_WORKERS)},
        {'name': 'anchor',
         'model': AnchorModel(token=target_token, token2idx=model_token2idx,
                              num_workers=N_WORKERS)},
        {'name': 'threshold1',
         'model': ThresholdModel(threshold=1, token=target_token, token2idx=model_token2idx,
                                 num_workers=N_WORKERS)},
        {'name': 'threshold2',
         'model': ThresholdModel(threshold=2, token=target_token, token2idx=model_token2idx,
                                 num_workers=N_WORKERS)},
        {'name': 'threshold3',
         'model': ThresholdModel(threshold=3, token=target_token, token2idx=model_token2idx,
                                 num_workers=N_WORKERS)},
    ]

    prediction_store = {}
    for e in estimators:
        model = e['model']
        predictions = model.predict(train_dataset, val_dataset)
        prediction_store.update({e['name']: predictions.numpy()})

    estimator_names = '_'.join([n['name'] for n in estimators])
    save_phenofile(train_dataset, prediction_store,
                   os.path.join(DATA_DIR, 'processed', 'pheprob', '{}_{}.tsv'.format(estimator_names, target_token)),
                   inverse_normal_rank_cols=global_params['inverse_normal_rank_cols'])
