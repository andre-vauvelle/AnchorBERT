import os

import pandas as pd
import torch
from torch.utils.data import DataLoader, ConcatDataset
from data.datasets import PheDataset
from model.bert.mlm import BERTMLM
from model.logistic_regression import LogisticRegressionModel
from model.bert.main import BERTAnchorModel
from model.binomial_mixture_model_r import BinomialMixtureModelR
from model.metrics import apply_inverse_normal_rank
from model.thresholder import ThresholdModel
from omni.common import load_pickle

import pytorch_lightning as pl

from definitions import DATA_DIR, MODEL_DIR, MONGO_STR, MONGO_DB, RESULTS_DIR

from sacred import Experiment
from sacred.observers import MongoObserver
from sacred.utils import apply_backspaces_and_linefeeds

base = os.path.basename(__file__)
experiment_name = os.path.splitext(base)[0]
ex = Experiment(experiment_name)
if MONGO_DB is not None:
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

# MI 411.2

# heart failure: 428.2

SYMBOLS = ['PAD',
           'MASK',
           'SEP',
           'CLS',
           'UNK', ]

DEBUG = __debug__
# DEBUG = True
# DEBUG = False
DEBUG_STRING = 'debug' if __debug__ else ''


@ex.config
def config():
    target_token = '411.2'
    global_params = {
        'with_codes': 'all',
        'max_len_seq': 256,
        'inverse_normal_rank_cols': ['logreg_anchor', 'bert_anchor'],  # None to activate for all cols
        'anchor_cols': ['logreg', 'bert'],
        'case_noise': 0,
        'control_noise': 0,
    }
    file_config = {
        'phenofile_name': '{}_ca{}_co{}'.format(target_token,
                                                str(global_params['case_noise']),
                                                str(global_params['control_noise'])) + DEBUG_STRING,
        'phe_vocab': os.path.join(DATA_DIR, 'processed', global_params['with_codes'], 'phecode_vocab.pkl'),
        # vocabulary idx2token, token2idx
        'code_vocab': os.path.join(DATA_DIR, 'processed', global_params['with_codes'], 'code_vocab.pkl'),
        # vocabulary idx2token, token2idx
        'age_vocab': os.path.join(DATA_DIR, 'processed', global_params['with_codes'], 'age_vocab.pkl'),
        # vocabulary idx2token, token2idx
        'train_data': os.path.join(DATA_DIR, 'processed', global_params['with_codes'], 'MLM', 'phe_train.parquet'),
        'val_data': os.path.join(DATA_DIR, 'processed', global_params['with_codes'], 'MLM', 'phe_val.parquet'),
        'test_data': os.path.join(DATA_DIR, 'processed', global_params['with_codes'], 'MLM', 'phe_test.parquet'),
        # formatted data
    }
    # bert_config = None
    bert_config = {
        # 'pretrained': 'mlm-epoch=49-val_loss=3.5780-val_AveragePrecision=0.1628val_Precision=0.2067.ckpt',
        'pretrained': '',
        'skip_training': False,
        "optim_config": {
            'lr': 1e-4,
            'warmup_proportion': 0.1,
            'weight_decay': 0.01
        },
        "train_params": {
            'epochs': 1 if __debug__ else 5,
            'batch_size': 64,
            'accumulate_grad_batches': 4,
            'effective_batch_size': 256,
            'gpus': -1 if torch.cuda.is_available() else 0,
            'auto_scale_batch_size': False,
            'auto_lr_find': False,
            'val_check_interval': 0.2,
        },
        "model_config": {
            # if False then train from scratch, else look in os.path.join(MODEL_DIR, 'lightning')
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
        'bert_checkpoint_dir': os.path.join(MODEL_DIR, file_config['phenofile_name']),
        'tensorboard_dir': os.path.join(MODEL_DIR, 'tensorboard', 'pheno_bert'),
        "feature_dict": {
            'age': False,
            'seg': True,
            'position': True,
            'word': True
        }
    }
    # logreg_config = {'C': 1}
    logreg_config = {
        'param_grid': {'C': [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]}, 'C': 1,
        'checkpoint_path': os.path.join(MODEL_DIR, 'sklearn_regression', '{}.pkl'.format(file_config['phenofile_name']))
    }

    n_workers = 0 if DEBUG else os.cpu_count()


def save_phenofile(dataset, prediction_store, filename, anchor_cols=None, inverse_normal_rank_cols=None):
    """
    Saved file according to plink2 phenofile format https://www.cog-genomics.org/plink/2.0/input#pheno
    :param inverse_normal_rank_cols:
    :param dataset: trial_n dataframe with eid in
    :param predictions: predictions must in the same order and unshuffled
    :param filename: baseline_path without file extention
    :return:
    """

    pheno_columns = list(prediction_store.keys())
    prediction_store.update({'eid': dataset.eid.values})
    predictions_df = pd.DataFrame(prediction_store)

    if anchor_cols is not None:
        for col in anchor_cols:
            predictions_df.loc[:, col + '_anchor'] = apply_anchor(predictions=predictions_df.loc[:, col],
                                                                  y_anchor=predictions_df.loc[:, 'threshold1'])

    inverse_normal_rank_cols = [col for col in inverse_normal_rank_cols if col in predictions_df.columns]
    for col in inverse_normal_rank_cols:
        predictions_df.loc[:, col + '_inr'] = apply_inverse_normal_rank(predictions_df.loc[:, col])

    bridge_file = os.path.join(DATA_DIR, 'external', 'ukb12113_ukb58356_bridgefile_tabdelim.txt')
    bridge = pd.read_csv(bridge_file, delimiter='\t')

    phenofile = pd.merge(predictions_df, bridge, left_on='eid', right_on='ID_1_58356', how='inner')

    phenofile = phenofile.loc[:, ['ID_1_12113'] + pheno_columns]
    phenofile.columns = ['IID'] + pheno_columns
    phenofile.to_csv(filename, sep='\t', index=False)
    print("pheno file saved at: {}".format(filename))
    return phenofile


def update_phenofile(func, columns, phenofile=None, filename=None, new_filename=None, update_colnames='',
                     drop_cols=True):
    """
    Update phenotypes with a function applied to a pandas series
    :param func: Function that accepts pandas series
    :param columns:
    :param phenofile:
    :param filename:
    :param new_filename:
    :param update_colnames:
    :return:
    """
    if phenofile is None:
        phenofile = pd.read_csv(filename, sep='\t')

    if filename is None and phenofile is None:
        raise ValueError("Please include either phenofile or filename")

    for col in columns:
        phenofile.loc[:, col + update_colnames] = func(phenofile.loc[:, col])

    if drop_cols:
        phenofile = phenofile.drop(columns=columns)

    if new_filename is not None:
        phenofile.to_csv(new_filename, sep='\t', index=False)
    return phenofile


def apply_anchor(predictions: pd.Series, y_anchor: pd.Series):
    """Update controls with prediction probabilites, cases are set to 1"""
    y_anchor_inv = (-1) * (y_anchor - 1)
    predictions = y_anchor_inv * predictions  # updating controls
    predictions = predictions + y_anchor  # setting cases to 1
    return predictions


def anchor_decorator(func, anchor):
    def wrapper(predictions):
        return func(predictions, anchor)

    return wrapper


def gen_covariates_file(filepath=os.path.join(DATA_DIR, 'processed', 'covariates', 'covariates.tsv')):
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


def apply_anchor_dict(prediction_store, names):
    for name, data in prediction_store.items():
        if name in names:
            anchor_data = apply_anchor(data, prediction_store['threshold1'])
            prediction_store.update({name + '_anchor': anchor_data})


def apply_inverse_normal_rank_dict(prediction_store, names):
    for name, data in prediction_store.items():
        if name in names:
            inr_data = apply_inverse_normal_rank(data)
            prediction_store.update({name + '_inr': inr_data})


def metrics_tensor_to_float(metrics_store: dict):
    for name, metrics in metrics_store.items():
        for metric_name, value in metrics.items():
            metrics_store[name][metric_name] = round(float(value), 4)


@ex.automain
def main(target_token, global_params, file_config, bert_config, logreg_config, n_workers):
    target_token = str(target_token)

    code_vocab = load_pickle(file_config['code_vocab'])
    age_vocab = load_pickle(file_config['age_vocab'])
    phe_vocab = load_pickle(file_config['phe_vocab'])

    model_token2idx = phe_vocab['token2idx']

    train_data = pd.read_parquet(file_config['train_data'])
    val_data = pd.read_parquet(file_config['val_data'])
    test_data = pd.read_parquet(file_config['test_data'])

    # No hyperparameter tuning. Merge test data into training and only use validation
    # data for performance for anchor variable prediction
    # train_data = pd.concat([train_data,], axis=0)

    train_data = train_data.head(2_000) if DEBUG else train_data
    val_data = val_data.head(1_000) if DEBUG else val_data
    test_data = test_data.head(1_000) if DEBUG else test_data

    train_dataset = PheDataset(target_token, train_data, code_vocab['token2idx'], age_vocab['token2idx'],
                               max_len=global_params['max_len_seq'],
                               phe2idx=phe_vocab['token2idx'],
                               case_noise=global_params['case_noise'],
                               control_noise=global_params['control_noise'])
    val_dataset = PheDataset(target_token, val_data, code_vocab['token2idx'], age_vocab['token2idx'],
                             max_len=global_params['max_len_seq'],
                             phe2idx=phe_vocab['token2idx'],
                             case_noise=global_params['case_noise'],
                             control_noise=global_params['control_noise'])
    test_dataset = PheDataset(target_token, test_data, code_vocab['token2idx'], age_vocab['token2idx'],
                              max_len=global_params['max_len_seq'],
                              phe2idx=phe_vocab['token2idx'],
                              case_noise=global_params['case_noise'],
                              control_noise=global_params['control_noise'])

    if bert_config is None:
        bert_model = None
    else:
        if bert_config['pretrained']:
            pretrained_bert = BERTMLM.load_from_checkpoint(token2idx=model_token2idx, bert_config=bert_config,
                                                           checkpoint_path=os.path.join(MODEL_DIR, 'mlm',
                                                                                        bert_config['pretrained'])
                                                           )
            tensorboard_name = file_config['phenofile_name']
            bert_model = BERTAnchorModel(
                token=target_token, token2idx=model_token2idx, bert_config=bert_config, file_config=file_config,
                skip_training=bert_config['skip_training'], num_workers=n_workers, tensorboard_name=tensorboard_name,
                pretrained_bert=pretrained_bert.bert)
        else:
            tensorboard_name = file_config['phenofile_name']
            bert_model = BERTAnchorModel(token=target_token, token2idx=model_token2idx,
                                         bert_config=bert_config, file_config=file_config,
                                         num_workers=n_workers,
                                         tensorboard_name=tensorboard_name)
    if logreg_config is None:
        logreg_model = None
    else:
        logreg_model = LogisticRegressionModel(token=target_token, token2idx=model_token2idx, C=logreg_config['C'],
                                               num_workers=n_workers, param_grid=logreg_config['param_grid'],
                                               checkpoint_path=logreg_config['checkpoint_path'])

    estimators = [
        {'name': 'bert',
         'model': bert_model},
        {'name': 'logreg',
         'model': logreg_model},
        {'name': 'binomial_r',
         'model': BinomialMixtureModelR(token=target_token, token2idx=model_token2idx,
                                        num_workers=n_workers)},
        {'name': 'threshold1',
         'model': ThresholdModel(threshold=1, token=target_token, token2idx=model_token2idx,
                                 num_workers=n_workers)},
        {'name': 'threshold1_noised',
         'model': ThresholdModel(threshold=1, token=target_token, token2idx=model_token2idx,
                                 num_workers=n_workers, label_noise=True)},
        {'name': 'threshold2',
         'model': ThresholdModel(threshold=2, token=target_token, token2idx=model_token2idx,
                                 num_workers=n_workers)},
        {'name': 'threshold3',
         'model': ThresholdModel(threshold=3, token=target_token, token2idx=model_token2idx,
                                 num_workers=n_workers)},
    ]
    noise_used = global_params['case_noise'] != 0 or global_params['control_noise'] != 0
    if noise_used:
        estimators.append(
            {'name': 'threshold1_noised',
             'model': ThresholdModel(threshold=1, token=target_token, token2idx=model_token2idx,
                                     num_workers=n_workers, label_noise=True)}
        )

    prediction_store = {}
    metrics_store = {}
    for e in estimators:
        model = e['model']
        if model is None:
            continue
        predictions, metrics = model.predict(train_dataset, val_dataset, test_dataset)
        prediction_store.update({e['name']: predictions.numpy()})
        metrics_store.update({e['name']: metrics})

    # estimator_names = '_'.join([e['name'] for e in estimators if e['model'] is not None])

    if val_dataset is not None:
        full_data = pd.concat([train_data, val_data, test_data], axis=0)
    else:
        full_data = train_data

    phenofile = save_phenofile(full_data, prediction_store,
                               os.path.join(DATA_DIR, 'processed', 'phenotypes',
                                            file_config['phenofile_name'] + '.tsv'),
                               anchor_cols=global_params['anchor_cols'],
                               inverse_normal_rank_cols=global_params['inverse_normal_rank_cols'])

    anchor_var = phenofile.threshold1_noised if noise_used else phenofile.threshold1
    threshold1_anchor_func = anchor_decorator(apply_anchor, anchor_var)

    phenofile = update_phenofile(threshold1_anchor_func, global_params['anchor_cols'], phenofile=phenofile,
                                 new_filename=os.path.join(DATA_DIR, 'processed', 'phenotypes',
                                                           file_config['phenofile_name'] + '_anchor.tsv'),
                                 update_colnames='_anchor', drop_cols=True)

    update_phenofile(apply_inverse_normal_rank, global_params['inverse_normal_rank_cols'], phenofile=phenofile,
                     new_filename=os.path.join(DATA_DIR, 'processed', 'phenotypes',
                                               file_config['phenofile_name'] + '_anchor_inr.tsv'),
                     update_colnames='_inr', drop_cols=True)
    metrics_tensor_to_float(metrics_store)
    df_results = pd.DataFrame(metrics_store)
    results_path = os.path.join(RESULTS_DIR, 'pheno_results', 'classification_metrics',
                                file_config['phenofile_name'] + '.csv')
    df_results.to_csv(results_path)
    ex.add_artifact(results_path, name='classification_metrics')
    return metrics_store


##### HyperOpt support #####################
from hyperopt import STATUS_OK, STATUS_FAIL


# noinspection PyUnresolvedReferences

def hyperopt_objective(params):
    config = {}

    try:
        if type(params) == dict:
            params = params.items()

        for (key, value) in params:
            config[key] = value
        run = ex.run(config_updates=config, )
        err = run.result

        if config['evaluate_on_test']:
            result = {'loss': 1, 'status': STATUS_OK}
            result.update(err)
            return result
        return {'loss': 1 - np.mean(err['best_validation_auc']), 'status': STATUS_OK}
    except Exception as e:
        return {'status': STATUS_FAIL,
                'time': time.time(),
                'exception': str(e)}

##### End HyperOpt support #################
