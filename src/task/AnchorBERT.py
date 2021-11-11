import os

import pandas as pd
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader, ConcatDataset
from data.datasets import PheDataset
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

from task.pheno import save_phenofile, apply_anchor, update_phenofile, anchor_decorator

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

# MI 411.2

# heart failure: 428.2

SYMBOLS = ['PAD',
           'MASK',
           'SEP',
           'CLS',
           'UNK', ]

DEBUG = __debug__
# DEBUG = True
DEBUG_STRING = 'debug' if __debug__ else ''


@ex.config
def config():
    target_token = '411.2'
    global_params = {
        'with_codes': 'all',
        'max_len_seq': 256,
        'inverse_normal_rank_cols': ['bert_anchor'],  # None to activate for all cols
        'anchor_cols': ['bert'],
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
        'pretrained': False,
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
        'tensorboard_dir': os.path.join(MODEL_DIR, 'tensorboard', 'anchor_bert'),
        "feature_dict": {
            'age': False,
            'seg': True,
            'position': True,
            'word': True
        }
    }

    n_workers = 0 if DEBUG else os.cpu_count()


@ex.automain
def main(_run, target_token, global_params, file_config, bert_config, n_workers):
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

    train_data = train_data.head(20_000) if DEBUG else train_data
    val_data = val_data.head(10_000) if DEBUG else val_data
    test_data = val_data.head(10_000) if DEBUG else test_data

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

    train_dataloader = DataLoader(dataset=train_dataset, batch_size=bert_config['train_params']['batch_size'],
                                  shuffle=True,
                                  num_workers=n_workers)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=bert_config['train_params']['batch_size'],
                                shuffle=False,
                                num_workers=n_workers)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=bert_config['train_params']['batch_size'],
                                 shuffle=False,
                                 num_workers=n_workers)

    if bert_config['pretrained']:
        model = BERTAnchorModel.load_from_checkpoint(
            token=target_token, token2idx=model_token2idx, bert_config=bert_config, file_config=file_config,
            checkpoint_path=os.path.join(file_config['bert_checkpoint_dir'], file_config['bert_checkpoint_name']),
            skip_training=bert_config['skip_training'], num_workers=n_workers)
    else:
        tensorboard_name = file_config['phenofile_name'] + '_' + str(_run._id)
        model = BERTAnchorModel(token=target_token, token2idx=model_token2idx,
                                bert_config=bert_config, file_config=file_config,
                                num_workers=n_workers,
                                tensorboard_name=tensorboard_name)

    checkpoint_filename = '{epoch}-{val_loss:.4f}{val_AveragePrecision:.4f}{val_AUROC:.4f}'
    checkpoint_filename = checkpoint_filename + 'debug' if __debug__ else checkpoint_filename
    checkpoint_callback = ModelCheckpoint(dirpath=model.bert_checkpoint_dir,
                                          filename=checkpoint_filename,
                                          monitor='val_AveragePrecision',
                                          mode='max')
    trainer = pl.Trainer(max_epochs=model.train_params['epochs'], check_val_every_n_epoch=1,
                         val_check_interval=model.train_params['val_check_interval'],
                         checkpoint_callback=True,
                         callbacks=[checkpoint_callback, ],
                         gpus=model.train_params['gpus'],
                         num_sanity_val_steps=-1,
                         accumulate_grad_batches=model.train_params['accumulate_grad_batches'],
                         logger=model.tb_logger)

    trainer.fit(model, train_dataloader, val_dataloader)
    metrics = {}
    metrics.update(trainer.validate(model, val_dataloader)[0])
    metrics.update(trainer.test(model, test_dataloader)[0])

    full_dataset = ConcatDataset([train_dataset, val_dataset, test_dataset])

    prediction_dataloader = DataLoader(dataset=full_dataset, batch_size=512,
                                       shuffle=False, num_workers=model.num_workers)
    prediction_trainer = pl.Trainer(gpus=model.train_params['gpus'])
    results = prediction_trainer.predict(model, prediction_dataloader)

    predictions = torch.cat([r['predictions'] for r in results]).cpu()

    if val_dataset is not None:
        full_data = pd.concat([train_data, val_data, test_data], axis=0)
    else:
        full_data = train_data

    phenofile = save_phenofile(full_data, {'bert': predictions},
                               os.path.join(DATA_DIR, 'processed', 'phenotypes',
                                            file_config['phenofile_name'] + '.tsv'),
                               anchor_cols=global_params['anchor_cols'],
                               inverse_normal_rank_cols=global_params['inverse_normal_rank_cols'])

    anchor_var = phenofile.threshold1
    threshold1_anchor_func = anchor_decorator(apply_anchor, anchor_var)

    update_phenofile(threshold1_anchor_func, global_params['anchor_cols'], phenofile=phenofile,
                     new_filename=os.path.join(DATA_DIR, 'processed', 'phenotypes',
                                               file_config['phenofile_name'] + '_anchor.tsv'),
                     update_colnames='_anchor', drop_cols=True)

    return metrics
