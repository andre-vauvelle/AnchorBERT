import os

import pandas as pd
import torch
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from torch.utils.data import DataLoader, ConcatDataset
from data.datasets import PheDataset
from model.bert.mlm import BERTMLM
from omni.common import load_pickle
import time

import pytorch_lightning as pl

from definitions import DATA_DIR, MODEL_DIR, MONGO_STR, MONGO_DB, RESULTS_DIR

from sacred import Experiment
from sacred.observers import MongoObserver
from sacred.utils import apply_backspaces_and_linefeeds

base = os.path.basename(__file__)
experiment_name = os.path.splitext(base)[0]
ex = Experiment(experiment_name)
ex.observers.append(MongoObserver(url=MONGO_STR, db_name=MONGO_DB))
ex.captured_out_filter = apply_backspaces_and_linefeeds

SYMBOLS = ['PAD',
           'MASK',
           'SEP',
           'CLS',
           'UNK', ]

# DEBUG = __debug__
# DEBUG = True
DEBUG = False
# DEBUG_STRING = 'debug' if __debug__ else ''
DEBUG_STRING = ''


@ex.config
def config():
    name = 'default'
    global_params = {
        'use_code': 'code',  # 'phecode'
        'with_codes': 'all',
        'max_len_seq': 256,
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
        # formatted data
    }
    # bert_config = None
    bert_config = {
        'skip_training': False,
        "optim_config": {
            'lr': 1e-4,
            'warmup_proportion': 0.1,
            'weight_decay': 0.01
        },
        "train_params": {
            'epochs': 1 if DEBUG else 100,
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
        'bert_checkpoint_dir': os.path.join(MODEL_DIR, 'mlm'),
        'bert_checkpoint_name': None,
        # 'bert_checkpoint_name': 'mlm-epoch=0-val_loss=0.1265-val_AveragePrecision=1.0000.ckpt',
        'tensorboard_dir': os.path.join(MODEL_DIR, 'tensorboard', 'mlm', name),
        "feature_dict": {
            'age': False,
            'seg': True,
            'position': True,
            'word': True
        }
    }

    n_workers = 0 if DEBUG else os.cpu_count()


@ex.automain
def main(_run, global_params, file_config, bert_config, n_workers):
    code_vocab = load_pickle(file_config['code_vocab'])
    age_vocab = load_pickle(file_config['age_vocab'])
    phe_vocab = load_pickle(file_config['phe_vocab'])

    model_token2idx = phe_vocab['token2idx']

    if not DEBUG:
        train_data = pd.read_parquet(file_config['train_data'])
        val_data = pd.read_parquet(file_config['val_data'])
        test_data = pd.read_parquet(file_config['test_data'])
    else:
        train_data = pd.read_parquet(file_config['train_data'] + '.debug')
        val_data = pd.read_parquet(file_config['val_data'] + '.debug')
        test_data = pd.read_parquet(file_config['test_data'] + '.debug')

    train_dataset = PheDataset(None, train_data, code_vocab['token2idx'], age_vocab['token2idx'],
                               phe2idx=phe_vocab['token2idx'],
                               max_len=global_params['max_len_seq'],
                               mlm=True)
    val_dataset = PheDataset(None, val_data, code_vocab['token2idx'], age_vocab['token2idx'],
                             phe2idx=phe_vocab['token2idx'],
                             max_len=global_params['max_len_seq'],
                             mlm=True)
    test_dataset = PheDataset(None, test_data, code_vocab['token2idx'], age_vocab['token2idx'],
                              phe2idx=phe_vocab['token2idx'],
                              max_len=global_params['max_len_seq'],
                              mlm=True)

    train_dataloader = DataLoader(dataset=train_dataset, batch_size=bert_config['train_params']['batch_size'],
                                  shuffle=True,
                                  num_workers=n_workers)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=bert_config['train_params']['batch_size'],
                                shuffle=False,
                                num_workers=n_workers)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=bert_config['train_params']['batch_size'],
                                 shuffle=False,
                                 num_workers=n_workers)
    if bert_config['bert_checkpoint_name'] is not None:
        model = BERTMLM.load_from_checkpoint(token2idx=model_token2idx,
                                             bert_config=bert_config, file_config=file_config,
                                             num_workers=n_workers,
                                             tensorboard_name=str(_run._id) + DEBUG_STRING,
                                             checkpoint_path=os.path.join(bert_config['bert_checkpoint_dir'],
                                                                          # 'mlm-epoch=0-val_loss=0.1251.ckpt'))
                                                                          bert_config['bert_checkpoint_name']))
    else:
        model = BERTMLM(token2idx=model_token2idx,
                        bert_config=bert_config, file_config=file_config,
                        num_workers=n_workers,
                        tensorboard_name=str(_run._id) + DEBUG_STRING)

    checkpoint_filename = 'mlm-{epoch}-{val_loss:.4f}-{val_AveragePrecision:.4f}{val_Precision:.4f}'
    checkpoint_filename = checkpoint_filename + 'debug' if DEBUG else checkpoint_filename
    checkpoint_callback = ModelCheckpoint(dirpath=bert_config['bert_checkpoint_dir'],
                                          filename=checkpoint_filename,
                                          monitor='val_Precision',
                                          mode='max'
                                          )

    early_stop_callback = EarlyStopping(monitor="val_Precision", min_delta=1e-6, patience=15, verbose=True, mode="max")

    trainer = pl.Trainer(max_epochs=bert_config['train_params']['epochs'], check_val_every_n_epoch=1,
                         val_check_interval=bert_config['train_params']['val_check_interval'],
                         checkpoint_callback=True,
                         callbacks=[checkpoint_callback, early_stop_callback],
                         gpus=bert_config['train_params']['gpus'],
                         num_sanity_val_steps=-1,
                         accumulate_grad_batches=bert_config['train_params']['accumulate_grad_batches'],
                         logger=model.tb_logger,
                         )

    if bert_config['train_params']['auto_lr_find']:
        # Run learning rate finder
        lr_finder = trainer.tuner.lr_find(model, train_dataloader, val_dataloader)
        # Plot with
        fig = lr_finder.plot(suggest=True)
        fig.show()
        # Pick point based on plot, or get suggestion
        new_lr = lr_finder.suggestion()
        # update lr of the model
        lr = new_lr

    trainer.fit(model, train_dataloader, val_dataloader)
    metrics = {}
    metrics.update(trainer.validate(model, val_dataloader)[0])
    metrics.update(trainer.test(model, test_dataloader)[0])
    return metrics


##### HyperOpt support #####################
from hyperopt import STATUS_OK, STATUS_FAIL


# noinspection PyUnresolvedReferences

def hyperopt_objective(config_updates):
    try:
        run = ex.run(config_updates=config_updates)
        err = run.result

        return {'loss': 1 - err['val_Precision'], 'status': STATUS_OK}
    except Exception as e:
        return {'status': STATUS_FAIL,
                'time': time.time(),
                'exception': str(e)}

##### End HyperOpt support #################
