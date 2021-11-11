import os

import numpy as np

import hyperopt
import torch
from hyperopt import fmin, hp
from hyperopt.mongoexp import MongoTrials
from hyperopt.pyll import scope

from definitions import MODEL_DIR
from omni.common import load_pickle
from task.pheno import hyperopt_objective

import warnings

# see https://stackoverflow.com/a/40846742
warnings.filterwarnings("ignore", message="numpy.dtype size changed")

DEBUG_STRING = 'debug' if __debug__ else ''

if __name__ == '__main__':
    target_token = '411.2'
    max_evals = 50

    global_params = {
        'with_codes': 'all',
        'max_len_seq': 256,
        'case_noise': 0,
        'control_noise': 0}

    phenofile_name = '{}_ca{}_co{}'.format(target_token,
                                           str(global_params['case_noise']),
                                           str(global_params['control_noise'])) + DEBUG_STRING

    # Define the search space
    space = {
        'target_token': target_token,

        'bert_config': {
            'pretrained': False,
            'skip_training': False,
            "optim_config": {
                "lr": hp.loguniform('lr', np.log(1e-6), np.log(1e-1)),
            },
            "train_params": {
                'epochs': 1 if __debug__ else 5,
                'batch_size': 64,
                'accumulate_grad_batches': 4,
                'effective_batch_size': 256,
                'gpus': -1 if torch.cuda.is_available() else 0,
                'auto_scale_batch_size': False,
                'val_check_interval': 0.2,
            },
            "model_config": {
                'hidden_size': hp.uniform('hidden_size', 50, 300),  # word embedding and seg embedding hidden size
                'hidden_dropout_prob': hp.uniform('hidden_dropout_prob', 0.1, 0.3),  # dropout rate
                'num_hidden_layers': hp.uniformint('num_hidden_layers', 2, 8),
                # number of multi-head attention layers required
                'num_attention_heads': hp.uniformint('num_attention_heads', 6, 12),  # number of attention heads
                'attention_probs_dropout_prob': hp.uniform('attention_probs_dropout_prob', 0.1, 0.3),
                # multi-head attention dropout rate
                'intermediate_size': hp.uniformint('intermediate_size', 128, 516),
                # the size of the "intermediate" layer in the transformer encoder
            },
            'bert_checkpoint_dir': os.path.join(MODEL_DIR, phenofile_name),
        }
    }

    trials = MongoTrials('mongo://bigtop:27017/hyperopt/jobs',
                         exp_key='{}'.format(phenofile_name))
    print('using exp key {}'.format(phenofile_name))
    print('Pending on workers to connect ..')
    argmin = fmin(fn=hyperopt_objective,
                  space=space,
                  algo=hyperopt.tpe.suggest,
                  max_evals=max_evals,
                  trials=trials,
                  verbose=True)
    best_acc = 1 - trials.best_trial['result']['loss']

    print('best val acc=', best_acc, 'params:', argmin)
