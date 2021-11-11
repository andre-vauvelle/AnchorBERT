import math
import numpy as np

import hyperopt
from hyperopt import fmin, hp
from hyperopt.mongoexp import MongoTrials
from hyperopt.pyll import scope

from task.pretrain_mlm import hyperopt_objective

import warnings

# see https://stackoverflow.com/a/40846742
warnings.filterwarnings("ignore", message="numpy.dtype size changed")

if __name__ == '__main__':
    name = 'pretrain_mlm'
    version = 1.5
    exp_key = '{}_{}'.format(name, version)
    max_evals = 100

    space = {
        "name": exp_key,
        "bert_config": {
            "model_config": {
                # hidden size much be multiple of attention heads
                "hidden_size": scope.int(hp.quniform('hidden_size', 120, 360, 12)),
                "intermediate_size": scope.int(hp.quniform('intermediate_size', 128, 516, 1)),
                "num_attention_heads": 12,
                "num_hidden_layers": scope.int(hp.quniform('num_hidden_layers', 2, 10, 1)),
            },
            "optim_config": {
                "lr": hp.loguniform('lr', np.log(1e-5), np.log(1e-3)),
            }
        }
    }

    trials = MongoTrials('mongo://bigtop:27017/hyperopt/jobs',
                         exp_key=exp_key)
    print('using exp key {}'.format(exp_key))
    print('Pending on workers to connect ..')
    argmin = fmin(fn=hyperopt_objective,
                  space=space,
                  algo=hyperopt.tpe.suggest,
                  max_evals=max_evals,
                  trials=trials,
                  verbose=True)
    best_acc = 1 - trials.best_trial['result']['loss']

    print('best val precision=', best_acc, 'params:', argmin)
