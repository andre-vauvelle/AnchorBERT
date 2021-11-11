# import math
# import numpy as np
#
# import hyperopt
# from hyperopt import fmin, hp
# from hyperopt.mongoexp import MongoTrials
# from hyperopt.pyll import scope
#
#
#
# import warnings
#
# # see https://stackoverflow.com/a/40846742
# warnings.filterwarnings("ignore", message="numpy.dtype size changed")
#
# if __name__ == '__main__':
#     name = 'GRU'
#     version = 2.0
#     max_evals = 100
#
#     # Define the search space
#
#     space = {
#         "bert_config": {
#         "d_embedding": scope.int(hp.quniform('d_embedding', 12, 256, 1)),
#             }
#         "add_time": True,
#         "use_timestamps": True,
#         "t_scale": hp.uniform('t_scale', 86400, 604800),  # days and weeks
#         "t_max": hp.uniform('t_max', 0, 10),  # not to explode activations
#         "leadlag": False,
#         "batch_size": 128,
#         "verbose": True,
#         "epochs": 20,
#         "lr": hp.loguniform('lr', np.log(1e-5), np.log(1e-1)),
#         "wd": hp.loguniform('wd', np.log(1e-7), np.log(1e-2)),
#         "hidden_rnn_sz": scope.int(hp.quniform('hidden_rnn_sz', 32, 128, 1)),
#         "rnn_num_layers": hp.uniformint('rnn_num_layers', 1, 2),
#         "patience": 10,
#         "rnn_dropout": hp.uniform('rnn_dropout', 0, 0.9),
#         "feedforward_num_layers": hp.uniformint('feedforward_num_layers', 1, 3),
#         "min_count": 5,
#         "testing_subsample_size": None,
#         # "testing_subsample_size": 1000
#         "feedforward_hidden_dims": scope.int(hp.quniform('feedforward_hidden_dims', 32, 256, 4)),
#         "feedforward_activations": "relu",
#         "feedforward_dropout": hp.uniform('feedforward_dropout', 0, 0.7),
#         "evaluate_on_test": False
#     }
#
#     space.update(
#         {
#             "name": "{name}_{version}_leadlag{leadlag}_addtime{add_time}_timestamps{use_timestamps}_allcode{all_code_types}".format_map(
#                 space)})
#
#     trials = MongoTrials('mongo://bigtop:27017/hyperopt/jobs',
#                          exp_key='{}'.format(space['name']))
#     print('using exp key {}'.format(space['name']))
#     print('Pending on workers to connect ..')
#     argmin = fmin(fn=hyperopt_objective,
#                   space=space,
#                   algo=hyperopt.tpe.suggest,
#                   max_evals=max_evals,
#                   trials=trials,
#                   verbose=True)
#     best_acc = 1 - trials.best_trial['result']['loss']
#
#     print('best val acc=', best_acc, 'params:', argmin)