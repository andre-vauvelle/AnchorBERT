import os

import numpy as np
import sklearn
from sklearn.metrics import confusion_matrix
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
from sacred import Experiment
from sacred.observers import FileStorageObserver, MongoObserver
from sacred.utils import apply_backspaces_and_linefeeds

from data.datasets import SimulatedDataset
from data.simulation.markovian import calculate_beta_gene, generate_data
from definitions import DATA_DIR, MONGO_STR, MONGO_DB, RESULTS_DIR, MODEL_DIR
from model.anchor_model import AnchorModel
from model.bert.main import BERTAnchorModel
from model.metrics import get_p_val, get_odds_ratio, get_power
from model.thresholder import ThresholdModel
from model.binomial_mixture_model_r import BinomialMixtureModelR
import statsmodels.api as sm
import torchmetrics

base = os.path.basename(__file__)
experiment_name = os.path.splitext(base)[0]
ex = Experiment(experiment_name)
ex.observers.append(MongoObserver(url=MONGO_STR, db_name=MONGO_DB))
ex.captured_out_filter = apply_backspaces_and_linefeeds


@ex.config
def config():
    simulation_config = {
        'name': 'default',
        'n': 100_000,
        'trials': 1,
        'base_prevalence': 0.2,
        # 'min_seq': 5,
        'mean_seq': 20,
        'odds_ratio': 1.1,
        'minor_allele_frequency': 0.1,
        'beta2': 0.3,
        'emission': [[0.005, 0.001, 0.04, 0.0005], [0.8, 0.8, 0.1, 0.6]],  # test if model is working
        # 'emission': [[0.0, 0, 0], [0.9, 1, 0]],
        'tv_split': 0.8  # train validation split
    }
    inverse_rank = False


# if __name__ == '__main__':
@ex.automain
def run(simulation_config, inverse_rank):
    store = []

    bert_config = {
        "optim_config": {
            # 'lr': 0.00478630092322638,
            'lr': 5e-4,
            'warmup_proportion': 0.1,
            'weight_decay': 0.01
        },
        "train_params": {
            'epochs': 5,
            'batch_size': 256,
            'accumulate_grad_batches': 2,
            'effective_batch_size': 256*2,
            'gpus': -1 if torch.cuda.is_available() else 0,
            'auto_scale_batch_size': False,
            'auto_lr_find': False,
            'val_check_interval': 0.2
        },
        "model_config": {
            # 'pretrained': os.path.join(MODEL_DIR, '60train_model_min_prop'),
            'pretrained': None,
            # if None then train from scratch
            'vocab_size': len(simulation_config['emission'][0]) + 3,  # +3 for sep, cls, pad token
            # number of disease + symbols for word embedding
            'seg_vocab_size': 2,  # number of vocab for seg embedding
            'age_vocab_size': None,
            # number of vocab for age embedding
            'max_position_embedding': simulation_config['mean_seq'],  # maximum number of tokens
            'hidden_size': 8,  # word embedding and seg embedding hidden size
            'hidden_dropout_prob': 0.0,  # dropout rate
            'num_hidden_layers': 2,  # number of multi-head attention layers required
            'num_attention_heads': 4,  # number of attention heads
            'attention_probs_dropout_prob': 0.22,  # multi-head attention dropout rate
            'intermediate_size': 64,
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

    token_idx_target = [3]
    estimators = [
        {'name': 'bert',
         'model': BERTAnchorModel(token=token_idx_target, token2idx=None, bert_config=bert_config,
                                  checkpoint_path=os.path.join(MODEL_DIR, 'lightning', 'BERTAnchorModel_sim.ckpt'))
         # 'model': BERTAnchorModel.load_from_checkpoint(
         #     token=token_idx_target, token2idx=None, bert_config=bert_config,
         #     checkpoint_path=os.path.join(MODEL_DIR, 'lightning', 'BERTAnchorModel_sim.ckpt'), do_training=False)
         },
        {'name': 'binomial_r',
         'model': BinomialMixtureModelR(token=token_idx_target, token2idx=None, num_workers=0)},
        {'name': 'anchor',
         'model': AnchorModel(token=token_idx_target, token2idx=None, num_workers=0,
                              emission_size=len(simulation_config['emission'][0]))},
        {'name': 'threshold1',
         'model': ThresholdModel(threshold=1, token=token_idx_target, token2idx=None, num_workers=0)},
        {'name': 'threshold2',
         'model': ThresholdModel(threshold=2, token=token_idx_target, token2idx=None, num_workers=0)},
        {'name': 'threshold3',
         'model': ThresholdModel(threshold=3, token=token_idx_target, token2idx=None, num_workers=0)},
    ]

    import itertools

    for trial_n in tqdm(range(simulation_config['trials']), desc='Trial progress'):
        # beta1 = calculate_beta(prevalence=simulation_config['base_prevalence'],
        #                          seq_len=simulation_config['mean_seq'])
        # beta1 = get_loggnormal_beta(prevalence=simulation_config['base_prevalence'], )
        beta2 = simulation_config['beta2']
        beta1, beta_g = calculate_beta_gene(prevalence=simulation_config['base_prevalence'],
                                            seq_len=simulation_config['mean_seq'],
                                            minor_allele_frequency=simulation_config['minor_allele_frequency'],
                                            odds_ratio=simulation_config['odds_ratio'])
        # for p1, p2 in tqdm(itertools.product(np.arange(0.9, 1, step=0.01), np.arange(0, 0.2, step=0.1))):
        X, Y, G = generate_data(simulation_config['n'],
                                minor_allele_frequency=simulation_config['minor_allele_frequency'],
                                mean_seq=simulation_config['mean_seq'],
                                transition_coeff=[beta1, beta2], genetic_coeff=beta_g, num_states_Y=2,
                                num_states_G=3,
                                emission=simulation_config['emission'],
                                g_dist='bernoulli')
        y_any = [1 if any(y) else 0 for y in Y]
        average_y = sum([sum(y) for y in np.array(Y)[np.array(y_any) > 0]]) / sum(y_any)
        prevalence_true = sum(y_any) / simulation_config['n']
        OR = get_odds_ratio(y_any, G)
        pval_true = get_p_val(y_any, G, inverse_rank=inverse_rank)

        store.append(
            {'trail': trial_n,
             'name': 'true',
             'prevalence': prevalence_true,
             'odds_ratio': OR,
             'p_val': pval_true,
             'power': get_power(y_any, G),
             'auroc': 1,
             'average_y': average_y})

        dataset = SimulatedDataset(X, max_len=simulation_config['mean_seq'] + 80)
        prediction_store = {}
        for e in estimators:
            model = e['model']
            # predictions = model.predict(train_dataset, train_dataset)
            predictions = model.predict(dataset)
            prediction_store.update({e['name']: predictions.numpy()})

        for name, predictions in prediction_store.items():
            predictions_r = predictions.round()
            prevalence_est = sum(predictions_r) / simulation_config['n']
            auroc = float(
                torchmetrics.functional.auroc(preds=torch.Tensor(predictions), target=torch.Tensor(y_any).int(),
                                              average='macro', pos_label=1))
            p_val_est = get_p_val(predictions, G, inverse_rank=False, family=sm.families.Binomial())
            p_val_est_ir = get_p_val(predictions, G, inverse_rank=True, family=sm.families.Gaussian())
            store.append(
                {'trail': trial_n, 'name': name, 'prevalence': prevalence_est,
                 'odds_ratio': get_odds_ratio(predictions, G),
                 'power': get_power(predictions_r, G), 'p_val': p_val_est, 'p_val_est_ir': p_val_est_ir,
                 'auroc': auroc, 'average_y': average_y}, )

    df = pd.DataFrame(store)
    results_path = os.path.join(RESULTS_DIR, 'simulation_results', simulation_config['name'] + '.csv')
    df.to_csv(results_path)
    ex.add_artifact(results_path)

    # # df.to_csv(os.path.join(DATA_DIR, 'processed', 'simulation', 'trails.csv'))
