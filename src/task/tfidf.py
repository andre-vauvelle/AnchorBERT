from ast import literal_eval
import os

from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer, CountVectorizer
from sklearn.linear_model import LogisticRegressionCV
import scipy.stats as ss
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sacred import Experiment
from sacred.observers import FileStorageObserver, MongoObserver
from sacred.utils import apply_backspaces_and_linefeeds

# %%
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture

from definitions import MONGO_STR, MONGO_DB, DATA_DIR, MODEL_DIR, EXTERNAL_DATA_DIR

from omni.common import save_pickle, load_pickle

# chosen_phecode = '244.5'  # Hypothyroidism
chosen_phecode = '250.2'  # Type two diabetes
# chosen_phecode = '290.1' # Dementia

file_config = {'vocab': os.path.join(DATA_DIR, 'processed', 'code_vocab.pkl'),  # vocabulary idx2token, token2idx
               'age_vocab': os.path.join(DATA_DIR, 'processed', 'age_vocab.pkl'),  # vocabulary idx2token, token2idx
               'data_train': os.path.join(DATA_DIR, 'processed', 'all', 'MLM', 'phe_train.parquet'),  # formatted data
               'data_val': os.path.join(DATA_DIR, 'processed', 'all', 'MLM', 'phe_val.parquet'),  # formatted data
               'model_dir': os.path.join(MODEL_DIR, 'tfidf', 'all'),  # where to save model
               }

global_params = {
    'max_seq_len': 999999999,
    'min_visit': 1,
    'gradient_accumulation_steps': 1
}


def get_phenotype_documents(df):
    """
    Get phenotype_documents which are all codes from patients that share trial_n phecode_col
    :param df:
    :return:
    """
    phecodes = df.phecode.explode()
    phecodes = phecodes[~phecodes.isin(['MASK', 'SEP'])]

    storeDoc = {}
    for i, phecode in enumerate(phecodes.unique()):
        idx_contain_phecode = phecodes[phecodes == phecode].index.drop_duplicates()
        codes_ = df.loc[idx_contain_phecode, 'code_col'].explode()
        storeDoc.update({phecode: ' '.join(codes_[codes_ != 'SEP'].tolist())})
    return pd.Series(storeDoc)


def get_tfidf_weights(phenotype_documents):
    """
    Returns tfidf weights for every code_col and every phecode_col
    :param phenotype_documents:
    :return:
    """
    vectorizer = TfidfVectorizer(token_pattern=r"(?u)[^ ]+", lowercase=False)
    # vectorizer = CountVectorizer(token_pattern=r"(?<=\s)(.*?)(?=\s)", lowercase=False)
    X = vectorizer.fit_transform(phenotype_documents)
    indices = np.array(list(np.ndindex(X.shape)))
    X_norm = pd.DataFrame({'value': X.toarray().flatten(), 'phe_idx': indices[:, 0], 'code_idx': indices[:, 1]})
    code2idx = vectorizer.vocabulary_
    phe2idx = dict(zip(phenotype_documents.index, range(phenotype_documents.shape[0])))

    return X_norm, code2idx, phe2idx


def get_scores(df, X_norm, code2idx, phe2idx, chosen_phecode: str = '250.2', pooling='sum'):
    """
    Get the phescore from tfidf weights in X_norm
    :param df:
    :param X_norm:
    :param code2idx:
    :param phe2idx:
    :param chosen_phecode:
    :return:
    """
    chosen_phe_idx = phe2idx[chosen_phecode]
    df.set_index('eid', inplace=True)

    codes = df.code.explode().map(code2idx).dropna().rename('code_idx').reset_index()
    phe_scores = X_norm.loc[X_norm.phe_idx == chosen_phe_idx, :]
    score = pd.merge(codes, phe_scores, on=['code_idx'], how='left')

    return score.groupby('eid').value.sum()


phe_train = pd.read_parquet(file_config['data_train'])
phe_val = pd.read_parquet(file_config['data_val'])

phenotype_documents = get_phenotype_documents(phe_train)

# phenotype_documents.to_csv(os.path.join(DATA_DIR, 'processed, 'phenotype_documents.csv'))

phenotype_documents = pd.read_csv(os.path.join(DATA_DIR, 'processed', 'phenotype_documents.csv'), index_col=0,
                                  squeeze=True)

phenotype_documents.to_csv(os.path.join(DATA_DIR, 'processed', 'all', 'phenotype_documents.csv'))
phenotype_documents = phenotype_documents.dropna()

X_norm, code2idx, phe2idx = get_tfidf_weights(phenotype_documents)

save_pickle((X_norm, code2idx, phe2idx), os.path.join(file_config['model_dir'], 'X_norm.pickle'))

X_norm, code2idx, phe2idx = load_pickle(os.path.join(file_config['model_dir'], 'X_norm.pickle'))

scores = get_scores(phe_train, X_norm, code2idx, phe2idx, chosen_phecode)

phe_train = pd.merge(phe_train, scores.rename('scores'), on='eid')

model = GaussianMixture(n_components=2, random_state=0, max_iter=500).fit(phe_train.scores.values.reshape(-1, 1))

phe_train.loc[:, 'predictions'] = model.predict_proba(phe_train.value.values.reshape(-1, 1))[:, 1]
phe_train.loc[:, 'predicted_case'] = phe_train.predictions > 0.5

# Some plots

phe_train.loc[:, 'case'] = phe_train.phecode.apply(lambda x: chosen_phecode in x)

phe_train.boxplot(column=['value'], by='case')

phe_train.boxplot(column=['value'], by='predicted_case')

x = np.linspace(0, 125, 5000).reshape((-1, 1))
y = model.score_samples(x)
plt.plot(x, np.exp(y), label='pdf')
phe_train[phe_train.scores < 125].value.hist(density=True, bins=100)
plt.legend()
plt.show()


import sklearn
print('AUROC: {}'.format(sklearn.metrics.roc_auc_score(phe_train.case, phe_train.predictions, average='weighted')))
print('AUPRC: {}'.format(sklearn.metrics.average_precision_score(phe_train.case, phe_train.predictions, average='weighted')))
