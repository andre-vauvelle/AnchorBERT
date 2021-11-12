from typing import List, Dict

import pandas as pd


def fit_vocab(data: List, min_count=None, min_proportion=None, padding_token='PAD',
              separator_token='SEP', unknown_token='UNK', mask_token='MASK', cls_token='CLS') -> Dict:
    """
    Fits a vocabulary to some data, returns as a dict
    :param data:
    :param min_count:
    :param min_proportion:
    :param padding_token:
    :param separator_token:
    :param unknown_token:
    :param mask_token:
    :param cls_token:
    :return:
    """
    counts = pd.Series(data).value_counts()
    counts.drop([padding_token, mask_token, separator_token, unknown_token, cls_token], inplace=True, errors='ignore')
    proportions = counts / counts.sum()
    if min_count is not None:
        excluded_tokens = set(counts[counts < min_count].index)
    elif min_proportion is not None:
        excluded_tokens = set(proportions[proportions < min_proportion].index)
    else:
        excluded_tokens = set()

    data_tokens = set(data)
    symbol_tokens = {padding_token, separator_token, unknown_token, mask_token, cls_token}
    data_tokens = data_tokens - symbol_tokens - excluded_tokens
    unique_tokens = list(symbol_tokens) + list(data_tokens)
    idx2token = dict(enumerate(unique_tokens))
    token2idx = dict([(v, k) for k, v in idx2token.items()])
    vocab = {'idx2token': idx2token,
             'token2idx': token2idx}
    return vocab
