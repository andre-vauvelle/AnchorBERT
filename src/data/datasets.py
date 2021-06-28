import os

import pandas as pd
import numpy as np
import torch
from torch.utils.data.dataset import Dataset


def drop_mask(tokens, symbol='MASK'):
    seq = []
    for token in tokens:
        if token == symbol:
            continue
        else:
            seq.append(token)
    return seq


def pad_sequence(tokens, max_len, symbol='PAD'):
    seq = []
    token_len = len(tokens)
    for i in range(max_len):
        if i < token_len:
            seq.append(tokens[i])
        else:
            seq.append(symbol)
    return seq


def index_seg(tokens, symbol='SEP'):
    flag = 0
    seg = []

    for token in tokens:
        if token == symbol:
            seg.append(flag)
            if flag == 0:
                flag = 1
            else:
                flag = 0
        else:
            seg.append(flag)
    return seg


def position_idx(tokens, symbol='SEP'):
    pos = []
    flag = 0

    for token in tokens:
        if token == symbol:
            pos.append(flag)
            flag += 1
        else:
            pos.append(flag)
    return pos


def get_token2idx(tokens, token2idx, drop_mask=False, mask_symbol='MASK'):
    output_idx = []
    for i, token in enumerate(tokens):
        if drop_mask and token == mask_symbol:
            continue
        else:
            output_idx.append(token2idx.get(token, token2idx['UNK']))
    return output_idx


class PheDataset(Dataset):
    def __init__(self, dataframe, token2idx, age2idx, max_len, drop_mask=True, token_col='code', age_col='age',
                 phecode_col='phecode',
                 phe2idx=None, token=None):
        """

        :param dataframe:
        :param token2idx:
        :param age2idx:
        :param max_len:
        :param drop_mask: drop phecode idx which are unmapped, these are represented with 'MASK' Symbol
        :param token_col:
        :param age_col:
        :param phecode_col:
        :param phe2idx:
        :param token:
        """
        self.vocab = token2idx
        self.max_len = max_len
        self.eid = dataframe['eid']
        self.tokens = dataframe[token_col]
        self.age = dataframe[age_col]
        self.age2idx = age2idx
        self.drop_mask = drop_mask
        self.phecode = dataframe[phecode_col] if phecode_col is not None else None
        self.phe2idx = phe2idx if phe2idx is not None else None

    def __getitem__(self, index):
        """
        return: age_col, code_col, position, segmentation, mask, label
        """

        # extract data
        age = self.age[index][(-self.max_len + 1):]
        tokens = self.tokens[index][(-self.max_len + 1):]
        if self.phecode is not None:
            phecode = self.phecode[index][(-self.max_len + 1):]
        else:
            phecode = None

        # avoid data cut with first element to be 'SEP'
        if tokens[0] != 'SEP':
            tokens = np.append(np.array(['CLS']), tokens)
            phecode = np.append(np.array(['CLS']), phecode)
            age = np.append(np.array(age[0]), age)
        else:
            tokens[0] = 'CLS'
            if phecode is not None:
                phecode[0] = 'CLS'

        # pad age_col sequence and code_col sequence
        age = pad_sequence(age, self.max_len)
        tokens = pad_sequence(tokens, self.max_len)
        age_idx = get_token2idx(age, self.age2idx)
        token_idx = get_token2idx(tokens, self.vocab)
        if phecode is not None:
            if self.drop_mask:
                phecode = drop_mask(phecode)
            phecode = pad_sequence(phecode, self.max_len)
            phecode_idx = get_token2idx(phecode, self.phe2idx, drop_mask=self.drop_mask)
        else:
            phecode_idx = None

        position = position_idx(tokens)
        segment = index_seg(tokens)

        return torch.LongTensor(token_idx), torch.LongTensor(age_idx), torch.LongTensor(position), torch.LongTensor(
            segment), torch.LongTensor(phecode_idx)

    def __len__(self):
        return len(self.tokens)


class SimulatedDataset(Dataset):
    def __init__(self, X, max_len):
        reserved_symbol_tokens = 3
        self.token_idxs = pd.Series(np.zeros(len(X))).astype(object)
        for i, record in enumerate(X):
            record_idxs = []
            record_idxs.extend([2])  # 2 is CLS token
            for token in record:
                # Get indices of nonezero
                token_idxs = token.nonzero()[0] + reserved_symbol_tokens
                token_idxs = token_idxs.tolist()
                token_idxs.extend([1])  # 1 is SEP token
                record_idxs.extend(token_idxs)
            record_idxs = pad_sequence(record_idxs, symbol=0, max_len=max_len)  # 0 is PAD token
            self.token_idxs.at[i] = record_idxs
        self.max_len = max_len

    def __getitem__(self, index):
        """
        return: age_col, code_col, position, segmentation, mask, label
        """

        # extract data
        token_idx = self.token_idxs[index][(-self.max_len):]

        # pad age_col sequence and code_col sequence
        position = position_idx(token_idx)
        segment = index_seg(token_idx)

        return torch.LongTensor(token_idx), torch.LongTensor(), torch.LongTensor(position), torch.LongTensor(
            segment), torch.LongTensor(token_idx)

    def __len__(self):
        return self.token_idxs.shape[0]
