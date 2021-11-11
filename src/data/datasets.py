import os

import re
import pandas as pd
import numpy as np
import torch
from torch.utils.data.dataset import Dataset
import pytorch_lightning as pl

from omni.common import load_pickle


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
    """
    Alternates between visits
    :param tokens:
    :param symbol:
    :return:
    """
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
    """
    Increments per vist
    :param tokens:
    :param symbol:
    :return:
    """
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


def get_random_mask(phecode_idx, mask_prob=0.12, rand_prob=0.015):
    """

    :param phecode_idx:
    :param token2idx:
    :param mask_prob:
    :param rand_prob: TODO
    :return:
    """
    output_mask = []
    output_mask_label = []
    output_phecode_idx = []

    for i, idx in enumerate(phecode_idx):
        # exclude special symbols from masking
        if idx in (0, 1, 2, 3, 4):  # PAD MASK SEP CLS UNK
            # output_mask.append(1)
            output_mask_label.append(-1)
            output_phecode_idx.append(idx)
            continue
        prob = random.random()

        if prob < mask_prob:
            # mask with 0 which means do not attend to this value (effectively drops value)
            # output_mask.append(0)  # do not attend masked value
            output_mask_label.append(idx)  # add label for loss calc
            output_phecode_idx.append(1)  # change token to mask token
        else:
            # output_mask.append(1)  # attend this value
            output_mask_label.append(-1)  # exclude form loss func
            output_phecode_idx.append(idx)  # keep original token if not masked

    return output_phecode_idx, output_mask_label


# class PheDataModule(pl.LightningDataModule):
#     def __init__(self,
#                  code_vocab_path, age_vocab_path, phe_vocab_path, train_data_path, val_data_path, max_len_seq,
#                  debug=False):
#         super().__init__()
#         code_vocab = load_pickle(code_vocab_path)
#         age_vocab = load_pickle(age_vocab_path)
#         phe_vocab = load_pickle(phe_vocab_path)
#
#         model_token2idx = phe_vocab['token2idx']
#
#         train_data = pd.read_parquet(train_data_path)
#         val_data = pd.read_parquet(val_data_path)
#         train_data = train_data.head(10_000) if debug else train_data
#         val_data = val_data.head(1_000) if debug else val_data
#
#         train_dataset = PheDataset(train_data, code_vocab['token2idx'], age_vocab['token2idx'],
#                                    max_len=max_len_seq,
#                                    phe2idx=phe_vocab['token2idx'])
#         val_dataset = PheDataset(val_data, code_vocab['token2idx'], age_vocab['token2idx'],
#                                  max_len=max_len_seq,
#                                  phe2idx=phe_vocab['token2idx'])


import random


def flip(p):
    return random.random() < p


class PheDataset(Dataset):
    def __init__(self, target_token, dataframe, token2idx, age2idx, max_len, drop_mask=True, token_col='code',
                 age_col='age',
                 phecode_col='phecode',
                 phe2idx=None, case_noise=0, control_noise=0,
                 mlm=False):
        """

        :param target_token:
        :param dataframe:
        :param token2idx:
        :param age2idx:
        :param max_len:
        :param drop_mask: drop phecode idx which are unmapped, these are represented with 'MASK' Symbol
        :param token_col:
        :param age_col:
        :param phecode_col:
        :param phe2idx:
        :param case_noise: number between 0-1 with 1 being all cases flipped to controls
        :param control_noise: number between 0-1 with 1 being all controls flipped to cases
        :param mlm: instead of returning label, return masks for mlm
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

        self.target_token = target_token
        self.target_token_list = [k for k in phe2idx.keys() if
                                  re.match(target_token, k) is not None] if target_token is not None else None

        self.case_noise = case_noise
        self.control_noise = control_noise

        self.mlm = mlm

        n_noised_cases = int(len(self) * case_noise)
        n_noised_controls = int(len(self) * control_noise)
        self.case_noise_list = [True] * n_noised_cases + [False] * (len(self) - n_noised_cases)
        self.control_noise_list = [True] * n_noised_controls + [False] * (len(self) - n_noised_controls)
        random.shuffle(self.case_noise_list)
        random.shuffle(self.control_noise_list)

    def __getitem__(self, index):
        """
        return: age_col, code_col, position, segmentation, mask, label
        """

        # extract data
        age = self.age.iloc[index][(-self.max_len + 1):]
        tokens = self.tokens.iloc[index][(-self.max_len + 1):]

        if self.phecode is not None:
            phecode = self.phecode.iloc[index][(-self.max_len + 1):]
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
                phecode = drop_mask(
                    phecode)  # Slightly confusing as mask here only drop tokens which are too small to be included in vocab
            mask = np.ones(self.max_len)
            mask[len(phecode):] = 0  # this is the attention mask to pad which is also excluded in embeddings? https://github.com/huggingface/transformers/issues/205
            phecode = pad_sequence(phecode, self.max_len)
            phecode_idx = get_token2idx(phecode, self.phe2idx, drop_mask=self.drop_mask)
        else:
            phecode_idx = None

        position = position_idx(tokens)
        segment = index_seg(tokens)

        if self.mlm:
            phecode_idx, mask_labels = get_random_mask(phecode_idx)
            return torch.LongTensor(token_idx), torch.LongTensor(age_idx), torch.LongTensor(position), torch.LongTensor(
                segment), torch.LongTensor(phecode_idx), torch.LongTensor(mask), torch.LongTensor(mask_labels)
        else:
            if self.target_token:
                label = any(t in phecode for t in self.target_token_list)
            else:
                label = False

            if self.case_noise != 0 and label == True:
                if self.case_noise_list[index]:  # we want True flip to make label False
                    label = False
            if self.control_noise != 0 and label == False:
                if self.control_noise_list[index]:  # we want True flip to make label True
                    label = True
            return torch.LongTensor(token_idx), torch.LongTensor(age_idx), torch.LongTensor(position), torch.LongTensor(
                segment), torch.LongTensor(phecode_idx), torch.LongTensor([label])

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
