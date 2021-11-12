# -*- coding: utf-8 -*-
import os

import random
import logging
from datetime import timedelta

import numpy as np
import pandas as pd
from sacred import Experiment
from sacred.utils import apply_backspaces_and_linefeeds
from sklearn.model_selection import train_test_split
from ast import literal_eval
import pyarrow

from data.preprocess.utils import fit_vocab
from definitions import DATA_DIR, EXTERNAL_DATA_DIR, ROOT_DIR
from src.omni.common import save_pickle, load_pickle

random.seed(42)
MAX_CODE_LENGTH = 4

logger = logging.getLogger(__name__)

PADDING_TOKEN = 'PAD'
SEPARATOR_TOKEN = 'SEP'
UNKNOWN_TOKEN = 'UNK'
MASK_TOKEN = 'MASK'

import argparse

base = os.path.basename(__file__)
experiment_name = os.path.splitext(base)[0]
ex = Experiment(experiment_name)
ex.captured_out_filter = apply_backspaces_and_linefeeds


@ex.config
def config():
    # Patient data
    patient_base_raw_path = os.path.join(DATA_DIR, 'raw', 'application58356', 'patient_base.csv'),
    event_data_path = os.path.join(DATA_DIR, 'raw', 'application58356', 'hesin.tsv'),
    diag_event_data_path = os.path.join(DATA_DIR, 'raw', 'application58356', 'hesin_diag.tsv'),
    opcs_event_data_path = os.path.join(DATA_DIR, 'raw', 'application58356', 'hesin_oper.tsv'),
    gp_event_data_path = os.path.join(DATA_DIR, 'raw', 'application58356', 'gp_clinical.tsv'),
    # External data (None patient sensitive)
    caliber_secondary_care_dict_path = os.path.join(EXTERNAL_DATA_DIR, 'caliber_secondary_care_dict.csv'),
    icd10_phecode = os.path.join(EXTERNAL_DATA_DIR, 'phecode_icd10.csv'),
    read2_phecode = os.path.join(EXTERNAL_DATA_DIR, 'read2_to_phecode.csv'),
    readctv3_phecode = os.path.join(EXTERNAL_DATA_DIR, 'readctv3_to_phecode.csv')
    verbose = True


class BiobankDataset:
    def __init__(self,
                 # Patient data
                 patient_base_raw_path=os.path.join(DATA_DIR, 'raw', 'application58356', 'patient_base.csv'),
                 event_data_path=os.path.join(DATA_DIR, 'raw', 'application58356', 'hesin.tsv'),
                 diag_event_data_path=os.path.join(DATA_DIR, 'raw', 'application58356', 'hesin_diag.tsv'),
                 opcs_event_data_path=os.path.join(DATA_DIR, 'raw', 'application58356', 'hesin_oper.tsv'),
                 gp_event_data_path=os.path.join(DATA_DIR, 'raw', 'application58356', 'gp_clinical.tsv'),
                 # External data (None patient sensitive)
                 caliber_secondary_care_dict_path=os.path.join(EXTERNAL_DATA_DIR, 'caliber_secondary_care_dict.csv'),
                 icd10_phecode=os.path.join(EXTERNAL_DATA_DIR, 'phecode_icd10.csv'),
                 read2_phecode=os.path.join(EXTERNAL_DATA_DIR, 'read2_to_phecode.csv'),
                 readctv3_phecode=os.path.join(EXTERNAL_DATA_DIR, 'readctv3_to_phecode.csv'),
                 verbose=True,
                 ):
        self.patient_base_raw_path = patient_base_raw_path
        self.patient_base_basic_path = os.path.join('interim', 'patient_base_basic.csv')
        self.event_data_path = event_data_path
        self.diag_event_data_path = diag_event_data_path
        self.opcs_event_data_path = opcs_event_data_path
        self.gp_event_data_path = gp_event_data_path
        self.caliber_secondary_care_dict_path = caliber_secondary_care_dict_path
        self.icd10_phecode = icd10_phecode
        self.read2_phecode = read2_phecode
        self.readctv3_phecode = readctv3_phecode
        self.verbose = verbose

    def get_patient_table(self) -> pd.DataFrame:
        """
        Extracts the raw patient baseline data. Also adds heart failure case column.
        :return patient_base: A dataframe with all patient level info
        """
        # You might need to run patient_base_and_hf_cohort.py here
        patient_base = pd.read_csv(self.patient_base_raw_path, parse_dates=['dob'])
        return patient_base

    def get_patient_base(self) -> pd.DataFrame:
        """
        Drop excluded participants, patients that died
        and any patients that did not have at least one
        consultation during their entire history.

        :return patient_base_basic: just whats needed for cohort matching
        """
        patient_base = self.get_patient_table()
        # patients must have event data to be included!
        event_data = pd.read_csv(self.event_data_path, delimiter='\t')
        patient_base = patient_base[patient_base.eid.isin(event_data.eid)]

        return patient_base

    def get_patient_events(self, patient_base) -> pd.DataFrame:
        """
        Get patient events from HES and Primary care, massage into a normalised form with each row being an event with
        columns {'eid': 'Int64', 'date': str, 'code_type': str, 'code': str, 'yob': 'Int16', 'age': 'Int16'}. Patients
        must also be in patient base (to calc age from yob).

        Also only keeps events from within study period.

        :param patient_base: must have eid col and yob col.
        :return: patient_event_data
        """
        # TODO: add try with only primary
        id_vars = ['eid', 'ins_index']
        event_data = pd.read_csv(self.event_data_path, delimiter='\t')
        event_data = event_data.loc[:, id_vars + ['epistart']]

        diag_data = pd.read_csv(self.diag_event_data_path, delimiter='\t')
        opcs_data = pd.read_csv(self.opcs_event_data_path, delimiter='\t', encoding='latin')
        gp_clinical = pd.read_csv(self.gp_event_data_path, delimiter='\t', encoding='latin')
        gp_clinical.drop(columns=['data_provider', 'value1', 'value2', 'value3'], inplace=True)

        diag_data = pd.merge(diag_data, event_data, on=id_vars)  # diag_data = diag_data[diag_data.level != 3]
        diag_data = diag_data[diag_data.level != 3]  # remove external diagnosis codes (only 2% of total)

        # massage into one df
        desired_columns = ['eid', 'date', 'code_type', 'code']
        diag_data.loc[:, 'code'] = diag_data.diag_icd10.str[:MAX_CODE_LENGTH]  # remove extra subsub chapter details
        opcs_data.loc[:, 'code'] = opcs_data.oper4
        gp_clinical.loc[:, 'code'] = gp_clinical.read_3.copy()
        gp_clinical.code.fillna(gp_clinical.read_2, inplace=True)

        diag_data.loc[:, 'code_type'] = diag_data.level.map({1: 'diag_primary', 2: 'diag_secondary'})
        opcs_data.loc[:, 'code_type'] = opcs_data.level.map({1: 'oper_primary', 2: 'oper_secondary'})
        gp_clinical.loc[:, 'code_type'] = 'read_3'
        gp_clinical.loc[gp_clinical[gp_clinical.read_2.notna()].index, 'code_type'] = 'read_2'

        diag_data.loc[:, 'date'] = diag_data.epistart
        opcs_data.loc[:, 'date'] = opcs_data.opdate
        gp_clinical.loc[:, 'date'] = gp_clinical.event_dt

        diag_data = diag_data.loc[:, desired_columns]
        opcs_data = opcs_data.loc[:, desired_columns]
        gp_clinical = gp_clinical.loc[:, desired_columns]

        patient_event_data = pd.concat([diag_data, opcs_data, gp_clinical], axis=0)

        patient_event_data = pd.merge(patient_event_data, patient_base.loc[:, ['eid', 'yob']], how='inner',
                                      on='eid')
        patient_event_data = patient_event_data[patient_event_data.code.notna()]
        patient_event_data.date = pd.to_datetime(patient_event_data.date)
        patient_event_data.loc[:, 'age'] = patient_event_data.date.dt.year - patient_event_data.yob

        return patient_event_data

    def caliber_convert(self, patient_event_data):
        secondary_caliber = pd.read_csv(self.caliber_secondary_care_dict_path)
        icd_caliber_dict = dict([(k, v) for k, v in zip(secondary_caliber.ICD10code, secondary_caliber.Disease)])
        opcs_caliber_dict = dict([(k, v) for k, v in zip(secondary_caliber.OPCS4code, secondary_caliber.Disease)])

        patient_event_data.loc[:, 'caliber'] = ''
        # match of chapter first
        patient_event_data.loc[patient_event_data.code_type == 'diag_icd10', 'caliber'] = patient_event_data[
                                                                                              patient_event_data.code_type == 'diag_icd10'].code.str[
                                                                                          :3].map(icd_caliber_dict)
        # then sub chapter
        patient_event_data.loc[patient_event_data.code_type == 'diag_icd10', 'caliber'] = patient_event_data[
            patient_event_data.code_type == 'diag_icd10'].code.map(icd_caliber_dict)
        patient_event_data.loc[patient_event_data.code_type == 'secondary_icd10', 'caliber'] = patient_event_data[
            patient_event_data.code_type == 'secondary_icd10'].code.map(icd_caliber_dict)
        patient_event_data.loc[patient_event_data.code_type == 'oper4', 'caliber'] = patient_event_data[
            patient_event_data.code_type == 'oper4'].code.map(opcs_caliber_dict)
        return patient_event_data

    def phecode_convert(self, patient_event_data, many_to_one=True):
        icd_phecode = pd.read_csv(self.icd10_phecode)
        icd_phecode.columns = ['code', 'description', 'phecode', 'phenotype', 'exclude_range_codes', 'exclude_range']
        icd_phecode.code = icd_phecode.code.str.replace('.', '')
        icd_phecode.loc[:, 'code_type_match'] = 'diag'
        icd_phecode = icd_phecode.loc[:, ['code', 'code_type_match', 'phecode']]

        read2_phecode = pd.read_csv(self.read2_phecode)
        read2_phecode.columns = ['code', 'phecode']
        read2_phecode.loc[:, 'code_type_match'] = 'read_2'
        read2_phecode = read2_phecode.loc[:, ['code', 'code_type_match', 'phecode']]

        read3_phecode = pd.read_csv(self.readctv3_phecode)
        read3_phecode.columns = ['code', 'phecode']
        read3_phecode.loc[:, 'code_type_match'] = 'read_3'
        read3_phecode = read3_phecode.loc[:, ['code', 'code_type_match', 'phecode']]

        phecode_lookup = pd.concat([icd_phecode, read2_phecode, read3_phecode], axis=0)
        # phecode_lookup.to_csv(os.path.join(EXTERNAL_DATA_DIR, 'phecode_primary_secondary_lookup.csv'), index=False)
        if not many_to_one:
            phecode_lookup = phecode_lookup[~phecode_lookup.loc[:, ['code', 'code_type_match']].duplicated()]

        patient_event_data.loc[:, 'code_type_match'] = patient_event_data.code_type.copy()
        bool_diag = patient_event_data.code_type.str.contains(r'diag_secondary|diag_primary', regex=True)
        patient_event_data.loc[bool_diag, 'code_type_match'] = 'diag'
        patient_event_data.loc[bool_diag, 'code'] = patient_event_data.loc[bool_diag, 'code'].str[:4]

        patient_event_data = pd.merge(patient_event_data, phecode_lookup, on=['code', 'code_type_match'], how='left')

        return patient_event_data

    @staticmethod
    def get_token_to_index_vocabulary(patient_event_data, indexing_col='age', padding_token=PADDING_TOKEN,
                                      separator_token=SEPARATOR_TOKEN, unknown_token=UNKNOWN_TOKEN,
                                      mask_token=MASK_TOKEN):
        tokens = patient_event_data[indexing_col]
        unique_tokens = list(tokens.unique())
        idx2token = dict(enumerate([padding_token, separator_token, unknown_token, mask_token] + unique_tokens))
        token2idx = dict([(v, k) for k, v in idx2token.items()])
        # patient_event_data.loc[:, indexing_col] = patient_event_data[indexing_col].map(token2idx)
        vocab = {'idx2token': idx2token,
                 'token2idx': token2idx}
        return vocab

    @staticmethod
    def get_sequences_df(df, code_col='code', phe_col='phecode', order_by='date', group_sequences=('eid',)):
        if order_by:
            df.loc[:, order_by] = pd.to_datetime(df.loc[:, order_by])
            df = df.sort_values(['eid', order_by])
        code_lists = df.groupby(list(group_sequences))[code_col].apply(list)
        phe_lists = df.groupby(list(group_sequences))[phe_col].apply(list)
        age_lists = df.groupby(list(group_sequences))['age'].apply(list)
        # df.groupby(list(group_sequences))[code_col, phe_col, 'age'].apply(list)
        df = pd.concat([age_lists, code_lists, phe_lists], axis=1)
        df = df.reset_index(drop=False)
        return df

    @staticmethod
    def add_separator(patient_event_data, sep_vars=('eid', 'date'), code_cols=('code', 'phecode',),
                      separator_token=SEPARATOR_TOKEN):
        """

        :param patient_event_data:
        :param sep_vars:
        :param code_col:
        :param separator_token:
        :return:
        """
        patient_event_data.loc[:, 'sorting_var'] = 0  # used to ensure token at the end of every date
        sep_rows_index = patient_event_data.loc[:, sep_vars].drop_duplicates().index
        sep_rows = patient_event_data.loc[sep_rows_index, :].copy()
        sep_rows.loc[:, code_cols] = separator_token
        sep_rows.loc[:, 'age'] = separator_token
        sep_rows.loc[:, 'code_type'] = 'token'
        sep_rows.loc[:, 'sorting_var'] = 1
        patient_event_data = pd.concat([patient_event_data, sep_rows], axis=0)
        patient_event_data = patient_event_data.sort_values(by=['eid', 'date', 'sorting_var'])
        patient_event_data.drop(columns='sorting_var', inplace=True)
        patient_event_data.reset_index(inplace=True)
        return patient_event_data

    @staticmethod
    def add_unknown(patient_event_data, min_proportion=0.01, unknown_token='UNK'):
        # TODO: add method to turn infrequent tokens into UNK tokens take out of vocab func
        pass

    def get_phe_data(self, patient_event_data=None, reload=True):
        """
        Gets a df with only event that have phe code mapping, returns in sequence format such that each row contain a
        patient record for each eid with lists of ages and phecodes
        :param patient_event_data: a df with each row as an event
        :param reload: load up from alraedy processeed version
        :return:
        """
        if reload:
            d_types = {'eid': 'Int64', 'date': str, 'code_type': str, 'code': str, 'yob': 'Int16', 'age': 'Int16'}
            patient_event_data = pd.read_csv(os.path.join(DATA_DIR, 'interim', 'patient_event_data.csv'),
                                             dtype=d_types, parse_dates=['date'])
            patient_event_data = self.phecode_convert(patient_event_data)
            patient_event_data = patient_event_data[patient_event_data.age > 0]
        phe_data = patient_event_data[patient_event_data.phecode.notna()]
        phe_data = self.add_separator(phe_data)
        # TODO: add filter for tokens with small volume. add_unknown
        age_vocab = self.get_token_to_index_vocabulary(phe_data, indexing_col='age')
        code_vocab = self.get_token_to_index_vocabulary(phe_data, indexing_col='phecode')
        return phe_data, age_vocab, code_vocab

    def phe_lists(self, with_codes, reload=True):
        if reload:
            phe_data, age_vocab, code_vocab = self.get_phe_data(reload=reload)
            save_pickle(age_vocab, os.path.join(DATA_DIR, 'processed', with_codes, 'age_vocab.pkl'))
            save_pickle(code_vocab, os.path.join(DATA_DIR, 'processed', with_codes, 'code_vocab.pkl'))
            phe_lists = self.get_sequences_df(phe_data)
            return phe_lists, age_vocab, code_vocab
        if not reload:
            phe_lists = pd.read_csv(os.path.join(DATA_DIR, 'processed', with_codes, 'phe_lists.csv'))
            phe_lists.age = phe_lists.age.apply(literal_eval)
            phe_lists.phecode = phe_lists.phecode.apply(literal_eval)
            age_vocab = load_pickle(os.path.join(DATA_DIR, 'processed', with_codes, 'age_vocab.pkl'))
            code_vocab = load_pickle(os.path.join(DATA_DIR, 'processed', with_codes, 'code_vocab.pkl'))
            return phe_lists, age_vocab, code_vocab


def train_val_test(df_input, stratify_colname=None, frac_train=0.6, frac_val=0.2, frac_test=0.2):
    '''
    Splits a Pandas dataframe into three subsets (train, val, and test)
    following fractional ratios provided by the user, where each subset is
    stratified by the values in a specific column (that is, each subset has
    the same relative frequency of the values in the column). It performs this
    splitting by running train_test_split() twice.

    Parameters
    ----------
    df_input : Pandas dataframe
        Input dataframe to be split.
    stratify_colname : str
        The name of the column that will be used for stratification. Usually
        this column would be for the label. If None then no stratification
    frac_train : float
    frac_val   : float
    frac_test  : float
        The ratios with which the dataframe will be split into train, val, and
        test data. The values should be expressed as float fractions and should
        sum to 1.0.
    random_state : int, None, or RandomStateInstance
        Value to be passed to train_test_split().

    Returns
    -------
    df_train, df_val, df_test :
        Dataframes containing the three splits.
    '''

    if frac_train + frac_val + frac_test != 1.0:
        raise ValueError('fractions %f, %f, %f do not add up to 1.0' % \
                         (frac_train, frac_val, frac_test))

    if stratify_colname is None:
        # Split original dataframe into train and temp dataframes.
        df_train, df_temp = train_test_split(df_input, test_size=(1.0 - frac_train))

        # Split the temp dataframe into val and test dataframes.
        relative_frac_test = frac_test / (frac_val + frac_test)
        df_val, df_test = train_test_split(df_temp, test_size=relative_frac_test)

        assert len(df_input) == len(df_train) + len(df_val) + len(df_test)

        return df_train, df_val, df_test

    if stratify_colname not in df_input.columns:
        raise ValueError('%s is not a column in the dataframe' % (stratify_colname))

    X = df_input  # Contains all columns.
    y = df_input[[stratify_colname]]  # Dataframe of just the column on which to stratify.

    # Split original dataframe into train and temp dataframes.
    df_train, df_temp, y_train, y_temp = train_test_split(X,
                                                          y,
                                                          stratify=y,
                                                          test_size=(1.0 - frac_train))

    # Split the temp dataframe into val and test dataframes.
    relative_frac_test = frac_test / (frac_val + frac_test)
    df_val, df_test, y_val, y_test = train_test_split(df_temp,
                                                      y_temp,
                                                      stratify=y_temp,
                                                      test_size=relative_frac_test)

    assert len(df_input) == len(df_train) + len(df_val) + len(df_test)

    return df_train, df_val, df_test


@ex.automain
def main(
        patient_base_raw_path=os.path.join(DATA_DIR, 'raw', 'application58356', 'patient_base.csv'),
        event_data_path=os.path.join(DATA_DIR, 'raw', 'application58356', 'hesin.tsv'),
        diag_event_data_path=os.path.join(DATA_DIR, 'raw', 'application58356', 'hesin_diag.tsv'),
        opcs_event_data_path=os.path.join(DATA_DIR, 'raw', 'application58356', 'hesin_oper.tsv'),
        gp_event_data_path=os.path.join(DATA_DIR, 'raw', 'application58356', 'gp_clinical.tsv'),
        # External data (None patient sensitive)
        caliber_secondary_care_dict_path=os.path.join(EXTERNAL_DATA_DIR, 'caliber_secondary_care_dict.csv'),
        icd10_phecode=os.path.join(EXTERNAL_DATA_DIR, 'phecode_icd10.csv'),
        read2_phecode=os.path.join(EXTERNAL_DATA_DIR, 'read2_to_phecode.csv'),
        readctv3_phecode=os.path.join(EXTERNAL_DATA_DIR, 'readctv3_to_phecode.csv'),
        verbose=True):
    from src.omni.common import save_pickle
    from ast import literal_eval

    # TODO: add argparse support
    with_codes = 'phecode'  # 'phecode'

    b = BiobankDataset(
        patient_base_raw_path,
        event_data_path,
        diag_event_data_path,
        opcs_event_data_path,
        gp_event_data_path,
        caliber_secondary_care_dict_path,
        icd10_phecode,
        read2_phecode,
        readctv3_phecode,
        verbose
    )
    raw_patient_table = b.get_patient_base()
    print('Raw. Total patients: {}'.format(raw_patient_table.eid.unique().shape[0]))

    d_types = {'eid': 'Int64', 'date': str, 'code_type': str, 'code': str, 'yob': 'Int16', 'age': str}
    patient_event_data = pd.read_csv(os.path.join(DATA_DIR, 'interim', 'patient_event_data.csv'),
                                     dtype=d_types, parse_dates=['date'])
    print('Within HES or primary care data. Total patients: {}, Total events: {}'.format(
        patient_event_data.eid.unique().shape[0],
        patient_event_data.shape[0]))

    patient_event_data = patient_event_data[(patient_event_data.date.dt.year > 1950) &
                                            (patient_event_data.date.dt.year < 2021)]
    print('Within Study period. Total patients: {}, Total events: {}'.format(patient_event_data.eid.unique().shape[0],
                                                                             patient_event_data.shape[0]))
    patient_event_data = b.phecode_convert(patient_event_data)

    if with_codes == 'all':
        phe_data = patient_event_data
        phe_data.phecode = phe_data.phecode.astype(str).fillna(MASK_TOKEN).astype('category')
    elif with_codes == 'phecode':
        phe_data = patient_event_data[patient_event_data.phecode.notna()]

    print('Post Phecode mapping. Total patients: {}, Total events: {}'.format(phe_data.eid.unique().shape[0],
                                                                              phe_data.shape[0]))
    phe_data = b.add_separator(phe_data)
    phe_data.to_csv(os.path.join(DATA_DIR, 'processed', with_codes, 'phe_data.csv'), index=False)
    phe_data = pd.read_csv(os.path.join(DATA_DIR, 'processed', with_codes, 'phe_data.csv'), parse_dates=['date'],
                           dtype={"phecode": 'category', "code": 'category', "age": 'category',
                                  'code_type': 'category', 'eid': 'int64', 'yob': "int16"})

    phe_lists = b.get_sequences_df(phe_data)
    phe_lists['length'] = phe_lists['phecode'].apply(
        lambda x: len([i for i in range(len(x)) if x[i] == SEPARATOR_TOKEN]))
    phe_lists = phe_lists[phe_lists['length'] >= 5]
    print('More than 5 events. Total patients: {}, Total events: {}'.
          format(phe_lists.shape[0], (phe_lists.phecode.explode().notin(SEPARATOR_TOKEN, MASK_TOKEN)).sum()))
    phe_lists.to_parquet(os.path.join(DATA_DIR, 'processed', with_codes, 'phe_lists.parquet'), index=False)
    phe_lists = pd.read_parquet(os.path.join(DATA_DIR, 'processed', with_codes, 'phe_lists.parquet'))

    phe_train, phe_val, phe_test = train_val_test(phe_lists)

    code_vocab = fit_vocab(phe_train.code.explode().astype(str), min_count=10)
    phecode_vocab = fit_vocab(phe_train.phecode.explode().astype(str), min_proportion=0.0001)
    age_vocab = fit_vocab(phe_train.age.explode().astype(str))

    save_pickle(age_vocab, os.path.join(DATA_DIR, 'processed', with_codes, 'age_vocab.pkl'))
    save_pickle(code_vocab, os.path.join(DATA_DIR, 'processed', with_codes, 'code_vocab.pkl'))
    save_pickle(phecode_vocab, os.path.join(DATA_DIR, 'processed', with_codes, 'phecode_vocab.pkl'))

    for df, name in zip([phe_train, phe_val, phe_test], ['phe_train.parquet', 'phe_val.parquet', 'phe_test.parquet']):
        # df = pd.read_parquet(os.path.join(DATA_DIR, 'processed', with_codes, 'MLM', name))
        df.to_parquet(os.path.join(DATA_DIR, 'processed', with_codes, 'MLM', name), index=False)
        print('mlm {}. Total patients {}, Total events {}'.format(name, df.shape[0], df.phecode.explode().shape[0]))
