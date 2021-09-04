import os
import re

import pandas as pd

from definitions import MODEL_DIR, DATA_DIR
from omni.common import load_pickle
import matplotlib.pyplot as plt

if __name__ == '__main__':
    global_params = {'with_codes': 'all'}
    phecode_definitions = pd.read_csv(os.path.join(DATA_DIR, 'vocabs', 'phecode_definitions1.2.csv'))
    phecode_definitions.phecode = phecode_definitions.astype(str)
    model = load_pickle(os.path.join(MODEL_DIR, 'sklearn_regression', 'anchor_250.pkl'))
    phe_vocab = load_pickle(os.path.join(DATA_DIR, 'vocabs', 'phecode_vocab.pkl'))
    idx2token = phe_vocab['idx2token']
    token2idx = phe_vocab['token2idx']
    coefficients = model.coef_

    symbol_tokens = ['PAD', 'MASK', 'CLS', 'UNK']
    token = '250.2'
    phe_tokens = [k for k in token2idx.keys() if re.match(token, k) is not None]
    remove_tokens = symbol_tokens + phe_tokens

    kept_tokens = [t for t in token2idx.keys() if t not in remove_tokens]
    keptidx2token = dict(zip(range(len(kept_tokens)), kept_tokens))

    df = pd.DataFrame({'tokens': kept_tokens, 'coefficients': coefficients.flatten()})
    df = df.sort_values(by='coefficients', ascending=False)

    phecode_definitions.phecode.apply(lambda x: df.tokens.str.match(x).index)
    df = df.merge(phecode_definitions, how='left', left_on='tokens', right_on='phecode')

    df.iloc[:10, [0, 1, 3]]
    # df.iloc[:10]
    # Out[76]:
    #      tokens  coefficients
    # 268   250.7      1.533473
    # 282   250.6      1.316567
    # 320  250.13      0.659046
    # 807   401.1      0.316244
    # 35    681.5      0.282696
    # 435   443.7      0.258147
    # 338   250.1      0.249216
    # 815   251.1      0.163367
    # 703   278.1      0.139532
    df.iloc[-10:, [0, 1, 3]]
    # Out[77]:
    #     tokens  coefficients
    # 686  380.4     -0.091501
    # 350  634.0     -0.091752
    # 779  913.0     -0.097667
    # 460  627.2     -0.100184
    # 634  650.0     -0.113954
    # 484  665.0     -0.127242
    # 641  735.3     -0.135439
    # 0      SEP     -0.254740
    # 365  681.3     -0.285217
    # 198  679.0     -0.340591

    # Get AUROC and average precision
    predictions = pd.read_csv(os.path.join(DATA_DIR, 'anchor_250.2.tsv.gz'), sep='\t')
