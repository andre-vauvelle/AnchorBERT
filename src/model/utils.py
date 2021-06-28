import numpy as np
import pandas as pd


def flatten_generated_data(X, Y=None, G=None, code_idx=0) -> pd.DataFrame:
    """
    Flattens the sequential generated data to fit the R pheprob::simdata format
    :param X:
    :param Y:
    :param G:
    :return: pd.DataFrame
    """
    n = len(X)
    Y_ = np.array([1 if any(y) else 0 for y in Y]) if Y is not None else None
    G_ = np.array(G).flatten() if G is not None else None
    G_ = G_ - G_.mean() if G_ is not None else None

    X_ = np.array([len(x) for x in X])
    S_ = np.zeros(n)
    for i, x in enumerate(X):
        codes = np.stack(x)
        total = codes[:, code_idx].sum()
        S_[i] = total

    return pd.DataFrame({'S': S_,
                         'C': X_,
                         'X': np.zeros(n),
                         'G': G_,
                         'Y': Y_})
