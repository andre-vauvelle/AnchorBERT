import numpy as np
from sklearn.metrics import confusion_matrix
from statsmodels.tools.sm_exceptions import PerfectSeparationError
import pandas as pd

import statsmodels.api as sm
import statsmodels.formula.api as smf
import scipy.stats as ss

EPS = 1e-10


def get_odds_ratio(Y_any_, G_):
    """Gets odds ratio, quantifies the strength of the association between two events. Here Disease and Gene"""
    G_ = G_ > 0

    p_dg = sum(np.array(Y_any_)[G_]) / sum(G_)
    odds_dg = p_dg / (1 - p_dg + EPS)

    p_d = sum(Y_any_) / len(Y_any_)
    odds_d = p_d / (1 - p_d + EPS)

    return odds_dg / odds_d


def inverse_normal_rank(series, c=3.0 / 8, stochastic=True):
    """ Perform rank-based inverse normal transformation on pandas series.
        If stochastic is True ties are given rank randomly, otherwise ties will
        share the same value. NaN values are ignored.
        Args:
            param1 (pandas.Series):   Series of values to transform
            param2 (Optional[float]): Constand parameter (Bloms constant)
            param3 (Optional[bool]):  Whether to randomise rank of ties

        Returns:
            pandas.Series
        From https://github.com/edm1/rank-based-INT/blob/master/rank_based_inverse_normal_transformation.py
    """

    # Check input
    assert (isinstance(series, pd.Series))
    assert (isinstance(c, float))
    assert (isinstance(stochastic, bool))

    # Take original series indexes
    orig_idx = series.index

    # Drop NaNs
    series = series.loc[~pd.isnull(series)]

    # Get ranks
    if stochastic == True:
        # Shuffle by index
        series = series.loc[np.random.permutation(series.index)]
        # Get rank, ties are determined by their position in the series (hence
        # why we randomised the series)
        rank = ss.rankdata(series, method="ordinal")
    else:
        # Get rank, ties are averaged
        rank = ss.rankdata(series, method="average")

    # Convert numpy array back to series
    rank = pd.Series(rank, index=series.index)

    # Convert rank to normal distribution
    transformed = rank.apply(rank_to_normal, c=c, n=len(rank))

    return transformed[orig_idx]


def rank_to_normal(rank, c, n):
    # Standard quantile function
    x = (rank - c) / (n - 2 * c + 1)
    return ss.norm.ppf(x)


def get_p_val(outcome, G, inverse_rank=False, family=sm.families.Binomial()):
    data = pd.DataFrame({'outcome': outcome, 'G': G, 'id': range(len(outcome))})
    if inverse_rank:
        data.outcome = inverse_normal_rank(data.outcome)
    try:
        mod = smf.gee('outcome~G', 'id', data=data, family=family)
        res = mod.fit()
        pval = res.pvalues.G
        return pval
    except PerfectSeparationError:
        return None


def get_power(Y, G):
    """Statistical power, or the power of trial_n hypothesis test is the probability that the test correctly rejects the
    null hypothesis."""
    confm = confusion_matrix(Y, G)
    total = confm.sum()
    tn, fp, fn, tp = confm.ravel()
    return tp / total
