import os
import numpy as np


def generate_data(n_patients, mean_seq=25,
                  minor_allele_frequency=0.2,
                  transition_coeff=[-5, 1],
                  genetic_coeff=0.9,
                  num_states_Y=2,
                  num_states_G=3,
                  emission=[[0.6, 0.4, 0.3, 0.2, 0.1, 0.01, 0.001, 0.1, 0.01, 0.001],
                            [0.002, 0.3, 0.1, 0.1, 0.6, 0.2, 0.1, 0.001, 0.1, 0.1]],
                  g_dist='bernoulli'
                  ):
    """
    Generate data using markov model with genetic prior component to transition matrix
    :param n_patients:
    :param mean_seq:
    :param init_state:
    :param transition_coeff: [beta1, beta2] p = sigmoid(beta) if no genetic coeff, p1 is probability to transition
    :param genetic_coeff:
    :param num_states_Y:
    :param num_states_G:
    :param emission:
    :return:
    """

    transition_coeff = np.array(transition_coeff)
    genetic_coeff = np.array([genetic_coeff, -genetic_coeff])  # when in disease effect on transtion is reversed

    emission = np.array(emission)
    # seq_lens = scale_shift_lognormal(min_len=min_seq, mean_len=mean_seq, size=(n_patients,))

    X_ = []
    Y_ = []
    if g_dist == 'binomial':
        G_ = np.random.binomial(num_states_G - 1, minor_allele_frequency, size=(n_patients,))
        G_ = G_ - G_.mean()
    elif g_dist == 'bernoulli':
        G_ = np.random.choice([0, 1], p=[1 - minor_allele_frequency, minor_allele_frequency], size=(n_patients,))

    for k in range(n_patients):
        # seq_len = seq_lens[k]
        seq_len = mean_seq
        Y_new = []
        X_new = []

        # For every record in the patient sequence
        for i in range(int(seq_len)):

            # Sample state from transition model
            if i == 0:
                # p1, p2 = sigmoid(transition_coeff[0]), sigmoid(transition_coeff[1])
                # init_p1, init_p2 = p2 / (p1 + p2), p1 / (p1 + p2)  # for steady MM
                # Y_new.append(np.random.choice(num_states_Y, 1, p=[init_p1, init_p2])[0])
                state = 0
            else:
                state = Y_new[-1]
            z = transition_coeff[state] + genetic_coeff[state] * G_[k]
            p_trans_new = sigmoid(z)
            Y_new.append(-1 * (state - 1) if np.random.binomial(n=1, p=p_trans_new) else state)  # bernoulli sample

            # Sample observations from emission model
            X_new.append(np.random.binomial(1, emission[Y_new[-1]]).flatten())
            # X_new.append(np.random.choice(emission.shape[1], 1, p=emission[Y_new[-1]])[0])

        Y_.append(Y_new)
        X_.append(X_new)

    return X_, Y_, G_


def get_attention_weights(seqlength, alpha, reverse_mode):
    """
    From https://github.com/vanderschaarlab/mlforhealthlabpub/blob/main/alg/attentivess/data/make_data.py

    :param seqlength:
    :param alpha:
    :param reverse_mode:
    :return:
    """
    init_weights = np.exp(-alpha * np.array(range(seqlength)))

    if reverse_mode:

        weights_ = init_weights / np.sum(init_weights)

    else:

        weights_ = np.flip(init_weights / np.sum(init_weights))

    return weights_


def calculate_beta(prevalence, seq_len):
    """
    From Prevalence = P(y_any) = \prod^{mean_seq}P(Y_t)
    Gets the required beta to simulate trial_n set prevalence
    :param prevalence:
    :param mean_seq:
    :return:
    """
    # return -np.log((1 / (1 - (1 - prevalence) ** (1 / (seq_len+1)))) - 1)
    return np.log((1 - prevalence) ** (-1 / (seq_len + 1)) - 1)


def calculate_beta_gene(prevalence, seq_len, minor_allele_frequency, odds_ratio):
    T = seq_len + 1

    P = minor_allele_frequency
    R = odds_ratio

    # With joint P(Y|G)P(G)
    A_g = (1 - prevalence) / ((R - 1) * prevalence + 1)
    # With conditional only P(Y|G)
    A = -(prevalence - 1) * (P - R * prevalence + prevalence - 1) / ((P - 1) * ((R - 1) * prevalence + 1))

    beta = np.log(A ** (-1 / T) - 1)
    beta_g = np.log(A_g ** (-1 / T) - 1) - beta

    # OR check
    # print((1 - A_g) * (1 - prevalence) /  (A_g * prevalence))
    # Prevalence check
    # print((1-A_g)*P + (1-A)*(1-P))
    return beta, beta_g


def sigmoid(x):
    return np.exp(-np.logaddexp(0, -x))

# def calculate_prevalence(beta, seq_len):
#     return 1 - (1 - beta) ** (seq_len)
# def get_loggnormal_beta(prevalence, min_len=5, mean_len=20, sampling_size=(1_000_000,)):
#     seq_len_distribution = scale_shift_lognormal(min_len=min_len, mean_len=mean_len, size=sampling_size)
#     beta_dist = calculate_beta(prevalence, seq_len_distribution)
#     return beta_dist.mean()
# def scale_shift_lognormal(min_len=5, mean_len=20, size=(1,)):
#     mu_0 = np.exp(0.5)
#     scale = (mean_len - min_len) / mu_0
#     shift = min_len
#     return np.round(shift + scale * np.random.lognormal(mean=0, sigma=1, size=size))
