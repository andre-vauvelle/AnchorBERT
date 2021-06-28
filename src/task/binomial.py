import torch
from torch.distributions import Poisson, LogNormal
from torch.distributions.bernoulli import Bernoulli
from torch.distributions.binomial import Binomial
from torch.distributions.normal import Normal
from torch.distributions.uniform import Uniform


def generate_data(N, minor_allele_frequency, snp_levels, healthcare_util, beta, p_success_bounds):
    """
    Generates synthetic EHR data"""

    # risk allele
    G = Binomial(snp_levels - 1, minor_allele_frequency).sample((N,))
    # clinical covariate
    X = Normal(0, 1).sample((N,))
    # for poisson variable
    lambda_c = Uniform(healthcare_util['min'], healthcare_util['max']).sample((N,))
    # Healthcare utilization
    C = Poisson(lambda_c).sample() + 1

    Y = torch.sigmoid(beta['prevalence'] + beta['geno_pheno'] * G + beta['clincal'] * X - beta['util'] * torch.log(C))

    p_success = torch.Tensor(N)
    for i, y in enumerate(Y):
        y = y.round()
        p_success[i] = Uniform(p_success_bounds[y.item()]['min'], p_success_bounds[y.item()]['max']).sample()

    S = Binomial(C, p_success).sample()
    return S, Y, X


if __name__ == '__main__':
    N = 2000
    minor_allele_frequency = 0.2
    snp_levels = 3
    healthcare_util = {
        'min': 10,
        'max': 500
    }

    beta = {'prevalence': -1.13,
            'geno_pheno': 0.1,
            'clincal': 1,
            'util': 0.1}

    p_success_bounds = {
        1: {
            'min': 0.02,
            'max': 0.3},
        0: {
            'min': 0,
            'max': 0.02}
    }

    S, Y, X = generate_data(N, minor_allele_frequency, snp_levels, healthcare_util, beta, p_success_bounds)
