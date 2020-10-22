# Stamatiadis Stefanos, AM2843

import pandas as pd
import numpy as np
import matplotlib.pyplot
from utils import load_dataset
from scipy.special import gamma

def test_statistic(do=None, de=None):
    DO = do.flatten()
    DE = de.flatten()
    return np.divide(np.power(DO - DE, 2), DE).sum()

def chi_pdf(t, df=1):
    return(t**((df-2)/2) * np.exp(-t/2)) / \
        2**(df/2) * gamma(df/2)

def get_pval_per_feature(do=None, de=None):
    p_values = np.zeros((1, do.shape[1]), dtype=float)
    for feature in range(do.shape[1]):
        # Calculate test statistic:
        T = test_statistic(
            do=do[:, feature],
            de=de[:, feature]
        )
        df = 1  # (data_expected.shape[0]-1) * (data_expected.shape[1])
        p_values[0, feature] = chi_pdf(T, df)
    return p_values

X, Y = load_dataset()
n_samples = X.size

# Compute pvalues per feature:
D = np.unique(X)
data_observed = np.zeros((2, D.size), dtype=int)
# Manually compute observed relative frequencies:
data_observed[0, :],  *_ = np.histogram(X[np.where(Y == 0)[0], :].flatten(),
                           np.append(D, D[-1]+1) )
data_observed[1, :],  *_ = np.histogram(X[np.where(Y == 1)[0], :].flatten(),
                                        np.append(D, D[-1]+1) )


data_probs = data_observed / n_samples
x_marginals = data_probs.sum(axis=0)
y_marginals = data_probs.sum(axis=1)

data_expected = np.zeros((2, D.size))
for j in range(D.size):
    for i in range(2):
        data_expected[i, j] = x_marginals[j] * y_marginals[i] * n_samples

# Assignment 3a: Use the chi squared test
p_values = get_pval_per_feature(
    do=data_observed,
    de=data_expected
)


# Assignment 3b: Permutation testing
iterations = 100
for iter in range(iterations):
    # Randomly permute X columns:

    # Manually compute observed relative frequencies:
    data_observed[0, :], *_ = np.histogram(X[np.where(Y == 0)[0], :].flatten(),
                                           np.append(D, D[-1] + 1))
    data_observed[1, :], *_ = np.histogram(X[np.where(Y == 1)[0], :].flatten(),
                                           np.append(D, D[-1] + 1))
    # Keep the pvals:
    p_values = get_pval_per_feature(
        do=data_observed,
        de=data_expected
    )


print("Pronto!")
