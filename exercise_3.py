# Stamatiadis Stefanos, AM2843

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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

def get_test_statistic_per_feature(do=None, de=None):
    t_values = np.zeros((1, do.shape[1]), dtype=float)
    for feature in range(do.shape[1]):
        # Calculate test statistic:
        t_values[0, feature] = test_statistic(
            do=do[:, feature],
            de=de[:, feature]
        )
    return t_values

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

t_vals_obs = get_test_statistic_per_feature(
    do=data_observed,
    de=data_expected
)

# Assignment 3b: Permutation testing
iterations = 100
t_values = np.zeros((iterations, D.size))
for iter in range(iterations):
    # Randomly permute X columns:
    # Numpy nuances....
    rnd_idx = np.zeros(X.shape, dtype=int)
    for i in range(X.shape[1]):
        rnd_idx[:, i] = np.random.permutation(X.shape[0])
    xv, *_ = np.meshgrid(
        np.arange(0, X.shape[1]),
        np.arange(0, X.shape[0])
    )
    X_perm = X.T[xv.flatten(), np.array(tuple(rnd_idx.T)).flatten()].reshape(
        100, -1)

    # Manually compute observed relative frequencies:
    data_observed[0, :], *_ = \
        np.histogram(
            X_perm[np.where(Y == 0)[0], :].flatten(),
            np.append(D, D[-1] + 1)
        )
    data_observed[1, :], *_ = \
        np.histogram(
            X_perm[np.where(Y == 1)[0], :].flatten(),
            np.append(D, D[-1] + 1)
        )
    # Keep the chi square vals:
    t_values[iter, :] = get_test_statistic_per_feature(
        do=data_observed,
        de=data_expected
    )


fig = plt.figure()
plot_axes = fig.add_subplot(111)
for feature in range(data_observed.shape[1]):
    histo, histo_bins = np.histogram(t_values[:, feature])
    histo = histo / iterations
    plot_axes.plot(histo_bins[:-1], histo, color='gray', linewidth=0.5)
    plot_axes.axvline(x=t_vals[0, feature], color='gray', linewidth=0.5)

plot_axes.axvline(x=t_vals[0, 11], color='C2', linewidth=1.5)
plot_axes.axvline(x=t_vals[0, 12], color='C3', linewidth=1.5)
plot_axes.plot(
    np.arange(0, 22, 22/100),
    chi_pdf(np.linspace(0,22,100)),
    color='C0', linewidth=2.0
)

plot_axes.set_xticks(histo_bins[:-1:2])
plot_axes.set_xticklabels(np.around(histo_bins[:-1:2], decimals=2))
plot_axes.set_xlabel('Chi-square test statistic')
plot_axes.set_ylabel(f'Relative frequency')
plot_axes.set_title('Distribution of Chi square test statistic, '
                    '100 permutations')
plt.savefig(f"Permutation Test")

# Bonus: Calculate pvals from permutation testing:
tmp = np.less(
    t_vals_obs,
    t_values
)
p_vals_perm_test = tmp.sum(axis=0) / n_samples

fig = plt.figure()
plot_axes = fig.add_subplot(111)
plot_axes.plot(np.arange(p_values.size), np.absolute(p_values - p_vals_perm_test).T,
               color='gray', linewidth=0.5)

plot_axes.set_xticks(histo_bins[:-1:2])
plot_axes.set_xticklabels(np.around(histo_bins[:-1:2], decimals=2))
plot_axes.set_xlabel('feature p value')
plot_axes.set_ylabel(f'P val estimation error')
plot_axes.set_title('P value permutation test estimation')
plt.savefig(f"Permutation Test p val estimation.png")



print("Pronto!")
