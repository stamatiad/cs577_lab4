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

#REVISION ERROR: Numerical error here too:
def chi_pdf(t, df=1):
    return(t**((df-2)/2) * np.exp(-t/2)) / \
          (2**(df/2) * gamma(df/2))

# REVISION: just a function name change for clearer code:
def get_pval(do=None, de=None):
    # REVISION: since I no longer process in bach, I have removed the for loop:
    # Calculate test statistic:
    T = test_statistic(
        do=do,
        de=de,
    )
    # REVISION ERROR: This is actually a mistake of mine, that I did notice
    # before
    # submission (but had no time to investigate): the degrees of freedom
    # should have been more than one. I correct it now.
    df = do.shape[1]
    p_value = chi_pdf(T, df)
    return p_value

def get_test_statistic(do=None, de=None):
    # Calculate test statistic:
    t_value = test_statistic(
        do=do,
        de=de
    )
    return t_value

X, Y = load_dataset()
n_samples = X.shape[0]

# Compute pvalues per feature:
p_values = np.zeros((1, X.shape[1]))
observed_t_vals = np.zeros((1, X.shape[1]))
for feature in range(X.shape[1]):
    D = np.unique(X[:, feature])
    data_observed = np.zeros((2, D.size), dtype=int)
    # Manually compute observed relative frequencies:
    # REVISION ERROR: I failed to see that histogram does NOT return the
    # different histogram PER COLUMN. As a result I built the resulting code
    # on the notion that each data_observed column is a different feature
    # (whereas is a bin of the aggregated dataset/junk!). That is why I report
    # FEATURE 12 and 13. To correct that i run a normal loop for each feature
    # and change the functions/arguments:
    data_observed[0, :],  *_ = np.histogram(X[np.where(Y == 0)[0],
                                              feature].flatten(),
                               np.append(D, D[-1]+1) )
    data_observed[1, :],  *_ = np.histogram(X[np.where(Y == 1)[0],
                                              feature].flatten(),
                                            np.append(D, D[-1]+1) )


    data_probs = data_observed / n_samples
    x_marginals = data_probs.sum(axis=0)
    y_marginals = data_probs.sum(axis=1)

    data_expected = np.zeros((2, D.size))
    for j in range(D.size):
        for i in range(2):
            data_expected[i, j] = x_marginals[j] * y_marginals[i] * n_samples

    # Assignment 3a: Use the chi squared test
    # REVISION: I need to change this function name (cleaner code):
    p_value = get_pval(
        do=data_observed,
        de=data_expected
    )
    p_values[0, feature] = p_value

    t_value = get_test_statistic(
        do=data_observed,
        de=data_expected
    )
    observed_t_vals[0, feature] = t_value

# Assignment 3b: Permutation testing
iterations = 100
t_values = np.zeros((iterations, D.size))
for iter in range(iterations):
    # Randomly permute X columns:
    # Numpy nuances....
    rnd_idx = np.zeros(X.shape, dtype=int)
    for i in range(X.shape[1]):
        rnd_idx[:, i] = np.random.permutation(X.shape[0])
    #REVISION ERROR: I had an indexing error here that messed with the resluts (
    # mixing the features) god damn it python, MATLAB is so straightforward
    # at this..:
    _, yv = np.meshgrid(
        np.arange(0, X.shape[0]),
        np.arange(0, X.shape[1])
    )
    X_perm = X.T[yv.flatten(), np.array(tuple(rnd_idx.T)).flatten()].reshape(
        10, -1).T

    # Manually compute observed relative frequencies:
    # REVISION ERROR: Same histogram error here, as above. Corrected...
    for feature in range(X.shape[1]):
        D = np.unique(X_perm[:, feature])
        data_observed = np.zeros((2, D.size), dtype=int)
        data_observed[0, :], *_ = \
            np.histogram(
                X_perm[np.where(Y == 0)[0], feature].flatten(),
                np.append(D, D[-1] + 1)
            )
        data_observed[1, :], *_ = \
            np.histogram(
                X_perm[np.where(Y == 1)[0], feature].flatten(),
                np.append(D, D[-1] + 1)
            )

        data_probs = data_observed / n_samples
        x_marginals = data_probs.sum(axis=0)
        y_marginals = data_probs.sum(axis=1)

        data_expected = np.zeros((2, D.size))
        for j in range(D.size):
            for i in range(2):
                data_expected[i, j] = x_marginals[j] * y_marginals[
                    i] * n_samples
        # Keep the chi square vals:
        blah = get_test_statistic(
            do=data_observed,
            de=data_expected
        )
        t_values[iter, feature] = blah
        print('blah')


# REVISION: Plot each histogram in its own figure:
for feature in range(X.shape[1]):
    fig = plt.figure()
    plot_axes = fig.add_subplot(111)
    histo, histo_bins = np.histogram(t_values[:, feature])
    histo = histo / iterations
    plot_axes.plot(histo_bins[:-1], histo, color='gray', linewidth=0.5)
    plot_axes.axvline(x=observed_t_vals[0, feature], color='gray',
                      linewidth=0.5)
    plot_axes.plot(
        np.arange(0, 22, 22/100),
        chi_pdf(
            np.linspace(0, 22, 100),
            df=np.unique(X[:, feature]).size - 1
        ),
        color='black', linewidth=2.0
    )


'''
plot_axes.set_xticks(histo_bins[:-1:2])
plot_axes.set_xticklabels(np.around(histo_bins[:-1:2], decimals=2))
plot_axes.set_xlabel('Chi-square test statistic')
plot_axes.set_ylabel(f'Relative frequency')
plot_axes.set_title('Distribution of Chi square test statistic, '
                    '100 permutations')
plt.savefig(f"Permutation Test")
'''

# Bonus: Calculate pvals from permutation testing:
tmp = np.less(
    t_values,
    observed_t_vals
)
p_vals_perm_test = tmp.sum(axis=0) / n_samples

fig = plt.figure()
plot_axes = fig.add_subplot(111)
plot_axes.plot(np.arange(p_values.size), np.absolute(p_values - p_vals_perm_test).T,
               color='gray', linewidth=0.5)

plot_axes.set_xlabel('feature p value')
plot_axes.set_ylabel(f'P val estimation error')
plot_axes.set_title('P value permutation test estimation')
plt.savefig(f"Permutation Test p val estimation.png")



print("Pronto!")
