# Stamatiadis Stefanos, AM2843

import pandas as pd
import numpy as np
import matplotlib.pyplot
from utils import load_dataset

def test_statistic(op=None, ep=None):
    OP = op.flatten()
    EP = ep.flatten()
    return np.divide(np.power(OP - EP, 2), EP).sum()

X, Y = load_dataset()
N = Y.size

# Compute probabilities:
#data = np.concatenate((X, Y), axis=1)

n_samples = X.size
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

expected_probs = np.zeros((2, D.size))
for i in range(D.size):
    for j in range(2):
        expected_probs[j, i] = x_marginals[i] * y_marginals[j] * n_samples

# Calculate test statistic:
T = test_statistic(
    op=data_observed,
    ep=expected_probs
)

print("Pronto!")
