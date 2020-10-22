import numpy as np
import re

def str2float(s):
    float_result = np.nan
    if s:
        float_result = float(s)
    return float_result

def load_dataset():
    # Pandas DataFrame uses dict which is not ordered, so we go manually...
    with open(f'Assignment4_Data/Dataset4.3_X.csv', 'r') as fid:
        X_vals = np.array([
            list(map(str2float, re.split(r',|\n', line)))
            for line in fid
        ])

    with open(f'Assignment4_Data/Dataset4.3_Y.csv', 'r') as fid:
        Y_vals = np.array([
            list(map(str2float, re.split(r',|\n', line)))
            for line in fid
        ])

    # Manually remove the nan (cr) lines:
    X_vals = X_vals[:, :-1].astype(int)
    # MANUAL: classes are always int:
    Y_vals = Y_vals[:, :-1].astype(int)
    #data = np.concatenate((X_vals[:,:-1], Y_vals[:,:-1]))

    return X_vals, Y_vals
