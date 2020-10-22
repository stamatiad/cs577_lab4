# Stamatiadis Stefanos, AM2843

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma

def t_statistic(x, y):
    return (x.mean() - y.mean()) / \
           (np.sqrt((x.std()**2/x.size) + (y.std()**2/y.size)))

def tdistr(t, df=1):
    return (gamma(1/2) * (1+t**2/df)**((-df+1)/2)) / \
            (np.sqrt(df*np.pi)*gamma(df/2))

# Initial estimation

X = pd.read_csv('Assignment4_Data/algorithm_accuracies.csv').values
data_min = X.min()
data_max = X.max()
histo_bins = np.linspace(data_min, data_max, 50)
data_Paolo = X[:, 0]
histo_Paolo = np.histogram(data_Paolo, histo_bins)[0]
histo_Paolo = histo_Paolo / data_Paolo.size
data_Jim = X[:, 1]
histo_Jim = np.histogram(data_Jim, histo_bins)[0]
histo_Jim = histo_Jim / data_Jim.size


fig = plt.figure()
plot_axes = fig.add_subplot(111)
plot_axes.plot(histo_bins[:-1], histo_Paolo, color='C0', label='Paolo')
plot_axes.plot(histo_bins[:-1], histo_Jim, color='C1', label='Jim')
plot_axes.axvline(x=data_Paolo.mean(), color='C0', linewidth=1.5)
plot_axes.axvline(x=data_Jim.mean(), color='C1', linewidth=1.5)
plot_axes.set_xticks(histo_bins[:-1:5])
plot_axes.set_xticklabels(np.around(histo_bins[:-1:5], decimals=2))
plot_axes.set_xlabel('Performance')
plot_axes.set_ylabel(f'Relative frequency')
plot_axes.set_title('Performance Histogram')
plot_axes.legend()
#plt.savefig(f"Initial Performance Histo.png")

# T-test first 20 values:
data_partial_Paolo = data_Paolo[:20]
data_partial_Jim = data_Jim[:20]

T = t_statistic(data_partial_Paolo, data_partial_Jim)

df = data_partial_Jim.size + data_partial_Paolo.size -2
p_val = tdistr(T, df=df)
print(f"First 20 P-value is: {p_val}.")

# T-test first 100 values:
data_partial_Paolo = data_Paolo[:100]
data_partial_Jim = data_Jim[:100]

T = t_statistic(data_partial_Paolo, data_partial_Jim)

df = data_partial_Jim.size + data_partial_Paolo.size -2
p_val = tdistr(T, df=df)
print(f"First 100 P-value is: {p_val}.")

# T-test first 500 values:
data_partial_Paolo = data_Paolo[:500]
data_partial_Jim = data_Jim[:500]

T = t_statistic(data_partial_Paolo, data_partial_Jim)

df = data_partial_Jim.size + data_partial_Paolo.size -2
p_val = tdistr(T, df=df)
print(f"First 500 P-value is: {p_val}.")

# T-test all values:

T = t_statistic(data_Paolo, data_Jim)

df = data_Jim.size + data_Paolo.size -2
p_val = tdistr(T, df=df)
print(f"All data P-value is: {p_val}.")



print("Pronto!")

