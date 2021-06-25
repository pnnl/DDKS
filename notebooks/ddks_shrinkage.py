#!/usr/bin/env python
# coding: utf-8

# In[1]:


from ddks.data import *
from ddks.data.openimages_dataset import LS
from ddks import methods
import numpy as np
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
import tqdm
plt.style.use('ah')
plt.show()


# In[4]:


methods_list = [methods.ddks_method,  methods.onedks_method, methods.hotelling_method, methods.kldiv_method]
datasets = [GVM, GVS, Skew, MM]
sample_size = 50

def within(x, y, eps=1.0E-3):
    return np.abs(x - y) < eps

def return_mean_significance(_method, Dataset, dimension, parameter, permutations=100):
    if dataset.name == 'GVM':
        kwargs = dict(mean_p=0.5+parameter, mean_t=0.5)
    elif dataset.name == 'GVS':
        kwargs = dict(std_p=0.5+parameter, std_t=0.5)
    elif dataset.name == 'Skew':
        kwargs = dict(lambda_p=0.5+parameter, lambda_t=0.5)
    elif dataset.name == 'MM':
        kwargs = dict(noise_fraction=1.0 - parameter)
        
    _dataset = Dataset(dimension=dimension, sample_size=sample_size, **kwargs)
    p, t = next(_dataset)
    return _method(p, t, permutations)

def bisection(_method, Dataset, dimension, permutations=100, trials=10):
    mids = []
    for i in tqdm.tqdm(np.arange(trials)):
        low = 0.0
        high = 5.0
        mid = (low + high) / 2
        low_sig = return_mean_significance(_method, Dataset, dimension, low, permutations)
        high_sig = return_mean_significance(_method, Dataset, dimension, high, permutations)
        mid_sig = return_mean_significance(_method, Dataset, dimension, mid, permutations)
        significance = 1.0
        while not within(mid_sig, 0.05):
            if low_sig > 0.05 and mid_sig < 0.05:
                new_mid = (low + mid) / 2
                high = mid
                high_sig = mid_sig
                mid = new_mid
                mid_sig = return_mean_significance(_method, Dataset, dimension, mid, permutations)
            elif mid_sig > 0.05 and high_sig < 0.05:
                new_mid = (mid + high) / 2
                low = mid
                low_sig = mid_sig
                mid = new_mid
                mid_sig = return_mean_significance(_method, Dataset, dimension, mid, permutations)
            else:
                #print(low_sig, mid_sig, high_sig)
                break
        mids.append(mid)
    

    return np.nanmean(mids), np.nanstd(mids), np.sum(np.isfinite(mids))
        


# In[ ]:


data = {}
for dataset in datasets:
    data[str(dataset)] = dict()
    for metric in methods_list:
        data[str(dataset)][metric.name] = bisection(metric, dataset, dimension=3)
        print(str(dataset), metric.name, data[str(dataset)][metric.name])


# In[ ]:


fig = plt.figure(figsize=(4, 4))
markers = ['o', 'x', 'v', '^']
#gvm
for metric, marker in zip(data[list(data.keys())[0]].keys(), markers):
    means = [data[_dataset][metric][0] for _dataset in data.keys()]
    stds = [data[_dataset][metric][1] for _dataset in data.keys()]
    plt.errorbar(np.arange(len(means)), means, yerr=stds, linestyle='none', marker=marker, label=metric, alpha=0.5)
plt.xticks(np.arange(len(datasets)), ['GVM', 'GVS', 'DVU', 'Skew', 'MM', 'LS'], rotation=45)
plt.ylabel('Number of samples' + "\n" + r'for $\alpha=0.05$')
plt.legend()
plt.show()

fig = plt.figure(figsize=(8, 4))
markers = ['o', 'x', 'v', '^']
#gvm
for i, (metric, marker) in enumerate(zip(data[list(data.keys())[0]].keys(), markers)):
    means = [data[_dataset][metric][0] for _dataset in data.keys()]
    stds = [data[_dataset][metric][1] for _dataset in data.keys()]
    plt.bar(np.arange(len(means)) + (i/5.0 - 0.5), means, label=metric, width=0.20, alpha=0.5)
    plt.errorbar(np.arange(len(means)) + (i/5.0 - 0.5), means, yerr=stds, linestyle='none', marker=marker)
plt.xticks(np.arange(len(datasets)) - 0.20, ['GVM', 'GVS', 'DVU', 'Skew', 'MM', 'LS'], rotation=45)
plt.ylabel('Number of samples' + "\n" + r'for $\alpha=0.05$')
plt.legend()
plt.show()


# In[ ]:


fig = plt.figure(figsize=(8, 8))
markers = ['o', 'x', 'v', '^']
#gvm
colors = ['#D77600', '#616265', '#A63F1E', '#F4AA00', '#007836', '#00338E', '#870150']
for i, (metric, marker, color) in enumerate(zip(data[list(data.keys())[0]].keys(), markers, colors)):
    plt.subplot(221+i)
    means = [data[_dataset][metric][0] for _dataset in data.keys()]
    stds = [data[_dataset][metric][1] for _dataset in data.keys()]
    plt.errorbar(np.arange(len(means)), means, yerr=stds, linestyle='none', color=color, marker=marker, label=metric, alpha=0.5)
    plt.xticks(np.arange(len(datasets)), ['GVM', 'GVS', 'DVU', 'Skew', 'MM', 'LS'], rotation=45)
    plt.ylabel('Number of samples' + "\n" + r'for $\alpha=0.05$')
    plt.ylim(0, 250)
    plt.legend()
plt.show()


# In[ ]:


markers = ['/', '\\', '|', '.']
theta = radar_factory(6, frame='polygon')
fig, axs = plt.subplots(figsize=(4, 4), nrows=1, ncols=1,
                        subplot_kw=dict(projection='radar'))
rns = np.geomspace(1, 200, 6).astype(int)
rgrids = np.log10(rns)
plt.gca().set_rgrids(rgrids, labels=rns)
colors = ['#D77600', '#616265', '#A63F1E', '#F4AA00', '#007836', '#00338E', '#870150']
for metric, marker, color in zip(data[list(data.keys())[0]].keys(), markers, colors):
    means = np.log10([data[_dataset][metric][0] for _dataset in data.keys()])
    stds = [data[_dataset][metric][1] for _dataset in data.keys()]
    line = plt.plot(theta, means, label=metric, color=color, linestyle='-')
    plt.fill(theta, means, alpha=0.25, facecolor=color)

plt.gca().set_varlabels(['GVM', 'GVS', 'DVU', 'Skew', 'MM', 'LS'])
plt.legend(loc=(0.9, .95))
plt.show()


# In[ ]:




