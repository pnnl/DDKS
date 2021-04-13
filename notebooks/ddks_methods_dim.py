#!/usr/bin/env python
# coding: utf-8

# In[4]:


from ddks.data import *
from ddks.data.openimages_dataset import LS
from ddks import methods
import numpy as np
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
import tqdm
import pickle
import os
plt.style.use('ah')
plt.show()


# In[12]:


methods_list = [methods.ddks_method,  methods.onedks_method, methods.hotelling_method, methods.kldiv_method]
datasets = [DVU, LS]
dimensions = np.arange(1, 10) + 1

def within(x, y, eps=1.0E-3):
    return np.abs(x - y) < eps

def return_mean_significance(_method, Dataset, dimension, sample_size, permutations=100):
    _dataset = Dataset(dimension=dimension, sample_size=sample_size)
    p, t = next(_dataset)
    return _method(p, t, permutations)

def bisection(_method, Dataset, dimension, permutations=100, trials=10, max_sample_size=100):
    mids = []
    for i in tqdm.tqdm(np.arange(trials)):
        low = 2
        high = 100
        mid = int((low + high) / 2)
        low_sig = return_mean_significance(_method, Dataset, dimension, low, permutations)
        high_sig = return_mean_significance(_method, Dataset, dimension, high, permutations)
        mid_sig = return_mean_significance(_method, Dataset, dimension, mid, permutations)
        significance = 1.0
        while not within(mid_sig, 0.05):
            if (high - low) <= 2:
                if (np.abs(high_sig - 0.05) < np.abs(mid_sig - 0.05))                         and (np.abs(high_sig - 0.05) < np.abs(low_sig - 0.05)):
                    mid = high
                    mid_sig = high_sig
                elif (np.abs(low_sig - 0.05) < np.abs(mid_sig - 0.05))                         and (np.abs(low_sig - 0.05) < np.abs(mid_sig - 0.05)):
                    mid = low
                    mid_sig = low_sig
                break
            if low_sig > 0.05 and mid_sig < 0.05:
                new_mid = int((low + mid) / 2)
                high = mid
                high_sig = mid_sig
                mid = new_mid
                mid_sig = return_mean_significance(_method, Dataset, dimension, mid, permutations)
            elif mid_sig > 0.05 and high_sig < 0.05:
                new_mid = int((mid + high) / 2)
                low = mid
                low_sig = mid_sig
                mid = new_mid
                mid_sig = return_mean_significance(_method, Dataset, dimension, mid, permutations)
            else:
                if high < max_sample_size:
                    high = high + 50
                    mid = int((low + high) / 2)
                    low_sig = return_mean_significance(_method, Dataset, dimension, low, permutations)
                    high_sig = return_mean_significance(_method, Dataset, dimension, high, permutations)
                    mid_sig = return_mean_significance(_method, Dataset, dimension, mid, permutations)
                else:
                    mid = np.nan
                    break
        mids.append(mid)

    return np.nanmean(mids), np.nanstd(mids), np.sum(np.isfinite(mids))
        


# In[13]:


data = {}
for dataset in datasets:
    print(dataset)
    if os.path.exists(f'ddks_dims_{dataset.name}.pkl'):
        data = pickle.load(open(f'ddks_dims_{dataset.name}.pkl', 'rb'))
    else:
        for dimension in dimensions:
            data[str(dimension)] = dict()
            for metric in methods_list:
                data[str(dimension)][metric.name] = bisection(metric, dataset, dimension=dimension)
                print(str(dimension), metric.name, data[str(dimension)][metric.name])
        pickle.dump(data, open(f'ddks_dims_{dataset.name}.pkl', 'wb'))


# In[ ]:




