#!/usr/bin/env python
import argparse
import pickle
import os
import time
import itertools
from ddks.data import GVM, GVS, Skew, DVU, MM
from ddks.data.openimages_dataset import LS
from ddks import methods
import numpy as np
import tqdm

methods_list = [methods.ddks_method,  methods.onedks_method, methods.hotelling_method, methods.kldiv_method]
methods_dict = dict(ddKS=methods.ddks_method, OnedKS=methods.onedks_method, HotellingT2=methods.hotelling_method,
                    KLDiv=methods.kldiv_method)
datasets = [GVM, GVS, Skew, DVU, LS, MM]
datasets_dict = dict(GVM=GVM, GVS=GVS, Skew=Skew, DVU=DVU, LS=LS, MM=MM)
dimensions = np.arange(1, 10) + 1

def within(x, y, eps=1.0E-3):
    return np.abs(x - y) < eps

def return_mean_significance(_method, Dataset, dimension, sample_size, permutations=100):
    _dataset = Dataset(dimension=dimension, sample_size=sample_size)
    p, t = next(_dataset)
    if _dataset.name == 'ddKS':
        p = p.to(torch.device('cuda:0'))
        t = t.to(torch.device('cuda:0'))
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
        print(f'{i},{_method.name},{Dataset.name},{dimension},{mid}')
    return np.nanmean(mids), np.nanstd(mids), np.sum(np.isfinite(mids))
        


if __name__ == "__main__":
    all_combs = np.array(np.meshgrid([method.name for method in methods_list],
                                     [_dataset.name for _dataset in datasets], dimensions)).T.reshape(-1, 3)
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, help='Index to the array built of all methods, dimensions, and types')
    args = parser.parse_args()
    n = args.n
    if n in np.arange(len(all_combs)):
        comb = all_combs[n]
        method = methods_dict[comb[0]]
        dataset = datasets_dict[comb[1]]
        dimension = int(comb[2])
        bisection(method, dataset, dimension)





