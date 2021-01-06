import sys
sys.path.append('/Users/jack755/PycharmProjects/ddks/')
import ddks
import ddks.methods as m
import ddks.tests as t
import ddks.data as d
import torch
import numpy as np

def set_dgen(mean,std):
    def dgen(n, d):
        '''
        Method for generating data consider using methods from ddks.data
        :   param n: Number of entries
        :param d: dimension of data
        :return:
        '''
        return torch.normal(mean,std,(n, d))
    return dgen
if __name__=='__main__':
    rdks = m.rdKS()
    vdks = m.vdKS()
    ddks = m.ddKS()
    pdks = m.pdKS()
    m1 = 0.0
    m2 = 0.1
    std1 = 1.0
    std2 = 1.0
    data_gen = set_dgen(m1,std1)
    data_gen2 = set_dgen(m2,std2)

    dvals = t.run_mpDims([pdks], data_gen, [1000], n=100)
    print(dvals)
    dvals.to_pickle(f'./QT2.pkl')
    #vals = t.run_mp([vdks,rdks],data_gen,d=3,data_gen2=data_gen2,nper=10,nmax=1e4,calc_P=True)
    #vals.to_pickle(f'./Perm2_{d}d_rks_N{m1}{std1}_N{m2}{std2}.pkl')
