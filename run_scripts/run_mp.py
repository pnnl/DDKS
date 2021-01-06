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
    m2 = 0.0
    std1 = 1.0
    std2 = 2.0
    data_gen = set_dgen(m1,std1)
    data_gen2 = set_dgen(m2,std2)
    d=3
    vals = t.run_mp([pdks, rdks, vdks, ddks],data_gen,d=3,data_gen2=data_gen2,nper=10,nmax=1e4)
    vals.to_pickle(f'./{d}d_prvdks_N{m1}{std1}_N{m2}{std2}.pkl')

    #vals = t.run_mp([rdks, vdks, ddks], data_gen, d=3, data_gen2=data_gen2, nper=10, nmax=1e4)
    #vals.to_pickle(f'./{d}d_rvdks_N{m1}{std1}_N{m2}{std2}.pkl')

    #dvals = t.run_mpDims([rdks,vdks,ddks], data_gen, [2,3,4,5,6,7], n=100)
    #dvals.to_pickle(f'./nd_rvdks_Nm1}{std1}_N{m2}{std2}')
    #vals = t.run_mp([vdks,rdks],data_gen,d=3,data_gen2=data_gen2,nper=10,nmax=1e4,calc_P=True)
    #vals.to_pickle(f'./Perm2_{d}d_rks_N{m1}{std1}_N{m2}{std2}.pkl')
