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
    std2 = 1.0
    data_gen = set_dgen(m1,std1)
    data_gen2 = set_dgen(m2,std2)

    p10 = m.pdKS(plane_per_dim=10)
    p50 = m.pdKS(plane_per_dim=50)
    p100 = m.pdKS(plane_per_dim=100)
    p250 = m.pdKS(plane_per_dim=250)
    p500 = m.pdKS(plane_per_dim=500)
    p1000 = m.pdKS(plane_per_dim=1000)
    p5000 = m.pdKS(plane_per_dim=5000)
    p10000 = m.pdKS(plane_per_dim=10000)
    vals = t.run_mp([ddks,p10, p50, p100, p250, p500, p5000,p10000], data_gen, d=3, data_gen2=data_gen2, nper=10, nmin=1E3,
                    nmax=1E3,
                    nsteps=1, name_list=['ddKS','p10', 'p50', 'p100', 'p250', 'p500', 'p1000','p5000','p10000'])

    print(vals)
    vals.to_pickle(f'./QT3.pkl')
    #vals = t.run_mp([vdks,rdks],data_gen,d=3,data_gen2=data_gen2,nper=10,nmax=1e4,calc_P=True)
    #vals.to_pickle(f'./Perm2_{d}d_rks_N{m1}{std1}_N{m2}{std2}.pkl')
