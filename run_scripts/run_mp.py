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
def dgen_norm(n,d):
    return torch.rand(n,d)
def set_dgen_poisson(scale):
    def dgen(n,d):
        return torch.poisson(torch.ones(n,d)*scale)
def bgcone_wrap(n,d):
    return ddks.data.Cone(n)
def cone_wrap(n,d):
    return ddks.data.make_true(n)


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
    ## Paper Figure 2: Time scaling + accuracy for all
    vals = t.run_mp([pdks, rdks, ddks],data_gen,d=3,data_gen2=data_gen2,nper=10,nmax=1e4)
    vals.to_pickle(f'./{d}d_prdks_N{m1}{std1}_N{m2}{std2}.pkl')

    #Dimensionality scaling for all + pdks + P scaling for pdks
    dvals = t.run_mpDims([pdks,rdks,ddks], data_gen, [2,3,4,5,6,7], n=100)
    dvals.to_pickle(f'./nd_prdks_Nm1}{std1}_N{m2}{std2}')
    dvals = t.run_mpDims([pdks], data_gen, [3, 10, 100, 200, 500, 800, 1000], n=100)
    dvals.to_pickle(f'./nd_pks_Nm1}{std1}_N{m2}{std2}')
    p5 = m.pdKS(plane_per_dim=5)
    p10 = m.pdKS(plane_per_dim=10)
    p25 = m.pdKS(plane_per_dim=25)
    p50 = m.pdKS(plane_per_dim=50)
    p100 = m.pdKS(plane_per_dim=100)
    p250 = m.pdKS(plane_per_dim=250)
    p500 = m.pdKS(plane_per_dim=500)
    p1000 = m.pdKS(plane_per_dim=1000)
    vals = t.run_mp([p5,p10,p25,p50,p100,p250,p500,p1000], data_gen, d=3, data_gen2=data_gen2, nper=10, nmin=1E2,nmax=1E2,
                    nsteps=1,name_list=['p5','p10','p25','p50','p100','p250','p500','p1000'])
    vals.to_pickle(f'./PVAR_{d}d_pdks_N{m1}{std1}_N{m2}{std2}.pkl')

    #Cone vs cone w/ background
    #Cone vs cone
    vals = t.run_mp([pdks, rdks, ddks], cone_wrap, d=3, data_gen2=cone_wrap, nper=10, nmax=1e4,nsteps=1)
    vals.to_pickle(f'./{d}d_prdks_cone_cone.pkl')

    vals = t.run_mp([pdks, rdks, ddks], bgcone_wrap, d=3, data_gen2=bgcone_wrap, nper=10, nmax=1e4, nsteps=1)
    vals.to_pickle(f'./{d}d_prdks_bgcone_bgcone.pkl')

    vals = t.run_mp([pdks, rdks, ddks], cone_wrap, d=3, data_gen2=bgcone_wrap, nper=10, nmax=1e4, nsteps=1)
    vals.to_pickle(f'./{d}d_prdks_cone_bgcone.pkl')

    ## Poisson if we want
    p1 = set_dgen_poisson(10)
    p2 = set_dgen_poisson(20)
    vals = t.run_mp([pdks, rdks, ddks], p1, d=3, data_gen2=p2, nper=10, nmax=1e4, nsteps=1)
    vals.to_pickle(f'./{d}d_prdks_pois10_pois20.pkl')
    #vals = t.run_mp([rdks, vdks, ddks], data_gen, d=3, data_gen2=data_gen2, nper=10, nmax=1e4)
    #vals.to_pickle(f'./{d}d_rvdks_N{m1}{std1}_N{m2}{std2}.pkl')

    #vals = t.run_mp([vdks,rdks],data_gen,d=3,data_gen2=data_gen2,nper=10,nmax=1e4,calc_P=True)
    #vals.to_pickle(f'./Perm2_{d}d_rks_N{m1}{std1}_N{m2}{std2}.pkl')
