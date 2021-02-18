import sys
sys.path.append('/Users/jack755/PycharmProjects/ddks/')
import ddks.data as data
import ddks.methods as m
import ddks.tests as t
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

def set_dgenNoise(mean,std,noise_percent,noise_function = None):
    if noise_function == None:
        noise_function = torch.rand
    def dgen(n,d):
        num_noise = int(noise_percent * n)
        goodvals = torch.normal(mean,std,(n-num_noise,d))
        noisevals =  noise_function(num_noise,d)
        return torch.cat([goodvals,noisevals])
    return dgen

def dgen_norm(n,d):
    return torch.rand(n,d)
def set_dgen_poisson(scale):
    def dgen(n,d):
        return torch.poisson(torch.ones(n,d)*scale)
def bgcone_wrap(n,d):
    func = data.Cone(15)
    stuff = func(n)
    filtered_data = stuff[~torch.any(stuff.isnan(), dim=1)]
    return filtered_data
def cone_wrap(n,d):
    stuff = data.make_true(N=n)
    filtered_data = stuff[~torch.any(stuff.isnan(), dim=1)]
    return filtered_data
def samp1(n,d):
    data=[]
    for i in range(n):
        rnd=np.random.uniform(0,1)
        s = torch.tensor([rnd for z in range(d)])
        data.append(s)
    return torch.stack(data)
def samp2(n,d):
    return torch.tensor(np.random.uniform(0,1,size=(n,d)))

def poop_gen(n,d):
    return torch.zeros


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
    #vals = t.run_mp([vdks, rdks, ddks],data_gen,d=3,data_gen2=data_gen2,nper=10,nmax=3e3,calc_P=True)
    #vals.to_pickle(f'../runs/p{d}d_vrdks_N{m1}{std1}_N{m2}{std2}.pkl')

    #Dimensionality scaling for all
    #nval = 1000
    #dvals = t.run_mpDims([vdks,rdks,ddks], data_gen, [2,3,4,5,6,7], n=nval)
    #dvals.to_pickle(f'./pnd_vrdks_N{m1}{std1}_N{nval}.pkl')



    #pdks
    #
    # dvals = t.run_mpDims([pdks], data_gen, [3, 10, 100, 200, 500, 800, 1000], n=100)
    # dvals.to_pickle(f'./nd_pks_N{m1}{std1}_N{m2}{std2}.pkl')
    #p5 = m.pdKS(plane_per_dim=5)
    #p10 = m.pdKS(plane_per_dim=10)
    #p25 = m.pdKS(plane_per_dim=25)
    #p50 = m.pdKS(plane_per_dim=50)
    #p100 = m.pdKS(plane_per_dim=100)
    #p250 = m.pdKS(plane_per_dim=250)
    #p500 = m.pdKS(plane_per_dim=500)
    #p1000 = m.pdKS(plane_per_dim=1000)
    #vals = t.run_mp([p5,p10,p25,p50,p100,p250,p500,p1000], data_gen, d=3, data_gen2=data_gen2, nper=10, nmin=1E2,nmax=1E2,
    #                nsteps=1,name_list=['p5','p10','p25','p50','p100','p250','p500','p1000'])
    #vals.to_pickle(f'./PVAR_{d}d_pdks_N{m1}{std1}_N{m2}{std2}.pkl')

    #Cone vs cone w/ background
    #Cone vs cone
    #vals = t.run_mp([vdks, rdks, ddks], cone_wrap, d=3, data_gen2=cone_wrap, nper=10, nmin=1e3,nmax=int(1e3),nsteps=1,calc_P=True)
    #vals.to_pickle(f'./p{d}d_vrdks_cone_cone.pkl')

    #vals = t.run_mp([vdks, rdks, ddks], bgcone_wrap, d=3, data_gen2=bgcone_wrap, nper=10, nmin=1e3,nmax=1e3, nsteps=1,calc_P=True)
    #vals.to_pickle(f'./p{d}d_vrdks_bgcone_bgcone.pkl')





    #vals = t.run_mp([vdks, rdks, ddks], cone_wrap, d=3, data_gen2=bgcone_wrap, nmin=1e4, nper=10, nmax=1e4, nsteps=1)
    #vals.to_pickle(f'./{d}d_vrdks_cone_bgcone.pkl')

    #Uniform vs diagonal
    #d=3
    #vals = t.run_mp([vdks, rdks, ddks],samp1,d=3,data_gen2=samp2,nper=10,nmax=3e3,calc_P=True)
    #vals.to_pickle(f'../runs/p{d}d_vrdks_uni_diag.pkl')
    #Gaussian vs Gaussian with various amounts of background noise
    #m1 = 0.5
    #std1 = 0.1
    #n_p = 0.5
    #m2  = 0.3
    #std2 = 0.1
    #for n_p in [0.0, 0.1, 0.25, 0.5, 0.75]:
    #    vals = t.run_mp([vdks, rdks, ddks], set_dgen(m1,std1), d=3, data_gen2=set_dgenNoise(m2,std2,n_p), nper=10, nmax=3e3,calc_P=True)
    #    vals.to_pickle(f'../runs/p3d_vrdks_noise_diffdist_{n_p}.pkl')
    #    vals = t.run_mp([vdks, rdks, ddks], set_dgen(m1, std1), d=3, data_gen2=set_dgenNoise(m1, std1, n_p), nper=10,
    #                    nmax=3e3, calc_P=True)
    #    vals.to_pickle(f'../runs/p3d_vrdks_noise_{n_p}.pkl')


    #Gaussians of increasing distance
    means = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8]
    std = 0.1
    dgen1 = [set_dgen(means[0],std) for m in means]
    dgen2 = [set_dgen(m, std) for m in means]
    d1_name = [f'N({means[0]},{std})' for m in means]
    d2_name = [f'N({m},{std})' for m in means]

    print('here')
    vals = t.run_mpGen([vdks, rdks, ddks], dgen1, dgen2, d1_name, d2_name, d=3, nper=10, name_list=None, nmin=10, nmax=1E3,
              nsteps=10, calc_P=True)
    vals.to_pickle(f'../runs/p3d_vrdks_changeGauss.pkl')
    print('here')
    #vals.to_pickle(f'./{d}d_vrdks_cone_bgcone.pkl')
    ## Poisson if we want
    #p1 = set_dgen_poisson(10)
    #p2 = set_dgen_poisson(20)
    #vals = t.run_mp([pdks, rdks, ddks], p1, d=3, data_gen2=p2, nper=10, nmax=1e4, nsteps=1)
    #vals.to_pickle(f'./{d}d_prdks_pois10_pois20.pkl')
    #vals = t.run_mp([rdks, vdks, ddks], data_gen, d=3, data_gen2=data_gen2, nper=10, nmax=1e4)
    #vals.to_pickle(f'./{d}d_rvdks_N{m1}{std1}_N{m2}{std2}.pkl')

    #vals = t.run_mp([vdks,rdks],data_gen,d=3,data_gen2=data_gen2,nper=10,nmax=1e4,calc_P=True)
    #vals.to_pickle(f'./Perm2_{d}d_rks_N{m1}{std1}_N{m2}{std2}.pkl')
