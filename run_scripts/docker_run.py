import sys
#sys.path.append('/Users/jack755/PycharmProjects/ddks/')
import ddks
import ddks.data as data
import ddks.methods as m
import ddks.tests as t
import torch
import numpy as np
from pandas import DataFrame
import time
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


def F(true,pred,xdks):
    '''
    Running the xdks test of choice
    Helper function to use multiprocessing
    :param true: true data set
    :param pred: pred data set
    :param xdks: method of ddks (see ddks.methods)
    :return:
    '''
    tic = time.time()
    D = xdks(true, pred)
    toc = time.time()
    return [D, toc - tic]


nmin = 10
nmax = 2000
nper = 10
nsteps = 10
rdks = m.rdKS()
vdks = m.vdKS()
ddks = m.ddKS()
dks_list = [rdks,vdks,ddks]
name_list = [xdks.__class__.__name__ for xdks in dks_list]
dgen1 = set_dgen(0.0,0.1)
dgen2 = dgen1
d = 3
vals =[]
for n in np.geomspace(nmin, nmax, nsteps):
    n = int(n)
    p_list = [dgen1(n, d) for i in range(nper)]
    t_list = [dgen2(n, d) for i in range(nper)]
    for xdks, name in zip(dks_list, name_list):
        store = []
        ress = []
        print(f'Running {name} for n={n}')
        for i in range(nper):
            pred = p_list[i]
            true = t_list[i]
            tmpD, tmpT = F(true,pred,xdks)
            vals.append([name, n, d, tmpD,tmpT])
                # df_vals = DataFrame(vals, columns=['name','dg1','dg2','n','d', 'D', 'T','p'])
df_vals = DataFrame(vals, columns=['name', 'n', 'd', 'D', 'T'])
df_vals.to_pickle(f'./docker_timetest_3d.pkl')
