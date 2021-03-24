'''
Files to generate data for ddks tests

We want:
n-d gaussian
cone - or is this waiting until subsequent publications
Multimodal gaussians?
'''
import torch
from .cone import  Cone
from .smalldata import SmallDataSet
from .cone import make_true



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
