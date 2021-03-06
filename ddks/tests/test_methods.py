import sys
sys.path.append('/Users/jack755/PycharmProjects/ddks')
import ddks
import ddks.methods as m
import torch
import time

#Generate sample dataset
N = 1
Nper = 1
dim = 3
input_bounds = torch.tensor([[0,1],[0,1],[0,1], [0,1],[0,1],[0,1]]).T
output_bounds = torch.tensor([[0,1],[0,1],[0,1],[0,1]]).T

data_gen = ddks.data.SmallDataSet(N, Nper, dim, input_bounds, output_bounds,addvar=False)
dset = data_gen.generate_data()
split = N//2
pred = dset[:split,-3:]
true = dset[split:,-3:]


def quick_test(pred, true, xdks):
    '''
    Quick sanity test to verify code runs.  Times run + reports value of D
    :param pred: data set
    :param true: data set
    :param xdks: ddks style class with inputs pred,true and outputs D
    :return:
    '''
    tic = time.time()
    D = xdks(pred,true)
    toc = time.time()
    name = xdks.__class__.__name__
    runtime = toc-tic
    print(f'{name}:{D}, time: {runtime}')
    return D, runtime

def data_gen(n, d):
    '''
    Method for generating data consider using methods from ddks.data
    :param n: Number of entries
    :param d: dimension of data
    :return:
    '''
    return torch.rand((n, d))


ddks = m.ddKS()
vdks = m.vdKS(vox_per_dim=3)
rdks = m.rdKS()

d=7
N = 1000
pred = data_gen(N,d)
true = data_gen(N,d)

quick_test(pred,true,ddks)
quick_test(pred,true,vdks)
quick_test(pred,true,rdks)



