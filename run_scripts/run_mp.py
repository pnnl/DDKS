import sys
sys.path.append('/Users/jack755/PycharmProjects/ddks/')
import ddks
import ddks.methods as m
import ddks.tests as t
import ddks.data as d
import torch
import numpy as np

def data_gen(n, d):
    '''
    Method for generating data consider using methods from ddks.data
    :param n: Number of entries
    :param d: dimension of data
    :return:
    '''
    return torch.rand((n, d))

if __name__=='__main__':
    rdks = m.rdKS()
    vdks = m.vdKS()
    vals = t.run_mp([rdks,vdks],data_gen,nmax=1e4)

    rval =vals[vals['name'] == 'rdKS']['D'].values
    vval = vals[vals['name'] == 'vdKS']['D'].values

    rval = np.asarray([list(rval[i]) for i in range(10)])
    vval = np.asarray([list(vval[i]) for i in range(10)])
    print(vals)
    print(mns)