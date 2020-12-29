import sys
sys.path.append('/Users/jack755/PycharmProjects/')
import ddks
import ddks.methods as m
import ddks.tests as t
import ddks.data as d
import torch

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
    vals,mns,std,ns = t.run_mp([rdks,vdks],data_gen)
    print(vals)
    print(mns)