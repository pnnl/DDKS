import sys
sys.path.append('/Users/jack755/PycharmProjects/ddks/')
import ddks
import ddks.methods as m
import ddks.tests as t
import ddks.data as d
import torch
import numpy as np
import matplotlib.pyplot as plt
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
def print_vals(vals):
    z=np.zeros(3)
    for i,x in enumerate(vals['D']):
        print(np.mean(np.asarray(x)))
        z[i] = np.mean(np.asarray(x))
    return(z)

def samp1(n,d):
    data=[]
    for i in range(n):
        rnd=np.random.uniform(0,1)
        s = torch.tensor([rnd for z in range(d)])
        data.append(s)
    return torch.stack(data)
def samp2(n,d):
    return torch.tensor(np.random.uniform(0,1,size=(n,d)))

if __name__=='__main__':
    rdks = m.rdKS()
    vdks = m.vdKS(vox_per_dim=2)
    vdks2 = m.vdKS(approx=False,vox_per_dim=1)
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
    data_gen = samp1
    data_gen2 = samp2
    name_list = ['ddKS', 'vdks', 'vdks_NA']
    #name_list = [ 'vdks', 'vdks_NA']
    #name_list = ['vdks_NA']
    vals = t.run_mp([ddks,vdks,vdks2], data_gen, d=3, data_gen2=data_gen2, nper=1, nmin=1E3,
                    nmax=1E3,
                    nsteps=1, name_list=name_list)
    print(vals)
    #p100 = m.pdKS(plane_per_dim=100)
    #p250 = m.pdKS(plane_per_dim=250)
    #p500 = m.pdKS(plane_per_dim=500)
    #p1000 = m.pdKS(plane_per_dim=1000)
    #p5000 = m.pdKS(plane_per_dim=5000)
    #p10000 = m.pdKS(plane_per_dim=10000)
    name_list = ['ddKS', 'p10', 'p50']
    num_stds = 50
    out_list = np.zeros((num_stds+1,3))
    '''
    std_list = 1.0+np.linspace(0.0,num_stds,num_stds+1)*0.2
    mn_list = np.linspace(0.0,num_stds,num_stds+1)*0.05
    for i,stds in enumerate(mn_list):
        m2=stds
        data_gen = set_dgen(m1, std1)
        data_gen2 = set_dgen(m2, std2)
        vals = t.run_mp([ddks,p10,p50], data_gen, d=3, data_gen2=data_gen2, nper=10, nmin=1E3,
                    nmax=1E3,
                    nsteps=1, name_list=name_list)

        out_list[i,:] = print_vals(vals)
        vals.to_pickle(f'./QT4.pkl')
    fig, ax = plt.subplots()
    ax.plot(mn_list, out_list[:, 0], label=name_list[0])
    ax.plot(mn_list, out_list[:, 1], label=name_list[1])
    ax.plot(mn_list, out_list[:, 2], label=name_list[2])
    ax.set_xlabel('Diff of STD')
    ax.set_xlabel('Diff of Mean')
    ax.set_ylabel('D')
    plt.legend()
    plt.show()
    #vals = t.run_mp([vdks,rdks],data_gen,d=3,data_gen2=data_gen2,nper=10,nmax=1e4,calc_P=True)
    #vals.to_pickle(f'./Perm2_{d}d_rks_N{m1}{std1}_N{m2}{std2}.pkl')
    '''