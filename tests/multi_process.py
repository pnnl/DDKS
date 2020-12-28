import time
import torch
import multiprocessing
import numpy as np
import tqdm
import sys
sys.path.append('/Users/jack755/PycharmProjects/')
import ddks
import ddks.methods as m
import matplotlib.pyplot as plt


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

def data_gen(n, d):
    '''
    Method for generating data consider using methods from ddks.data
    :param n: Number of entries
    :param d: dimension of data
    :return:
    '''
    return torch.rand((n, d))

if __name__ == '__main__':
    '''
    Base code for timing methods of ddks over many orders of magnitude using multiprocessing for speedup
    To use: 1.) set xdks = rdks/vdks/ddks (Warning: ddks scales as N^2) 
    '''
    # Setup ddks methods
    ddks = m.ddKS()
    vdks = m.vdKS()
    rdks = m.rdKS()
    ## In this case, using vdks:
    xdks = rdks
    #Setup processors
    p = multiprocessing.Pool(10)
    #Setup data method
    store = []
    mns = []
    std = []
    ns = []
    vals = []
    for n in np.geomspace(1.0, 1.0E6, 10):
        store = []
        ress = []
        tic = time.time()
        n = int(n)
        print(f'Running n={n}')
        for i in range(10):
            pred = data_gen(n,3)
            true = data_gen(n,3)
            res = p.apply_async(F, args=(pred, true, xdks))
            ress.append(res)
        for res in tqdm.tqdm(ress):
            store.append(res.get())
        store = np.asarray(store)
        vals.append([n,store[:,0]])
        print(store)
        ns.append(n)
        mns.append(np.mean(store[:,1]))
        std.append(np.std(store[:,1]))
        print(f'n:{n} mns:{mns[-1]} std:{std[-1]}')
    print(mns)
    print(std)
    np.save(f'./test_saves/{xdks.__class__.__name__}ns', np.asarray(ns))
    np.save(f'./test_saves/{xdks.__class__.__name__}mns', np.asarray(mns))
    np.save(f'./test_saves/{xdks.__class__.__name__}std',np.asarray(std))
    fig1, ax1 = plt.subplots()
    ax1.errorbar(np.asarray(ns), mns, std, label='Voxel ndKS')
    ax1.set_xlabel('Number of Points')
    ax1.set_ylabel('Runtime (s)')
    ax1.legend()
    ax1.set_title(f'{xdks.__class__.__name__} Runtime vs points')
    fig1.tight_layout()
    plt.savefig(f'{xdks.__class__.__name__}TimeVPoints.png',bbox_inches='tight')
    ovals = []
    for set in vals:
        for v in set[1]:
            pt = [set[0],v]
            ovals.append(pt)

    fig2, ax2 = plt.subplots()
    ovals = np.asarray(ovals)
    np.save(f'./test_saves/{xdks.__class__.__name__}ovals',np.asarray(ovals))
    ax2.scatter(ovals[:,0],ovals[:,1])
    ax2.set_xlabel('Number of Points')
    ax2.set_ylabel('D')
    fig2.tight_layout()
    plt.savefig(f'./test_saves/{xdks.__class__.__name__}DVPoints.png', bbox_inches='tight')