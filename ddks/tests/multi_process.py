import time
import multiprocessing
import numpy as np
import tqdm
from pandas import DataFrame
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


def run_mp(dks_list,data_gen1,d=3,data_gen2=None, nper=10,name_list = None,nmax=10E4):
    '''
    Times and runs a list of xdks methods using 10 sets of data ranging geometrically from n=10..nmax
    :param dks_list: List of xdks methods
    :param data_gen1: Method to generate pred/true dists
    :param d: dimension of system
    :param data_gen2: OPTIONAL if not provided data_gen1=data_gen2
    :param nper: OPTIONAL (default 10) Number of runs with fixed n
    :param name_list: OPTIONAL (default = xdks.name) Specify to label output data
    :param nmax: (default 10E4)  Largest dataset values generated
    :return: df_vals dataframe with D,Time values
    '''
    #if single ddks method is provided, convert to list
    if type(dks_list) != type([]):
        dks_list = [dks_list]
    #Check if two data_gen functions are provided
    if data_gen2 is None:
        data_gen2 = data_gen1
    if name_list is None:
        name_list = [xdks.__class__.__name__ for xdks in dks_list]

    #Setup multiprocessing pool
    p = multiprocessing.Pool(min([nper,multiprocessing.cpu_count()]))
    # Setup data method
    mns = []
    std = []
    ns = []
    vals = []

    for n in np.geomspace(10, nmax, 10):
        n = int(n)
        p_list = [data_gen1(n,d) for i in range(nper)]
        t_list = [data_gen2(n, d) for i in range(nper)]
        for xdks, name in zip(dks_list, name_list):
            store = []
            ress = []
            print(f'Running {name} for n={n}')
            for i in range(nper):
                pred = p_list[i]
                true = t_list[i]
                res = p.apply_async(F, args=(pred, true, xdks))
                ress.append(res)
            for res in tqdm.tqdm(ress):
                store.append(res.get())
            store = np.asarray(store)
            vals.append([name, n, store[:, 0],store[:,1]])
            df_vals = DataFrame(vals,columns=['name','n','D','T'])
            ns.append(n)
            mns.append(np.mean(store[:, 1]))
            std.append(np.std(store[:, 1]))
    return(df_vals)




    '''        
        print(mns)
        print(std)
        np.save(f'./test_saves/{xdks.__class__.__name__}ns', np.asarray(ns))
        np.save(f'./test_saves/{xdks.__class__.__name__}mns', np.asarray(mns))
        np.save(f'./test_saves/{xdks.__class__.__name__}std', np.asarray(std))
        fig1, ax1 = plt.subplots()
        ax1.errorbar(np.asarray(ns), mns, std, label='Voxel ndKS')
        ax1.set_xlabel('Number of Points')
        ax1.set_ylabel('Runtime (s)')
        ax1.legend()
        ax1.set_title(f'{xdks.__class__.__name__} Runtime vs points')
        fig1.tight_layout()
        plt.savefig(f'{xdks.__class__.__name__}TimeVPoints.png', bbox_inches='tight')
        ovals = []
        for set in vals:
            for v in set[1]:
                pt = [set[0], v]
                ovals.append(pt)

        fig2, ax2 = plt.subplots()
        ovals = np.asarray(ovals)
        np.save(f'./test_saves/{xdks.__class__.__name__}ovals', np.asarray(ovals))
        ax2.scatter(ovals[:, 0], ovals[:, 1])
        ax2.set_xlabel('Number of Points')
        ax2.set_ylabel('D')
        fig2.tight_layout()
        plt.savefig(f'./test_saves/{xdks.__class__.__name__}DVPoints.png', bbox_inches='tight')
    '''
'''
if __name__ == '__main__':

    parser = argparse.ArgumentParser
    parser.add_argument('-m', action='append',
                        dest='methods',
                        default=[],
                        help='Add repeated values to a list')
    parser.add_argument('-NP', action='store',
                        dest='numproc',
                        default=0,
                        type=int,
                        help='Number of Processors')
    results = parser.parse_args()



    # Setup ddks methods
    meth_dict = {'ddks':m.ddKS(), 'vdks':m.vdKS(),'rdks':m.rdKS()}
    funs = []
    for x in results.methods:
        if x in meth_dict:
            funs.append(meth_dict[x])
    if len(funs) == 0:
        print('No method specified! Defaulting to ddks')
        meth_dict = m.ddKS()
    #Setup processors
    if results.numproc == 0:
        p = multiprocessing.Pool(multiprocessing.cpu_count())
    else:
        p = multiprocessing.Pool(results.numproc)

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
'''