import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def extractDT(vals,parms=['D','T'],names=[]):
    if names is not None:
        names = np.unique(vals['name'].values)
    rvals={}
    for name in names:
        for parm in parms:
            tmp_val = vals[vals['name'] == name][parm].values
            rvals[name+'_'+parm] = np.asarray([list(tmp_val[i]) for i in range(len(tmp_val))])
    return rvals

dir = '../runs/'
sdir = '../paper_figs/'
#Accuracy wrt gaussian 3d
d=3
m1=0.0
m2=0.0
std1=1.0
std2=2.0
dpvals = pd.read_pickle(dir+'nd_pks_N0.01.0_N0.02.0.pkl')
dvals  = pd.read_pickle(dir+'nd_prdks_N0.01.0_N0.02.0.pkl')
vals = pd.read_pickle(dir+f'{d}d_prdks_N{m1}{std1}_N{m2}{std2}_ppd50.pkl')
#vals = pd.read_pickle('Perm_3d_rks_N0.01.0_N0.02.0.pkl')
rvals = extractDT(vals)

def ebplot(diff_x,ns,ax,label):
    mdiff_x = diff_x.mean(axis=1)
    stddiff_x = diff_x.std(axis=1)
    ax.errorbar(ns, mdiff_x, stddiff_x, label=label)
nlist = ['pdKS','rdKS','ddKS']
clist = ['k','r','b','g']
diff_pd = rvals['pdKS_D']-rvals['ddKS_D']
diff_rd = rvals['rdKS_D']-rvals['ddKS_D']
diff_pr = rvals['pdKS_D']-rvals['rdKS_D']
ns = vals[vals['name'] == 'rdKS']['n'].values
fig, ax = plt.subplots()
ebplot(diff_pd,ns,ax,r'$D_{pd}$')
ebplot(diff_rd,ns,ax,r'$D_{rd}$')
ebplot(diff_pr,ns,ax,r'$D_{pr}$')
ax.set_xlabel('Number of Points')
ax.set_ylabel(r"$D_{ij}=D_i-D_j$")
plt.legend()
ax.set_title(f'N({m1},{std1}) N({m2},{std2})')
fig.tight_layout()
plt.savefig(sdir+'DiffvsN.png', bbox_inches='tight')





fig2, ax2 = plt.subplots()
for i in range(10):
    for j in range(len(list(rvals.values())[0])):
        for k,name in enumerate(nlist):
            ax2.scatter(ns[i], rvals[name+'_D'][i][j], color=clist[k], label=name)
            ax2.set_xlabel('Number of Points')
            ax2.set_ylabel(r"$D")
plt.legend(nlist)
fig2.tight_layout()
plt.savefig(sdir+'DvsN.png', bbox_inches='tight')

#plt.show()

#Time figures
fig3,ax3 = plt.subplots()
for name in nlist:
    ebplot(rvals[name+'_T'],ns,ax3,name)
ax3.set_xlabel('Number of points')
ax3.set_ylabel('Time (s)')
plt.legend()
fig3.tight_layout()
plt.savefig(sdir+'TvsN.png', bbox_inches='tight')


ds = [2,3,4,5,6,7]
pds = [3, 10, 100, 200, 500, 800, 1000]
rvals = extractDT(dvals)
diff_pd = rvals['pdKS_D']-rvals['ddKS_D']
diff_rd = rvals['rdKS_D']-rvals['ddKS_D']
diff_pr = rvals['pdKS_D']-rvals['rdKS_D']
ns = vals[vals['name'] == 'rdKS']['n'].values
fig1b, ax1b = plt.subplots()
ebplot(diff_pd,ds,ax1b,r'$D_{pd}$')
ebplot(diff_rd,ds,ax1b,r'$D_{rd}$')
ebplot(diff_pr,ds,ax1b,r'$D_{pr}$')
ax1b.set_xlabel('Dimensions')
ax1b.set_ylabel(r"$D_{ij}=D_i-D_j$")
plt.legend()
ax1b.set_title(f'N({m1},{std1}) N({m2},{std2})')
fig.tight_layout()

fig2b, ax2b = plt.subplots()
for i in range(len(ds)):
    for j in range(len(list(rvals.values())[0])):
        for k,name in enumerate(nlist):
            ax2b.scatter(ds[i], rvals[name+'_D'][i][j], color=clist[k], label=name)
            ax2b.set_xlabel('Dimension')
            ax2b.set_ylabel(r"$D")
plt.legend(nlist)
fig2b.tight_layout()
plt.savefig(sdir+'DvsDim.png', bbox_inches='tight')

#plt.show()

#Time figures
fig3b,ax3b = plt.subplots()
for name in nlist:
    ebplot(rvals[name+'_T'],ds,ax3b,name)
ax3b.set_xlabel('Dimension')
ax3b.set_ylabel('Time (s)')
plt.legend()
fig3b.tight_layout()
plt.savefig(sdir+'TvsDim.png', bbox_inches='tight')

rvals = extractDT(dpvals)

fig3c,ax3c = plt.subplots()
for name in ['pdKS']:
    ebplot(rvals[name+'_T'],pds,ax3c,name)
ax3c.set_xlabel('Dimension')
ax3c.set_ylabel('Time (s)')
plt.legend()
fig3c.tight_layout()
plt.savefig(sdir+'pTvsDim.png', bbox_inches='tight')





#plt.show()

tpvals = pd.read_pickle(dir+'PVAR_3d_pdks_N0.01.0_N0.02.0.pkl')
pnames = ['p5','p10','p25','p50','p100','p250','p500','p1000']
ppd = [5,10,25,50,100,250,500,1000]
pvals = extractDT(tpvals,names=pnames)
pmean = []
pstds = []
for name in pnames:
    pmean.append(pvals[name+'_T'].mean())
    pstds.append( pvals[name+'_T'].std())
fig4,ax4 = plt.subplots()
print(pmean)
print(pstds)
ax4.errorbar(ppd, pmean, pstds)
ax4.set_xlabel('Planes per Dimension')
ax4.set_ylabel('Time (s)')
plt.savefig(sdir+'TvsPPD.png', bbox_inches='tight')
plt.show()



'''

plt.savefig(f'PRDVdiff_N{m1}{std1}_N{m2}{std2}.png', bbox_inches='tight')
fig2, ax2 = plt.subplots()
for i in range(10):
    for j in range(len(rval[i])):
        ax2.scatter(ns[i], pval[i][j], color='k', label='pdks')
        ax2.scatter(ns[i], rval[i][j], color='b', label='rdks')
        ax2.scatter(ns[i], vval[i][j], color='r', label='vdks')
        ax2.scatter(ns[i], dval[i][j], color='g', label='ddks')

ax2.set_title(f'N({m1},{std1}) N({m2},{std2})')
ax2.set_xlabel('Number of Points')
ax2.set_ylabel('D')
plt.legend(['pdks','rdks','vdks','ddks'])
fig2.tight_layout()
plt.savefig(f'PRDV_N{m1}{std1}_N{m2}{std2}.png', bbox_inches='tight')




rT = vals[vals['name'] == 'rdKS']['T'].values
vT = vals[vals['name'] == 'vdKS']['T'].values
dT = vals[vals['name'] == 'ddKS']['T'].values
pT = vals[vals['name'] == 'pdKS']['T'].values

rT = np.asarray([list(rT[i]) for i in range(10)])
vT = np.asarray([list(vT[i]) for i in range(10)])
dT = np.asarray([list(dT[i]) for i in range(10)])
pT = np.asarray([list(pT[i]) for i in range(10)])
print(rT)
mTr = rT.mean(axis=1)
stdTr = rT.std(axis=1)
mTv = vT.mean(axis=1)
stdTv = vT.std(axis=1)
mTd = dT.mean(axis=1)
stdTd= dT.std(axis=1)
mTp = pT.mean(axis=1)
stdTp= pT.std(axis=1)
fig3, ax3 = plt.subplots()
ax3.errorbar(ns,mTr,stdTr,label=r'$rdKS$')
ax3.errorbar(ns,mTv,stdTv,label=r'$vdKS$')
ax3.errorbar(ns,mTd,stdTd,label=r'$ddKS$')
ax3.errorbar(ns,mTp,stdTp,label=r'$pdKS$')
ax3.set_xlabel('Number of Points')
ax3.set_ylabel(r"Runtime (s)")
plt.legend()
ax3.set_title(f'3 Dimensions')
plt.savefig(f'TIME_PRDV_N{m1}{std1}_N{m2}{std2}.png', bbox_inches='tight')
plt.show()
'''