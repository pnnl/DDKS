import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
d=3
m1=0.0
m2=0.1
std1=1.0
std2=1.0
vals = pd.read_pickle(f'{d}d_prvdks_N{m1}{std1}_N{m2}{std2}.pkl')
#vals = pd.read_pickle('Perm_3d_rks_N0.01.0_N0.02.0.pkl')

rval = vals[vals['name'] == 'rdKS']['D'].values
vval = vals[vals['name'] == 'vdKS']['D'].values
dval = vals[vals['name'] == 'ddKS']['D'].values
pval = vals[vals['name'] == 'pdKS']['D'].values
D_rval = np.asarray([list(rval[i]) for i in range(10)])
D_vval = np.asarray([list(vval[i]) for i in range(10)])
D_dval = np.asarray([list(dval[i]) for i in range(10)])
D_pval = np.asarray([list(pval[i]) for i in range(10)])
diff_pd = D_pval-D_dval
diff_rd = D_rval-D_dval
diff_vd = D_vval-D_dval

mdiff_pd = diff_pd.mean(axis=1)
stddiff_pd=diff_pd.std(axis=1)
mdiff_rd = diff_rd.mean(axis=1)
stddiff_rd=diff_rd.std(axis=1)
mdiff_vd = diff_vd.mean(axis=1)
stddiff_vd=diff_vd.std(axis=1)
ns = vals[vals['name'] == 'rdKS']['n'].values
fig, ax = plt.subplots()
ax.errorbar(ns,mdiff_pd,stddiff_pd,label=r'$D_{pv}$')
ax.errorbar(ns,mdiff_rd,stddiff_rd,label=r'$D_{rd}$')
ax.errorbar(ns,mdiff_vd,stddiff_vd,label=r'$D_{vd}$')

#ax.scatter(ns,mdiff)
ax.set_xlabel('Number of Points')
ax.set_ylabel(r"$D_{rdks}-D_{vdks}$")
plt.legend()
ax.set_title(f'N({m1},{std1}) N({m2},{std2})')
fig.tight_layout()


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