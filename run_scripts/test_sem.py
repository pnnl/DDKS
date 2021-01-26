import sys
sys.path.append('/Users/jack755/PycharmProjects/ddks/')
import ddks.methods as m
import ddks.tests as t
import ddks.data as d
import torch
import numpy as np
import torchvision
import torchvision.datasets as datasets
import numpy as np

dir = '../runs/alexdata/'
x = np.load(dir+'all_paper__X_train.npy',allow_pickle=True)
xval = np.load(dir+'all_paper__X_val.npy',allow_pickle=True)


xval /= np.max(np.abs(xval))
print(np.max(xval))
y = np.load(dir+'all_paper__Y_train.npy',allow_pickle=True)
yval = np.load(dir+'all_paper__Y_val.npy',allow_pickle=True)

#y = [_y[-1] for _y in y]
#yval = [_y[-1] for _y in yval]

y = np.asarray([_y[-1] for _y in y[:-5]])
yval = np.asarray([_y[-1] for _y in yval[:-3]])



pdks = m.pdKS(plane_per_dim=10)

ind_list=[]
for i in range(len(np.unique(yval))):
    ind_list.append(np.where(yval==np.unique(yval)[i]))


print(pdks(torch.Tensor(xval[ind_list[0]]),torch.Tensor(xval[ind_list[1]])))
print(pdks.permute(50))
print(pdks(torch.Tensor(xval[ind_list[0]]),torch.Tensor(xval)))
print(pdks.permute(50))
test = torch.Tensor(xval[ind_list[0]])
l = len(test)
print(pdks(test[:l//2],test[l//2:]))
print(pdks.permute(50))