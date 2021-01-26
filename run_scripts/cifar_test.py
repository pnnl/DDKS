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




import torch
import torchvision
import torchvision.transforms as transforms

def im2pix(imlist):
    #Takes in 32x32 grayscale images
    return(torch.stack([torch.tensor([i/32,j/32,im[0,i,j]]) for i in range(32) for j in range(32) for im in imlist]))

if __name__ == '__main__':

    transform = transforms.Compose(
        [transforms.ToTensor(),transforms.Grayscale(),
         transforms.Normalize((0.5), (0.5))])
    n=1000
    trainset = torchvision.datasets.CIFAR10(root='../../../Documents/datasets', train=True,
                                            download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=n,
                                              shuffle=True, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')





    examples = enumerate(train_loader)
    batch_idx, (example_data, example_targets) = next(examples)
    print(example_data.shape)

    #print(im2pix(example_data).shape)

    #all_data = example_data.reshape([n,32*32])
    all_data = example_data
    val_loc = [np.asarray((example_targets == i).nonzero().squeeze()) for i in range(0,9)]

    vdks = m.vdKS(vox_per_dim=15)
    pdks = m.pdKS(plane_per_dim=10)
    z = len(val_loc[0])
    print(z)
    pred = im2pix(all_data[val_loc[0][:z//2]])
    true = im2pix(all_data[val_loc[0][z//2:]])

    print(pdks(pred,true))
    print(vdks(pred, true))
    pred = im2pix(all_data[val_loc[0]])
    true = im2pix(all_data[val_loc[1]])
    print(pdks(pred,true))
    print(vdks(pred, true))
#print(pdks(pred,true))
#print(pdks.permute(J=10))
#tmp_data = all_data.clone()
#D_list = []
#predz=torch.tensor([])


#for ind in val_loc[0][::-1]:
#    predz=torch.cat((predz,tmp_data[ind].reshape(1,28*28)))
#    print(predz.shape)
#    tmp_data = torch.cat((tmp_data[:ind],tmp_data[ind+1:]))
#    D_list.append(pdks(predz,tmp_data))
#    print(D_list[-1])
#print(D_list)