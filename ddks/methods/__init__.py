from .ddks import ddKS
from .vdks import vdKS
from .rdks import rdKS
from .pdks import pdKS
import numpy as np
import torch
from hotelling.stats import hotelling_t2

class Permute:
    def __init__(self, score_function):
        self.score_function = score_function

    def __call__(self, p, t, j):
        T_0 = self.score_function(p.double(), t.double())
        if isinstance(T_0, torch.Tensor):
            T_0 = T_0.item()
        comb = torch.cat((p, t)).double()
        idxc = np.arange(comb.shape[0])
        len_idxc = len(idxc)
        scores = []
        for _j in range(j):
            np.random.shuffle(idxc)
            _p = comb[idxc[:len_idxc//2]]
            _t = comb[idxc[len_idxc//2:]]
            scores.append(self.score_function(_p, _t))
        scores = np.array(scores)
        return (np.sum(np.less_equal(T_0, scores)) + 1.0) / (float(_j) + 2.0)


class Method:
    def __init__(self, score_object=None,
                 score_function=lambda x: x,
                 significance_function=None, name=''):
        self.name = name
        if score_object is None:
            self.score_function = score_function
            if significance_function is None:
                self.significance_function = Permute(score_function)
            else:
                self.significance_function = significance_function
        else:
            self.score_function = score_object
            self.significance_function = Permute(score_object)

    def __call__(self, p, t, j):
        return self.significance_function(p, t, j=j)


ddks_method = Method(ddKS(), name='ddKS')
vdks_method = Method(vdKS(), name='vdKS')
rdks_method = Method(rdKS(), name='rdKS')

class HotellingT2:
    def __call__(self, p, t):
        s, _, _, _ = hotelling_t2(p.double(), t.double())
        return s

hotelling_method = Method(score_object=HotellingT2(), name='HotellingT2')

class OneDKS:
    def __init__(self):
        self.ddks = ddKS()

    def __call__(self, p, t):
        ds = []
        for d in range(p.shape[1]):
            ds.append(self.ddks(p[:, d].unsqueeze(-1), t[:, d].unsqueeze(-1)))
        return max(ds)

onedks_method = Method(score_object=OneDKS(), name='OnedKS')

def kldiv_hist(pred, true):
    edges = []
    N = int(np.power(pred.shape[0], 3.0/5.0))
    pred = pred.detach().cpu()
    true = true.detach().cpu()
    for i in range(pred.shape[1]):
        _edges = np.linspace(np.nanmin([np.nanmin(pred[:, i]), np.nanmin(true[:, i])]),
                             np.nanmax([np.nanmax(pred[:, i]), np.nanmax(true[:, i])]),
                             N)
        edges.append(_edges)
    p, _ = np.histogramdd(pred, bins=edges)
    p /= pred.shape[0]
    t, _ = np.histogramdd(true, bins=edges)
    t /= true.shape[0]
    p = torch.from_numpy(p) + 1.0E-9
    t = torch.from_numpy(t) + 1.0E-9
    kld = torch.nn.KLDivLoss(reduction='sum')(torch.log(p), t)
    return torch.tensor([kld])

kldiv_method = Method(score_function=kldiv_hist, name='KLDiv')
