from .ddks import ddKS
from .vdks import vdKS
from .rdks import rdKS
from .pdks import pdKS
import numpy as np
import torch
import hotelling

class Permute:
    def __init__(self, score_function):
        self.score_function = score_function

    def __call__(self, p, t, j):
        T = self.score_function(p, t)
        comb = torch.cat((p, t))
        idxc = np.arange(comb.shape[0])
        len_idxc = len(idxc)
        scores = []
        for _j in range(j):
            np.random.shuffle(idxc)
            _p = comb[:len_idxc//2]
            _t = comb[len_idxc//2:]
            scores.append(self.score_function(_p, _t))
        scores = np.array(scores)
        return np.sum(scores > T) / len_idxc


class Method:
    def __init__(self, score_object=None,
                 score_function=lambda x: x,
                 significance_function=None):
        if score_object is None:
            self.score_function = score_function
            if significance_function is None:
                self.significance_function = Permute(score_function)
            else:
                self.significance_function = significance_function
        else:
            self.score_function = score_object
            self.significance_function = score_object.permute

    def __call__(self, p, t, j):
        return self.significance_function(p, t, j=j)


ddks_method = Method(ddKS())

hotelling_method = Method(score_function=hotelling.stats.hotelling_t2)

class OneDKS:
    def __init__(self):
        self.ddks = ddKS()

    def __call__(self, p, t):
        ds = []
        for d in range(p.shape[1]):
            ds.append(self.ddks(p[:, d], t[:, d]))
        return max(ds)

onedks_method = Method(score_object=OneDKS())

kldiv_method = Method(score_object=)