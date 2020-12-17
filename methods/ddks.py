import torch
import numpy as np


def S_(x, f):
    return np.power(-1, np.floor(4.0 * f * x))


class ddKS(object):
    def __init__(self, soft=False, T=0.1, method='all', n_test_points=10,
                 pts=None, norm=False, oneway=True):
        if soft:
            self.max = smooth_max(T=T)
            self.ge = self.softge
        else:
            self.max = torch.max
            self.ge = self.hardge
        self.method = method
        self.n_test_points = n_test_points
        self.pts = pts
        self.norm = norm
        self.oneway = oneway

    def __call__(self, pred, true):
        # Choose points from each dataset for ddks (Usually all)
        Q, U = self.getQU(pred, true)
        D = calcD(pred, true, Q, U)
        return D

    def getQU(self, pred, true):
        # Function to decide number of points to use for ddKS test
        if self.method == 'all':
            Q = pred;
            U = true
        elif self.method == 'subsample':
            idx = np.random.choice(np.arange(np.min([pred.shape[0], true.shape[0]])), size=self.n_test_points)
            Q = pred[idx, ...];
            U = true[idx, ...]
        elif self.method == 'linear':
            if self.pts is None:
                Q = torch.empty((self.n_test_points, pred.shape[1]))
                U = torch.empty((self.n_test_points, true.shape[1]))
                for dim in range(pred.shape[1]):
                    Q[:, dim] = torch.linspace(pred[:, dim].min(), pred[:, dim].max(), self.n_test_points)
                for dim in range(true.shape[1]):
                    U[:, dim] = torch.linspace(true[:, dim].min(), true[:, dim].max(), self.n_test_points)
            else:
                Q = self.pts.clone()
                U = self.pts.clone()
        return Q, U

    def calcD(self, pred, true, Q, U):
        if pred.shape[1] != 3:
            os_pp = self.get_orthants(pred, Q)
            os_pt = self.get_orthants(true, Q)
        else:
            os_pp = self.get_octants(pred, Q)
            os_pt = self.get_octants(true, Q)
        D1 = self.max((os_pp - os_pt).abs())
        if self.oneway:
            D = D1
        else:
            os_tt = self.get_octants(true, U)
            os_tp = self.get_octants(pred, U)
            D2 = self.max((os_tt - os_tp).abs())
            D = (D1 + D2) / 2.
        if self.norm:
            D = D / float(pred.shape[0])
        return D

    def softge(self, x, y):
        return (torch.tanh(10.0 * (x - y)) + 1.0) / 2.0

    def hardge(self, x, y):
        return torch.ge(x, y).long()

    def get_orthant_matrix(self, d):
        n_orthants = int(np.power(2, d))
        x = np.empty((n_orthants, d))
        for i in range(n_orthants):
            for j in range(d):
                x[i, j] = S_(i, np.power(2.0, -j - 2))
        return x

    def get_orthants(self, x, points):
        N = points.shape[0]
        d = points.shape[1]
        # shape our input and test points into the right shape (N, 3, 1)
        x = x.unsqueeze(-1)
        points = points.unsqueeze(-1)
        # repeat each input point in the dataset across the third dimension
        x = x.repeat((1, 1, points.shape[0]))
        # repeate each test in the dataset across the first dimension
        comp_x = points.repeat((1, 1, x.shape[0]))
        comp_x = comp_x.permute((2, 1, 0))
        # now compare the input points and comparison points to see how many
        # are bigger and smaller
        x = self.ge(x, comp_x)
        orthants = []
        orthant_matrix = self.get_orthant_matrix(d)
        for i in range(int(np.power(2, d))):
            membership = 1.0
            for j in range(d):
                membership *= (float(orthant_matrix[i, j] < 0) + orthant_matrix[i, j] * x[:, j, :]).abs()
            orthant = torch.sum(membership, dim=0).float() / N
            orthants.append(orthant)
        return torch.stack(orthants, dim=1)

    def get_octants(self, x, points):
        N = points.shape[0]
        # shape our input and test points into the right shape (N, 3, 1)
        x = x.unsqueeze(-1)
        points = points.unsqueeze(-1)
        # repeat each input point in the dataset across the third dimension
        x = x.repeat((1, 1, points.shape[0]))
        # repeate each test in the dataset across the first dimension
        comp_x = points.repeat((1, 1, x.shape[0]))
        comp_x = comp_x.permute((2, 1, 0))
        # now compare the input points and comparison points to see how many
        # are bigger and smaller
        x = self.ge(x, comp_x)
        nx = (1 - torch.clone(x)).abs()
        # now use the comparisoned points to construct each octant (& is logical and)
        # nd = torch.pow(2, len(x.shape[1]))
        # os = torch.empty((n, x.shape[1], nd))
        # os = []
        # for i in nd:
        #    _o = torch.ones((x.shape[0]))
        #    for j in range(x.shape[1]):
        #        _o *= x[:, j, :]
        o1 = torch.sum(x[:, 0, :] * x[:, 1, :] * x[:, 2, :], dim=0).float() / N
        o2 = torch.sum(x[:, 0, :] * x[:, 1, :] * nx[:, 2, :], dim=0).float() / N
        o3 = torch.sum(x[:, 0, :] * nx[:, 1, :] * x[:, 2, :], dim=0).float() / N
        o4 = torch.sum(x[:, 0, :] * nx[:, 1, :] * nx[:, 2, :], dim=0).float() / N
        o5 = torch.sum(nx[:, 0, :] * x[:, 1, :] * x[:, 2, :], dim=0).float() / N
        o6 = torch.sum(nx[:, 0, :] * x[:, 1, :] * nx[:, 2, :], dim=0).float() / N
        o7 = torch.sum(nx[:, 0, :] * nx[:, 1, :] * x[:, 2, :], dim=0).float() / N
        o8 = torch.sum(nx[:, 0, :] * nx[:, 1, :] * nx[:, 2, :], dim=0).float() / N
        # return the stack of octants, should be (n, 8)
        return torch.stack([o1, o2, o3, o4, o5, o6, o7, o8], dim=1)



    def permute(self, pred=None, true=None, J=1_000):
        if pred is None:
            pred = self.pred
        if true is None:
            true = self.true
        all_pts = torch.cat((pred, true), dim=0)
        T = self(pred, true)
        T_ = torch.empty((J,))
        total_shape = pred.shape[0] + true.shape[0]
        for j in range(J):
            idx = torch.randperm(total_shape)
            idx1, idx2 = torch.chunk(idx, 2)
            _pred = all_pts[idx1]
            _true = all_pts[idx2]
            T_[j] = self(_pred, _true)
        return float(torch.sum(T < T_) + 1) / float(J + 2), T, T_
