from methods.ddks import ddKS
import torch
import warnings

#Radial method for calculating ddks
#Method should always = 'all' subsampling not supported

class rdks(ddKS):
    def __init__(self, soft=False, T=0.1, method='all', n_test_points=10,
                 pts=None, norm=False, oneway=True):
        super().__init__(soft, T, method, n_test_points,
                 pts, norm, oneway)

    def setup(self, pred, true):
        return

    def calcD(self, pred, true):
        self.find_corners(pred, true)
        d_from_corner_p = self.get_d_from_corner(pred)
        d_from_corner_t = self.get_d_from_corner(true)
        os_pp = self.get_octants_from_d(d_from_corner_p, d_from_corner_p)
        os_pt = self.get_octants_from_d(d_from_corner_t, d_from_corner_p)
        D1 = self.max((os_pp - os_pt).abs())
        if self.oneway:
            D = D1
        else:
            warnings.warn("Only Oneway implemented for rdks")
        if self.norm:
            D = D / float(pred.shape[0])
        return D
    def get_d_from_corner(self, x):
        _x = x.unsqueeze(-1).repeat((1, 1, 8))
        d = _x - self.corners
        d = torch.sqrt(torch.sum(torch.pow(d, 2.0), dim=1))
        return d

    def get_octants_from_d(self, d, d_test):
        os = torch.empty((d.shape[0], 8))
        sorted_ds = d
        sorted_test_args = torch.empty((d_test.shape)).long()
        for i in range(8):
            sorted_ds[:, i], _ = torch.sort(d[:, i])
            _, sorted_test_args[:, i] = torch.sort(d_test[:, i])
        N = d.shape[0]
        for octant in range(8):
            test_point_index = 0
            point_index = 0
            while test_point_index < N:
                idx = sorted_test_args[test_point_index, octant]
                while sorted_ds[point_index, octant] < d_test[idx, octant] and point_index < N - 1:
                    point_index += 1
                os[idx, octant] = point_index
                test_point_index += 1
        return os
    def find_corners(self, x1, x2):
        cs = torch.empty((8, 3))
        row = 0
        for d1 in [torch.min, torch.max]:
            for d2 in [torch.min, torch.max]:
                for d3 in [torch.min, torch.max]:
                    cs[row, :] = torch.tensor([d1(torch.tensor([d1(x1), d1(x2)])),
                                               d2(torch.tensor([d2(x1), d2(x2)])),
                                               d3(torch.tensor([d3(x1), d3(x2)]))])
                    row = row + 1
        self.corners = cs.T.unsqueeze(0)