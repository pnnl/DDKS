import .ddks
import torch
import sys
import warnings

'''
Voxel nd-KS test: vndKS 
Bins data points in voxels to calculate the CPDFs 
Linear time scaling (with data points)  
Should work as is in n-dimensions (untested currently)


Not Yet Implemented: 
    Sparse matricies to deal with higher dimensional empty voxels
'''


class vndKS(ddKS):
    def __init__(self, soft=False, T=0.1, method='all', n_test_points=10,
                 pts=None, norm=False, oneway=True, numVoxel=None, d=3, bounds=[]):
        super().__init__(soft, T, method, n_test_points,
                 pts, norm, oneway)
        #If Number of voxels/dimension is not specified then assume 10/dimension
        if numVoxel is None:
            self.numVoxel = (10 * torch.ones(d)).long()
        else:
            self.numVoxel = numVoxel
        self.bounds = bounds


    def __call__(self, d1, d2, approx=True):
        self.d1 = d1
        self.d2 = d2
        self.set_bounds()
        if d1.shape[1] != d2.shape[1] or d1.shape[1] != self.bounds.shape[1]:
            warnings.warn(f'Dimension Mismatch between d1,d2,bounds')
        self.normalize_data()
        self.fill_voxels()
        D = 0
        if approx:
            for v_id in self.voxel_list.keys():
                V_tmp = torch.max(self.calc_voxel_oct(v_id))
                if V_tmp > D:
                    D = V_tmp
            return D
        else:
            for v_id in self.voxel_list.keys():
                print("Not Implemented")

    def set_bounds(self):
        # If no bounds are specified use data to figure out bounds
        if len(self.bounds) != 0:
            return
        lb_p = torch.min(self.d1, dim=0).values
        ub_p = torch.max(self.d1, dim=0).values
        lb_t = torch.min(self.d2, dim=0).values
        ub_t = torch.max(self.d2, dim=0).values
        bounds = torch.zeros(2, self.d)
        for i in range(len(lb_p)):
            bounds[0, i] = min(lb_p[i], lb_t[i])
            bounds[1, i] = max(ub_p[i], ub_t[i])
        self.bounds = bounds
        self.max_bounds = self.bounds[1, :] - self.bounds[0, :]
        return

    def normalize_data(self):
        # Force Data to be between (0..1)*numvoxels
        self.d1 = self.numvoxel * (self.d1 - self.bounds[0, :]) / (self.max_bounds + 1e-4)
        self.d2 = self.numvoxel * (self.d2 - self.bounds[0, :]) / (self.max_bounds + 1e-4)

    def get_voxel_index(self, pt):
        return tuple(pt.long())

    def fill_voxels(self):
        '''
        Fill voxels lists: d1_vox and d2_vox with points in d1 and d2
        voxel_list.keys() contains all nonempty voxels
        '''
        self.voxel_list = {}
        self.d1_vox = torch.zeros([int(x) for x in self.numvoxel])
        self.d2_vox = torch.zeros([int(x) for x in self.numvoxel])
        for pt_id, ids in enumerate(self.d1.long()):
            ids = tuple(ids)
            self.d1_vox[ids] += 1
            if ids not in self.voxel_list:
                self.voxel_list[ids] = [pt_id]
            else:
                self.voxel_list[ids].append(pt_id)
        for pt_id, ids in enumerate(self.d2.long()):
            ids = tuple(ids)
            self.d2_vox[ids] += 1
            if ids not in self.voxel_list:
                self.voxel_list[ids] = [pt_id]
            else:
                self.voxel_list[ids].append(pt_id)
        self.diff = self.d2_vox / self.d2.shape[0] - self.d1_vox / self.d1.shape[0]  # Calculate difference in voxels

    def get_index(self, v_id):
        ## Take in index to sum around spit out list of indicies
        inds = []
        for n in range(2 ** self.d):
            bitstring = format(n, f'0{self.d}b')
            ind = [slice(v_id[i]) if c == '0' else slice(v_id[i] + 1, None) for i, c in enumerate(bitstring)]
            inds.append(ind)
        return inds

    def calc_voxel_oct(self, v_id):
        ## Calculate
        inds = self.get_index(v_id)
        V_bin = [self.diff[inds[i]].sum() for i in range(2 ** self.d)]
        return abs(torch.tensor(V_bin))

    def calc_voxel_inside(self, pt, v_id):
        ## Take in point and generate octant values for inside voxel
        d1_pts = self.d1[self.voxel_list[v_id][:self.d1_vox[v_id]]]
        d2_pts = self.d2[self.voxel_list[v_id][self.d1_vox[v_id]:]]
        V1 = self.get_inside(pt, d1_pts)
        V2 = self.get_inside(pt, d2_pts)
        return V2 - V1

    def get_inside(self, x, points):
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

    def permute(self, J=1_000):
        all_pts = torch.cat((self.d1, self.d2), dim=0)
        T = self(self.d1, self.d2)
        T_ = torch.empty((J,))
        total_shape = self.d1.shape[0] + self.d2.shape[0]
        for j in range(J):
            idx = torch.randperm(total_shape)
            idx1, idx2 = torch.chunk(idx, 2)
            _d1 = all_pts[idx1]
            _d2 = all_pts[idx2]
            T_[j] = self(_d1, _d2)
        return torch.sum(T_ > T) / float(J), T, T_