from .ddks import ddKS
import torch
import warnings

'''
Voxel nd-KS test: vndKS 
Bins data points in voxels to calculate the CPDFs 
Linear time scaling (with data points)  
Should work as is in n-dimensions (untested currently)


Not Yet Implemented: 
    Sparse matricies to deal with higher dimensional empty voxels
'''


class vdKS(ddKS):
    def __init__(self, soft=False, T=0.1, method='all', n_test_points=10,
                 pts=None, norm=False, oneway=True, numVoxel=None, vox_per_dim=10, bounds=[], dataBounds=True, approx=True):
        super().__init__(soft, T, method, n_test_points,
                         pts, norm, oneway)
        # If Number of voxels/dimension is not specified then assume 10/dimension
        self.numVoxel = numVoxel
        self.bounds = bounds
        self.approx = True
        self.dataBounds = True
        self.vox_per_dim = vox_per_dim
    def setup(self, pred, true):
        '''
        Set Bounds using pred/true if dataBounds=False
        Normalize data to be between 0 and 1 using bounds
        Fill voxels
        '''
        self.pred = pred
        self.true = true
        self.d    = pred.shape[1]
        if self.numVoxel is None:
            self.numVoxel = (self.vox_per_dim * torch.ones(self.d)).long()
        self.set_bounds()
        if pred.shape[1] != true.shape[1] or pred.shape[1] != self.bounds.shape[1]:
            warnings.warn(f'Dimension Mismatch between d1,d2,bounds')
        self.normalize_data()
        self.fill_voxels()

    def calcD(self, pred, true):
        '''
        If approx==True don't look inside voxels for D calculation
        If approx==False do in-voxel comparisons
        :param pred: data distribution
        :param true:
        :return: D, returns ddks distance
        '''
        D = 0
        if self.approx:
            for v_id in self.voxel_list.keys():
                V_tmp = torch.max(self.calc_voxel_oct(v_id))
                if V_tmp > D:
                    D = V_tmp
            return D
        else:
            for v_id in self.voxel_list.keys():
                #For each v_id
                print("Not Implemented")

    ###
    # Setup sub-Functions
    ###
    def set_bounds(self):
        # If no bounds are specified use data to figure out bounds
        if (len(self.bounds) != 0) and not self.dataBounds:
            print("Reusing old/initilized bounds")
            return
        lb_p = torch.min(self.pred, dim=0).values
        ub_p = torch.max(self.pred, dim=0).values
        lb_t = torch.min(self.true, dim=0).values
        ub_t = torch.max(self.true, dim=0).values
        bounds = torch.zeros(2, self.d)
        for i in range(len(lb_p)):
            bounds[0, i] = min(lb_p[i], lb_t[i])
            bounds[1, i] = max(ub_p[i], ub_t[i])
        self.bounds = bounds
        self.max_bounds = self.bounds[1, :] - self.bounds[0, :]
        return

    def normalize_data(self):
        # Force Data to be between (0..1)*numVoxels
        self.pred = self.numVoxel * (self.pred - self.bounds[0, :]) / (self.max_bounds + 1e-4)
        self.true = self.numVoxel * (self.true - self.bounds[0, :]) / (self.max_bounds + 1e-4)

    def fill_voxels(self):
        '''
        Fill voxels lists: d1_vox and d2_vox with points in d1 and d2
        voxel_list.keys() contains all nonempty voxels
        '''
        self.voxel_list = {}
        self.pred_vox = torch.zeros([int(x) for x in self.numVoxel])
        self.true_vox = torch.zeros([int(x) for x in self.numVoxel])
        for pt_id, ids in enumerate(self.pred.long()):
            ids = tuple(ids)
            self.pred_vox[ids] += 1
            if ids not in self.voxel_list:
                self.voxel_list[ids] = [pt_id]
            else:
                self.voxel_list[ids].append(pt_id)
        for pt_id, ids in enumerate(self.true.long()):
            ids = tuple(ids)
            self.true_vox[ids] += 1
            if ids not in self.voxel_list:
                self.voxel_list[ids] = [pt_id]
            else:
                self.voxel_list[ids].append(pt_id)
        self.diff = self.true_vox / self.true.shape[0] - self.pred_vox / self.pred.shape[
            0]  # Calculate difference in voxels

    ###
    # calcD subfunctions
    ###
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
        d1_pts = self.pred[self.voxel_list[v_id][:self.pred_vox[v_id]]]
        d2_pts = self.true[self.voxel_list[v_id][self.pred_vox[v_id]:]]
        V1 = self.get_inside(pt, d1_pts)
        V2 = self.get_inside(pt, d2_pts)
        return V2 - V1

    def get_inside(self, x, points):
        '''
        Calculate the exact orthants for points
        :param x:
        :param points:
        :return:
        '''

    ###
    # Testing/Validation functions
    ###
    def permute(self, J=1_000):
        all_pts = torch.cat((self.pred, self.true), dim=0)
        T = self(self.pred, self.true)
        T_ = torch.empty((J,))
        total_shape = self.pred.shape[0] + self.true.shape[0]
        for j in range(J):
            idx = torch.randperm(total_shape)
            idx1, idx2 = torch.chunk(idx, 2)
            _d1 = all_pts[idx1]
            _d2 = all_pts[idx2]
            T_[j] = self(_d1, _d2)
        return torch.sum(T_ > T) / float(J), T, T_
