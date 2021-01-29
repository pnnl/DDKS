from .ddks import ddKS
import torch
import warnings

class pdKS(ddKS):
    def __init__(self, soft=False, T=0.1, method='all', n_test_points=10,
                 pts=None, norm=False, oneway=True, plane_per_dim=10, bounds=[], covariance_planes=False,dataBounds=True, approx=True):
        super().__init__(soft, T, method, n_test_points,
                         pts, norm, oneway)
        # If Number of voxels/dimension is not specified then assume 10/dimension
        #self.planes= planes
        self.bounds = bounds
        self.approx = True
        self.dataBounds = True
        self.plane_per_dim = plane_per_dim
        self.covariance_planes = covariance_planes
    def setup(self, pred, true):
        '''
        Set Bounds using pred/true if dataBounds=False
        Normalize data to be between 0 and 1 using bounds
        Fill voxels
        '''
        self.pred = pred
        self.true = true
        self.d    = pred.shape[1]



        # start by using only tensors consider adding plane class
        self.planes   =  torch.zeros(self.d,self.plane_per_dim,2,3)#(dim,Plane,true/pred n1 higher/low)
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
        return torch.max(abs(self.diff[:,:,1:]))

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
        # Force Data to be between (0..1)*planesperdim
        self.pred = self.plane_per_dim * (self.pred - self.bounds[0, :]) / (self.max_bounds + 1e-4)
        self.true = self.plane_per_dim * (self.true - self.bounds[0, :]) / (self.max_bounds + 1e-4)

    def fill_voxels(self):
        '''
        Fill voxels lists: d1_vox and d2_vox with points in d1 and d2
        voxel_list.keys() contains all nonempty voxels
        '''
        #self.voxel_list = {}
        for ids in self.pred.long():
            ids = tuple(ids)
            for d,p in enumerate(ids):
                self.planes[d,p,0,0] += 1
            #if ids not in self.voxel_list:
             #   self.voxel_list[ids] = 1
            #else:
            #    self.voxel_list[ids].append(pt_id)
        for pt_id, ids in enumerate(self.true.long()):
            ids = tuple(ids)
            for d,p in enumerate(ids):
                self.planes[d,p,1,0] +=1

            #if ids not in self.voxel_list:
            #    self.voxel_list[ids] = 1
            #else:
            #    self.voxel_list[ids].append(pt_id)
        self.diff = self.planes[:,:,1,:] / self.true.shape[0] - self.planes[:,:,0,:] / self.pred.shape[
            0]  # Calculate difference in voxels

        self.diff[:,0,2] = self.diff[:,1:,0].sum(axis=1)
        for p in range(1, self.plane_per_dim):
            self.diff[:, p, 1] = self.diff[:, p-1, 0]+self.diff[d, p-1, 1]
            self.diff[:, p, 2] = self.diff[d, p-1, 2]-self.diff[:, p, 0]



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
        return float(torch.sum(T < T_) + 1) / float(J + 2), T, T_
        #return torch.sum(T_ > T) / float(J), T, T_
