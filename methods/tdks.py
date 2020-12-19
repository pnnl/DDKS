'''
Work in progress (Mostly garbage in here right now)
Idea: Cut space into 2**d divisions each step until each segement has some threshold amount of particles or less
Problem: How to efficiently use new spatial relationships to quickly calculate ddKS
'''
class vox_tree(object):
    def __init__(self, d1, d2, threshold, bounds, threshold_type='raw'):
        '''
        Takes in two datasets (d1,d2), splits space into equal sized voxels until all voxels are above threshold
        Threshold types - percent: Split until each voxel has no more than X% of total points: no more than X points/per dataset

        Fill root -> Check Threshold -> Create kids -> Check threshold ->
        Assume Normalized data for now:
        Bounds are between 0..1 in all dimensions
        Diff in root is 0
        '''
        diff = 0
        self.root = Voxel(diff, None, 0)


class Voxel(object):
    def __init__(self, diff, parent, level):
        self.diff = diff
        self.parent = parent
        self.level = level
        self.kids = None


class vox_node(object):
    def __init__(self, d1_list, d2_list, bounds, parents, level, threshold, dim):
        self.d1 = len(d1_list)
        self.d2 = len(d2_list)
        self.bounds = bounds
        self.parents = parents
        self.level = level
        self.dim = dim
        self.kids = self.genkids(d1_list, d2_list, threshold)

    def genkids(self, d1_list, d2_list, threshold):
        if len(d1_list < threshold) and (d2_list < threshold):
            return None
        # Create empty voxels
        voxlist = torch.zeros([])
        # fill voxels
        voxlist = fill_voxels(d1_list, d2_list, vox)
        # For each non-empty voxel genkids()

        fillvoxels(d1_list, d2_list, self.bounds)

    def fill_voxels(self, pred, true):
        '''
        Fill voxels lists: d1_vox and d2_vox with points in d1 and d2
        voxel_list.keys() contains all nonempty voxels
        '''
        self.voxel_list = {}
        pred_vox = torch.zeros([int(2) for x in range(self.dim)])  # torch.zeros([int(x) for x in self.numVoxel])
        true_vox = torch.zeros([int(2) for x in range(self.dim)])  # torch.zeros([int(x) for x in self.numVoxel])
        bined_pred =

        for pt_id, ids in enumerate(self.pred.long()):
            ids = tuple(ids)
            self.pred_vox[ids] += 1
            if ids not in self.voxel_list:
                self.voxel_list[ids] = [pt_id]
            else:
                self.voxel_list[ids].append(pt_id)
        for pt_id, ids in enumerate(self.d2.long()):
            ids = tuple(ids)
            self.true_vox[ids] += 1
            if ids not in self.voxel_list:
                self.voxel_list[ids] = [pt_id]
            else:
                self.voxel_list[ids].append(pt_id)
        self.diff = self.true_vox / self.true.shape[0] - self.pred_vox / self.pred.shape[
            0]  # Calculate difference in voxels


class voxel(object):
    def __init__(self, d1, d2, edges, parent, level, children=[]):
        self.parent = parent
        self.edges = []
        self.pred_list = pred_list
        self.true_list = true_list

    def fill_voxels(self, threshold):
        '''
        Creates children
        :return:
        '''
        if len(self.pred_list) < threshold:
            return
        if self.level < levelmax:
            return
