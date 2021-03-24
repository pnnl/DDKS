import unittest
import torch
import sys
sys.path.append('/Users/jack755/PycharmProjects/ddks/')

import ddks



class MethodTests1D(unittest.TestCase):
    def setUp(self) -> None:
        d = 1
        n = 10
        self.pred = torch.normal(0,1.0,(n,d))
        self.true = torch.normal(0, 1.0, (n, d))
        self.true2 = torch.normal(0, 1.0, (2*n, d))
        self.pred1pt = torch.normal(0,1.0,(1,d))

    def test_ddks(self):
        xdks = ddks.methods.ddKS()
        self.assertGreaterEqual(1.0, xdks(self.pred,self.true))
    def test_vdks(self):
        xdks = ddks.methods.vdKS()
        self.assertGreaterEqual(1.0, xdks(self.pred,self.true))
    def test_rdks(self):
            xdks = ddks.methods.rdKS()
            self.assertGreaterEqual(1.0, xdks(self.pred, self.true))
    def test_ddks_diffpt(self):
        xdks = ddks.methods.ddKS()
        self.assertGreaterEqual(1.0, xdks(self.pred,self.true2))
    def test_vdks_diffpt(self):
        xdks = ddks.methods.vdKS()
        self.assertGreaterEqual(1.0, xdks(self.pred,self.true2))
    def test_rdks_diffpt(self):
        xdks = ddks.methods.rdKS()
        self.assertGreaterEqual(1.0, xdks(self.pred, self.true2))
    def test_ddks_onept(self):
        xdks = ddks.methods.ddKS()
        self.assertGreaterEqual(1.0, xdks(self.pred1pt, self.true))
    def test_vdks_onept(self):
        xdks = ddks.methods.vdKS()
        self.assertGreaterEqual(1.0, xdks(self.pred1pt, self.true))
    def test_rdks_onept(self):
        xdks = ddks.methods.rdKS()
        self.assertGreaterEqual(1.0, xdks(self.pred1pt, self.true))
    def test_ddks_diffptrev(self):
        xdks = ddks.methods.ddKS()
        self.assertGreaterEqual(1.0, xdks(self.true2,self.pred))
    def test_vdks_diffptrev(self):
        xdks = ddks.methods.vdKS()
        self.assertGreaterEqual(1.0, xdks(self.true2,self.pred))
    def test_rdks_diffptrev(self):
        xdks = ddks.methods.rdKS()
        self.assertGreaterEqual(1.0, xdks(self.true2, self.pred))
    def test_ddks_oneptrev(self):
        xdks = ddks.methods.ddKS()
        self.assertGreaterEqual(1.0, xdks(self.true,self.pred1pt))
    def test_vdks_oneptrev(self):
        xdks = ddks.methods.vdKS()
        self.assertGreaterEqual(1.0, xdks(self.true,self.pred1pt))
    def test_rdks_oneptrev(self):
        xdks = ddks.methods.rdKS()
        self.assertGreaterEqual(1.0, xdks(self.true,self.pred1pt))

    def test_permute_d_diff(self):
        xdks = ddks.methods.ddKS()
        p, _, _ = xdks.permute(pred=self.pred, true=self.true2, J=100)
        self.assertGreaterEqual(1.0, p)

    def test_permute_v_diff(self):
        xdks = ddks.methods.vdKS()
        p, _, _ = xdks.permute(pred=self.pred, true=self.true2, J=100)
        self.assertGreaterEqual(1.0, p)

    def test_permute_r_diff(self):
        xdks = ddks.methods.rdKS()
        p, _, _ = xdks.permute(pred=self.pred, true=self.true2, J=100)
        self.assertGreaterEqual(1.0, p)

    def test_permute_d_1pt(self):
        xdks = ddks.methods.ddKS()
        p, _, _ = xdks.permute(pred=self.pred1pt, true=self.true, J=100)
        self.assertGreaterEqual(1.0, p)

    def test_permute_v_1pt(self):
        xdks = ddks.methods.vdKS()
        p, _, _ = xdks.permute(pred=self.pred1pt, true=self.true, J=100)
        self.assertGreaterEqual(1.0, p)

    def test_permute_r_1pt(self):
        xdks = ddks.methods.rdKS()
        p, _, _ = xdks.permute(pred=self.pred1pt, true=self.true, J=100)
        self.assertGreaterEqual(1.0, p)

class MethodTests2D(unittest.TestCase):
    def setUp(self) -> None:
        d = 2
        n = 10
        self.pred = torch.normal(0,1.0,(n,d))
        self.true = torch.normal(0, 1.0, (n, d))
        self.true2 = torch.normal(0, 1.0, (2 * n, d))
        self.pred1pt = torch.normal(0, 1.0, (1, d))
    def test_ddks(self):
        xdks = ddks.methods.ddKS()
        self.assertGreaterEqual(1.0, xdks(self.pred,self.true))
    def test_vdks(self):
        xdks = ddks.methods.vdKS()
        self.assertGreaterEqual(1.0, xdks(self.pred,self.true))
    def test_rdks(self):
            xdks = ddks.methods.rdKS()
            self.assertGreaterEqual(1.0, xdks(self.pred, self.true))
    def test_ddks_diffpt(self):
        xdks = ddks.methods.ddKS()
        self.assertGreaterEqual(1.0, xdks(self.pred,self.true2))
    def test_vdks_diffpt(self):
        xdks = ddks.methods.vdKS()
        self.assertGreaterEqual(1.0, xdks(self.pred,self.true2))
    def test_rdks_diffpt(self):
        xdks = ddks.methods.rdKS()
        self.assertGreaterEqual(1.0, xdks(self.pred, self.true2))
    def test_ddks_onept(self):
        xdks = ddks.methods.ddKS()
        self.assertGreaterEqual(1.0, xdks(self.pred1pt, self.true))
    def test_vdks_onept(self):
        xdks = ddks.methods.vdKS()
        self.assertGreaterEqual(1.0, xdks(self.pred1pt, self.true))
    def test_rdks_onept(self):
        xdks = ddks.methods.rdKS()
        self.assertGreaterEqual(1.0, xdks(self.pred1pt, self.true))
    def test_ddks_diffptrev(self):
        xdks = ddks.methods.ddKS()
        self.assertGreaterEqual(1.0, xdks(self.true2,self.pred))
    def test_vdks_diffptrev(self):
        xdks = ddks.methods.vdKS()
        self.assertGreaterEqual(1.0, xdks(self.true2,self.pred))
    def test_rdks_diffptrev(self):
        xdks = ddks.methods.rdKS()
        self.assertGreaterEqual(1.0, xdks(self.true2, self.pred))
    def test_ddks_oneptrev(self):
        xdks = ddks.methods.ddKS()
        self.assertGreaterEqual(1.0, xdks(self.true,self.pred1pt))
    def test_vdks_oneptrev(self):
        xdks = ddks.methods.vdKS()
        self.assertGreaterEqual(1.0, xdks(self.true,self.pred1pt))
    def test_rdks_oneptrev(self):
        xdks = ddks.methods.rdKS()
        self.assertGreaterEqual(1.0, xdks(self.true,self.pred1pt))
    def test_permute_d_diff(self):
        xdks = ddks.methods.ddKS()
        p,_,_= xdks.permute(pred=self.pred, true=self.true2,J=100)
        self.assertGreaterEqual(1.0, p)
    def test_permute_v_diff(self):
        xdks = ddks.methods.vdKS()
        p,_,_= xdks.permute(pred=self.pred, true=self.true2,J=100)
        self.assertGreaterEqual(1.0, p)
    def test_permute_r_diff(self):
        xdks = ddks.methods.rdKS()
        p,_,_= xdks.permute(pred=self.pred, true=self.true2,J=100)
        self.assertGreaterEqual(1.0, p)
    def test_permute_d_1pt(self):
        xdks = ddks.methods.ddKS()
        p,_,_= xdks.permute(pred=self.pred1pt, true=self.true,J=100)
        self.assertGreaterEqual(1.0, p)
    def test_permute_v_1pt(self):
        xdks = ddks.methods.vdKS()
        p,_,_= xdks.permute(pred=self.pred1pt, true=self.true,J=100)
        self.assertGreaterEqual(1.0, p)
    def test_permute_r_1pt(self):
        xdks = ddks.methods.rdKS()
        p,_,_= xdks.permute(pred=self.pred1pt, true=self.true,J=100)
        self.assertGreaterEqual(1.0, p)

class MethodTests3D(unittest.TestCase):

    def setUp(self) -> None:
        d = 3
        n = 10
        self.pred = torch.normal(0, 1.0, (n, d))
        self.true = torch.normal(0, 1.0, (n, d))
        self.true2 = torch.normal(0, 1.0, (2 * n, d))
        self.pred1pt = torch.normal(0, 1.0, (1, d))
    def test_ddks(self):
        xdks = ddks.methods.ddKS()
        self.assertGreaterEqual(1.0, xdks(self.pred,self.true))
    def test_vdks(self):
        xdks = ddks.methods.vdKS()
        self.assertGreaterEqual(1.0, xdks(self.pred,self.true))
    def test_rdks(self):
        xdks = ddks.methods.rdKS()
        self.assertGreaterEqual(1.0, xdks(self.pred, self.true))
    def test_permute_d(self):
        xdks = ddks.methods.ddKS()
        p,_,_= xdks.permute(pred=self.pred, true=self.true,J=100)
        self.assertGreaterEqual(1.0, p)
    def test_permute_v(self):
        xdks = ddks.methods.vdKS()
        p,_,_= xdks.permute(pred=self.pred, true=self.true,J=100)
        self.assertGreaterEqual(1.0, p)
    def test_permute_r(self):
        xdks = ddks.methods.rdKS()
        p,_,_= xdks.permute(pred=self.pred, true=self.true,J=100)
        self.assertGreaterEqual(1.0, p)
    def test_ddks_diffpt(self):
        xdks = ddks.methods.ddKS()
        self.assertGreaterEqual(1.0, xdks(self.pred,self.true2))
    def test_vdks_diffpt(self):
        xdks = ddks.methods.vdKS()
        self.assertGreaterEqual(1.0, xdks(self.pred,self.true2))
    def test_rdks_diffpt(self):
        xdks = ddks.methods.rdKS()
        self.assertGreaterEqual(1.0, xdks(self.pred, self.true2))
    def test_ddks_onept(self):
        xdks = ddks.methods.ddKS()
        self.assertGreaterEqual(1.0, xdks(self.pred1pt, self.true))
    def test_vdks_onept(self):
        xdks = ddks.methods.vdKS()
        self.assertGreaterEqual(1.0, xdks(self.pred1pt, self.true))
    def test_rdks_onept(self):
        xdks = ddks.methods.rdKS()
        self.assertGreaterEqual(1.0, xdks(self.pred1pt, self.true))
    def test_ddks_diffptrev(self):
        xdks = ddks.methods.ddKS()
        self.assertGreaterEqual(1.0, xdks(self.true2,self.pred))
    def test_vdks_diffptrev(self):
        xdks = ddks.methods.vdKS()
        self.assertGreaterEqual(1.0, xdks(self.true2,self.pred))
    def test_rdks_diffptrev(self):
        xdks = ddks.methods.rdKS()
        self.assertGreaterEqual(1.0, xdks(self.true2, self.pred))
    def test_ddks_oneptrev(self):
        xdks = ddks.methods.ddKS()
        self.assertGreaterEqual(1.0, xdks(self.true,self.pred1pt))
    def test_vdks_oneptrev(self):
        xdks = ddks.methods.vdKS()
        self.assertGreaterEqual(1.0, xdks(self.true,self.pred1pt))
    def test_rdks_oneptrev(self):
        xdks = ddks.methods.rdKS()
        self.assertGreaterEqual(1.0, xdks(self.true,self.pred1pt))

    def test_permute_d_diff(self):
        xdks = ddks.methods.ddKS()
        p,_,_= xdks.permute(pred=self.pred, true=self.true2,J=100)
        self.assertGreaterEqual(1.0, p)
    def test_permute_v_diff(self):
        xdks = ddks.methods.vdKS()
        p,_,_= xdks.permute(pred=self.pred, true=self.true2,J=100)
        self.assertGreaterEqual(1.0, p)
    def test_permute_r_diff(self):
        xdks = ddks.methods.rdKS()
        p,_,_= xdks.permute(pred=self.pred, true=self.true2,J=100)
        self.assertGreaterEqual(1.0, p)
    def test_permute_d_1pt(self):
        xdks = ddks.methods.ddKS()
        p,_,_= xdks.permute(pred=self.pred1pt, true=self.true,J=100)
        self.assertGreaterEqual(1.0, p)
    def test_permute_v_1pt(self):
        xdks = ddks.methods.vdKS()
        p,_,_= xdks.permute(pred=self.pred1pt, true=self.true,J=100)
        self.assertGreaterEqual(1.0, p)
    def test_permute_r_1pt(self):
        xdks = ddks.methods.rdKS()
        p,_,_= xdks.permute(pred=self.pred1pt, true=self.true,J=100)
        self.assertGreaterEqual(1.0, p)

class ddKSTest(unittest.TestCase):
    def setUp(self) -> None:
        d = 3
        n = 10
        self.pred = torch.normal(0, 1.0, (n, d))
        self.true = torch.normal(0, 1.0, (n, d))
    def test_soft(self):
        xdks = ddks.methods.ddKS(soft=True)
        self.assertGreaterEqual(1.0, xdks(self.pred, self.true))
    def test_norm(self):
        xdks = ddks.methods.ddKS(norm=True)
        self.assertGreaterEqual(1.0, xdks(self.pred, self.true))
    def test_oneway(self):
        xdks = ddks.methods.ddKS(oneway=False)
        self.assertGreaterEqual(1.0, xdks(self.pred, self.true))
    def test_subsample(self):
        xdks = ddks.methods.ddKS(method='subsample')
        self.assertGreaterEqual(1.0, xdks(self.pred, self.true))
    def test_linear(self):
        xdks = ddks.methods.ddKS(method='linear')
        self.assertGreaterEqual(1.0, xdks(self.pred, self.true))

class rdKSTest(unittest.TestCase):
    def setUp(self) -> None:
        d = 3
        n = 10
        self.pred = torch.normal(0, 1.0, (n, d))
        self.true = torch.normal(0, 1.0, (n, d))
    def test_norm(self):
        xdks = ddks.methods.rdKS(norm=True)
        self.assertGreaterEqual(1.0, xdks(self.pred, self.true))



class VoxelTests(unittest.TestCase):

    def setUp(self) -> None:
        d = 3
        n = 10
        self.pred = torch.normal(0, 1.0, (n, d))
        self.true = torch.normal(0, 1.0, (n, d))
    def test_numVox(self):
        numVox = torch.tensor([4,2,1])
        xdks = ddks.methods.vdKS(numVoxel=numVox)
        self.assertGreaterEqual(1.0, xdks(self.pred, self.true))







if __name__ == '__main__':
    unittest.main()
