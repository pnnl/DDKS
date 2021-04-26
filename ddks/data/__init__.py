'''
Files to generate data for ddks tests

We want:
n-d gaussian
cone - or is this waiting until subsequent publications
Multimodal gaussians?
'''
import torch
import torchvision
from .cone import  Cone
from .smalldata import SmallDataSet
from .cone import make_true

class Dataset:
    def __init__(self, dimension=3, dgf=lambda x: x, params=dict(), sample_size=100):
        self.d = dimension
        self.dgf = dgf
        self.params = params
        self.sample_size = sample_size
        self.len = 200
        self.i_batch = 0

    def __len__(self):
        return self.len

    def __iter__(self):
        self.i_batch = 0
        return self

    def __next__(self):
        if self.i_batch <= self.len:
            result = self.dgf(size=(self.sample_size, self.d), **self.params)
            self.i_batch += 1
            return result
        else:
            raise StopIteration

class TwoSample:
    def __init__(self, dimension=3, dgf_p=lambda x: x, dgf_t=lambda x: x, params_p=dict(), params_t=dict(),
                 sample_size=100):
        self.dataset_p = Dataset(dimension=dimension, dgf=dgf_p, params=params_p, sample_size=sample_size)
        self.dataset_t = Dataset(dimension=dimension, dgf=dgf_t, params=params_t, sample_size=sample_size)
        self.len = len(self.dataset_p)
        self.i_batch = 0

    def __len__(self):
        return self.len

    def __iter__(self):
        self.i_batch = 0
        self.dataset_p.i_batch = 0
        self.dataset_t.i_batch = 0
        return self

    def __next__(self):
        if self.i_batch <= self.len:
            result_p = next(self.dataset_p)
            result_t = next(self.dataset_t)
            self.i_batch += 1
            return (result_p, result_t)
        else:
            raise StopIteration

class GVM(TwoSample):
    name = 'GVM'
    def __init__(self, mean_p=1.0, mean_t=0.0, std=1.0, **kwargs):
        dgf = torch.normal
        params_p = dict(mean=mean_p, std=std)
        params_t = dict(mean=mean_t, std=std)
        super().__init__(dgf_p=dgf, params_p=params_p,
                         dgf_t=dgf, params_t=params_t, **kwargs)

class GVS(TwoSample):
    name = 'GVS'
    def __init__(self, std_p=1.0, std_t=0.5, mean=0.0, **kwargs):
        dgf = torch.normal
        params_p = dict(mean=mean, std=std_p)
        params_t = dict(mean=mean, std=std_t)
        super().__init__(dgf_p=dgf, params_p=params_p,
                         dgf_t=dgf, params_t=params_t, **kwargs)

class DVU(TwoSample):
    name = 'DVU'
    def __init__(self, width_p=0.0, **kwargs):
        self.uniform_object = torch.distributions.Uniform(low=0.0, high=1.0)
        def dgf_p(size, **kwargs):
            diag_size = [size[0], 1]
            u = self.uniform_object.sample(sample_shape=size)
            diag = self.uniform_object.sample(sample_shape=diag_size).repeat(1, size[1])
            b = (1.0 - width_p)
            diag = (b * diag + width_p * u)
            return diag
        def dgf_t(size, **kwargs):
            return self.uniform_object.sample(sample_shape=size)
        params = dict()
        super().__init__(dgf_p=dgf_p, params_p=params,
                         dgf_t=dgf_t, params_t=params, **kwargs)

class DVUHighDim(TwoSample):
    name = 'DVUHighDim'
    def __init__(self, width_p=0.0, n_diag_dims=2, **kwargs):
        self.uniform_object = torch.distributions.Uniform(low=0.0, high=1.0)
        def dgf_p(size, **kwargs):
            diag_size = [size[0], 1]
            u_size = [size[0], n_diag_dims]
            u = self.uniform_object.sample(sample_shape=u_size)
            diag = self.uniform_object.sample(sample_shape=diag_size).repeat(1, n_diag_dims)
            b = (1.0 - width_p)
            diag = (b * diag + width_p * u)
            non_diag_size = [size[0], size[1] - n_diag_dims]
            overall_u = self.uniform_object.sample(sample_shape=non_diag_size)
            diag = torch.cat((overall_u, diag), dim=1)
            return diag
        def dgf_t(size, **kwargs):
            return self.uniform_object.sample(sample_shape=size)
        params = dict()
        super().__init__(dgf_p=dgf_p, params_p=params,
                         dgf_t=dgf_t, params_t=params, **kwargs)

class Skew(TwoSample):
    name = 'Skew'
    def __init__(self, lambda_p=1.0, lambda_t=2.0, **kwargs):
        self.exp_p = torch.distributions.Exponential(lambda_p)
        self.exp_t = torch.distributions.Exponential(lambda_t)
        def dgf_p(size, **kwargs):
            return self.exp_p.sample(sample_shape=size)
        def dgf_t(size, **kwargs):
            return self.exp_t.sample(sample_shape=size)
        params = dict()
        super().__init__(dgf_p=dgf_p, params_p=params,
                         dgf_t=dgf_t, params_t=params, **kwargs)

class MM(TwoSample):
    name = 'MM'
    def __init__(self, mean_p=1.0, mean_t=0.0, std=1.0, noise_fraction=0.5, **kwargs):
        self.noise_fraction = noise_fraction
        self.normal_p = torch.distributions.Normal(loc=mean_p, scale=std)
        self.normal_t = torch.distributions.Normal(loc=mean_t, scale=std)
        self.uniform = torch.distributions.Uniform(low=-3.0*std, high=3.0*std)
        def dgf_p(size, **kwargs):
            size_n = list(size)
            size_n[0] = int(size_n[0] * (1.0 - noise_fraction))
            size_u = list(size)
            size_u[0] = size[0] - size_n[0]
            if size_n[0] > 0:
                normal = self.normal_p.sample(sample_shape=size_n)
            else:
                normal = torch.empty(size_n)
            if size_u[0] > 0:
                uniform = self.uniform.sample(sample_shape=size_u)
            else:
                uniform = torch.empty(size_u)
            sample = torch.cat((normal, uniform), dim=0)
            sample = sample[torch.randperm(sample.size()[0])]
            return sample
        def dgf_t(size, **kwargs):
            size_n = list(size)
            size_n[0] = int(size_n[0] * (1.0 - noise_fraction))
            size_u = list(size)
            size_u[0] = size[0] - size_n[0]
            if size_n[0] > 0:
                normal = self.normal_t.sample(sample_shape=size_n)
            else:
                normal = torch.empty(size_n)
            if size_u[0] > 0:
                uniform = self.uniform.sample(sample_shape=size_u)
            else:
                uniform = torch.empty(size_u)
            sample = torch.cat((normal, uniform), dim=0)
            sample = sample[torch.randperm(sample.size()[0])]
            return sample
        params = dict()
        super().__init__(dgf_p=dgf_p, params_p=params,
                         dgf_t=dgf_t, params_t=params, **kwargs)



def set_dgen(mean,std):
    def dgen(n, d):
        '''
        Method for generating data consider using methods from ddks.data
        :   param n: Number of entries
        :param d: dimension of data
        :return:
        '''
        return torch.normal(mean,std,(n, d))
    return dgen

def set_dgenNoise(mean,std,noise_percent,noise_function = None):
    if noise_function == None:
        noise_function = torch.rand
    def dgen(n,d):
        num_noise = int(noise_percent * n)
        goodvals = torch.normal(mean,std,(n-num_noise,d))
        noisevals =  noise_function(num_noise,d)
        return torch.cat([goodvals,noisevals])
    return dgen

def dgen_norm(n,d):
    return torch.rand(n,d)
def set_dgen_poisson(scale):
    def dgen(n,d):
        return torch.poisson(torch.ones(n,d)*scale)
def bgcone_wrap(n,d):
    func = data.Cone(15)
    stuff = func(n)
    filtered_data = stuff[~torch.any(stuff.isnan(), dim=1)]
    return filtered_data
def cone_wrap(n,d):
    stuff = data.make_true(N=n)
    filtered_data = stuff[~torch.any(stuff.isnan(), dim=1)]
    return filtered_data
def samp1(n,d):
    data=[]
    for i in range(n):
        rnd=np.random.uniform(0,1)
        s = torch.tensor([rnd for z in range(d)])
        data.append(s)
    return torch.stack(data)
def samp2(n,d):
    return torch.tensor(np.random.uniform(0,1,size=(n,d)))
