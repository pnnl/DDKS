import torch
import numpy as np

def make_true(N=100,device=torch.device("cpu")):
    n_per = 10
    N = N//n_per
    # speed of light
    c = 299_792_458.0 * 100.0
    # column 1 is x, columy 2 is y, column 3 is t
    cone = torch.empty((N*n_per, 3))
    plane = torch.empty((N*n_per, 3))
    ls = torch.empty((N*n_per, 1))
    phi = 15.0 * np.pi / 180.0
    beta = 1.0 / (1.4 * np.cos(phi))
    v = c * beta
    t_max = 0.0
    t_min = np.inf
    for i in np.arange(N):
        z = 20.0 * np.random.uniform()
        phi = 15.0 * np.pi / 180.0
        chi = np.cos(phi) * z
        for _n_per in range(n_per):
            theta = np.random.uniform(0.0, 2.0 * np.pi)
            x = chi * np.cos(theta)
            y = chi * np.sin(theta)
            d = np.sqrt(np.power(x, 2) + np.power(y, 2) + np.power(z, 2))
            track_t = (20.0 - z) / v
            t = 1.0E9 * ((d / c) + track_t)
            cone[i*n_per + _n_per, 0] = x
            cone[i*n_per + _n_per, 1] = y
            cone[i*n_per + _n_per, 2] = t
            ls[i*n_per + _n_per, 0] = z
    cone[:, 0] = (cone[:, 0] - cone[:, 0].min()) / (cone[:, 0].max() - cone[:, 0].min())
    cone[:, 1] = (cone[:, 1] - cone[:, 1].min()) / (cone[:, 1].max() - cone[:, 1].min())
    cone[:, 2] = (cone[:, 2] - cone[:, 2].min()) / (cone[:, 2].max() - cone[:, 2].min())
    true1 = cone.to(device)
    true2 = cone.to(device)
    true = true1#torch.cat((true1, true2), dim=0)
    labels = ls#torch.cat((ls, ls), dim=0)
    idx = torch.randperm(N*n_per)
    print(true.shape)
    true = true[idx, :]
    labels = labels[idx, :]
    return true
class Cone(object):
    def __init__(self, phi, x=0.0, y=0.0, background=False, bounds=[-100.0, 100.0, -100.0, 100.0], n_per=10):
        self.n_per = 10
        self.phi = phi * np.pi / 180.0
        self.x = x
        self.y = y
        self.background = background
        self.bounds = bounds

    def __call__(self, N=100):
        n_per = self.n_per
        N = N // n_per
        # speed of light
        c = 299_792_458.0 * 100.0
        # column 1 is x, columy 2 is y, column 3 is t
        cone = torch.empty((N * n_per, 3))
        ls = torch.empty((N * n_per, 1))
        beta = 1.0 / (1.4 * np.cos(self.phi))
        v = c * beta
        t_max = 0.0
        t_min = np.inf
        for i in np.arange(N):
            z = 20.0 * np.random.uniform()
            chi = np.cos(self.phi) * z
            for _n_per in range(n_per):
                theta = np.random.uniform(0.0, 2.0 * np.pi)
                x = chi * np.cos(theta)
                y = chi * np.sin(theta)
                d = np.sqrt(np.power(x, 2) + np.power(y, 2) + np.power(z, 2))
                track_t = 0.0  # (20.0 - z) / v
                t = 1.0E9 * ((d / c) + track_t)
                cone[i * n_per + _n_per, 0] = x
                cone[i * n_per + _n_per, 1] = y
                cone[i * n_per + _n_per, 2] = t
        background = torch.empty((N * n_per, 3))
        for i in np.arange(N * n_per):
            x = np.random.uniform(self.bounds[0], self.bounds[1])
            y = np.random.uniform(self.bounds[2], self.bounds[3])
            z = np.random.uniform(0.0, 20.0)
            phi = np.random.uniform(0.0, np.pi / 2.0)  # np.pi * 90.0 * torch.rand() / 180.0
            theta = np.random.uniform(0.0, 2.0 * np.pi)  # 2.0 * np.pi * torch.rand()
            chi = np.cos(phi) * 20.0
            _x = chi * np.cos(theta)
            _y = chi * np.sin(theta)
            d = np.sqrt(np.power(x - _x, 2.0) + np.power(y - _y, 2.0) + np.power(20.0 - z, 2.0))
            t = 1.0E9 * (d / c)
            background[i, 0] = x
            background[i, 1] = y
            background[i, 2] = t
        dataset = torch.empty((N * n_per, 3))
        n_background = int(self.background * N * n_per)
        idx = np.random.choice(np.arange(N * n_per), replace=False, size=(n_background,))
        dataset[:n_background, :] = background[idx, :]
        n_cone = N * n_per - n_background
        idx = np.random.choice(np.arange(N * n_per), replace=False, size=(n_cone,))
        dataset[n_background:n_background + n_cone, :] = cone[idx, :]
        idx = np.arange(N * n_per)
        np.random.shuffle(idx)
        dataset = dataset[idx, :]
        return dataset

