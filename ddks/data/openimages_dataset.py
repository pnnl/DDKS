import os
import glob
import torch
import tqdm
import torchvision
from skimage import io
import numpy as np
from ddks.data import TwoSample
import ddks
from openimages.download import download_images


curr_path = os.path.dirname(ddks.data.__file__)
download_path = os.path.join(curr_path, 'openimages_data')

class OpenImagesDataset:
    def __init__(self, path=download_path, image_class='person'):
        self.image_class = image_class.lower()
        self.files = glob.glob(os.path.join(path, self.image_class, 'images', '*.jpg'))
        self.len = len(self.files)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        image = io.imread(self.files[idx])
        image = torch.from_numpy(image).float()
        return image, self.image_class

def build_pca_matrix(dataset1, dataset2, model):
    latent_spaces = []
    counter = 0
    for image, _ in tqdm.tqdm(dataset1):
        if len(image.shape) < 3:
            image = image.unsqueeze(-1).repeat((1, 1, 3))
        with torch.no_grad():
            model = model.to(torch.device('cuda:0'))
            image = image.to(torch.device('cuda:0'))
            latent_space = model(image.permute((2, 0, 1)).unsqueeze(0))
            latent_space = latent_space.detach().cpu()
        latent_spaces.append(latent_space)
        counter += 1
    for image, _ in tqdm.tqdm(dataset2):
        if len(image.shape) < 3:
            image = image.unsqueeze(-1).repeat((1, 1, 3))
        with torch.no_grad():
            model = model.to(torch.device('cuda:0'))
            image = image.to(torch.device('cuda:0'))
            latent_space = model(image.permute((2, 0, 1)).unsqueeze(0))
            latent_space = latent_space.detach().cpu()
        latent_spaces.append(latent_space)
        counter += 1
    latent_spaces = torch.cat(latent_spaces)
    _, _, V = torch.pca_lowrank(latent_spaces, 20)
    pcas = torch.matmul(latent_spaces, V[:, :20])
    pcas = pcas[:, :20]
    pcas1 = pcas[:len(dataset1), ...]
    pcas2 = pcas[len(dataset1):, ...]
    return pcas1, pcas2

class LS(TwoSample):
    name = 'LS'
    def __init__(self, force_rebuild=False, dimension=10, **kwargs):
        vehicle_path = os.path.join(curr_path, f'vehicle_latent_spaces.csv')
        person_path = os.path.join(curr_path, f'person_latent_spaces.csv')
        if not force_rebuild and (os.path.exists(vehicle_path)) and (os.path.exists(person_path)):
            latent_spaces1 = np.loadtxt(vehicle_path)
            latent_spaces2 = np.loadtxt(person_path)
        else:
            person = OpenImagesDataset(download_path, 'person')
            vehicle = OpenImagesDataset(download_path, 'truck')
            if len(person)  < 1000:
                exclusions_path = os.path.join(curr_path, 'openimages_exclusions.txt')
                download_images(download_path, ['Person', 'Truck'], exclusions_path=exclusions_path, csv_dir=download_path, limit=1000)
                person = OpenImagesDataset(download_path, 'person')
                vehicle = OpenImagesDataset(download_path, 'truck')
            self.model = torchvision.models.resnet18(pretrained=True)
            latent_spaces1, latent_spaces2 = build_pca_matrix(person, vehicle, self.model)
            np.savetxt(vehicle_path, latent_spaces1)
            np.savetxt(person_path, latent_spaces2)
            latent_spaces1 = np.loadtxt(vehicle_path)
            latent_spaces2 = np.loadtxt(person_path)
        latent_spaces1 = latent_spaces1[:, :dimension]
        latent_spaces2 = latent_spaces2[:, :dimension]
        latent_spaces1 = torch.from_numpy(latent_spaces1).float()
        latent_spaces2 = torch.from_numpy(latent_spaces2).float()
        def dgf_p(size, **kwargs):
            idx = np.arange(latent_spaces1.shape[0])
            np.random.shuffle(idx)
            idx = idx[:size[0]]
            return latent_spaces1[idx, :]
        def dgf_t(size, **kwargs):
            idx = np.arange(latent_spaces2.shape[0])
            np.random.shuffle(idx)
            idx = idx[:size[0]]
            return latent_spaces2[idx, :]
        params = dict()
        super().__init__(dgf_p=dgf_p, params_p=params,
                         dgf_t=dgf_t, params_t=params, **kwargs)
