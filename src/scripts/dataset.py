import sys
import os
import torch
import h5py
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms as T
import matplotlib.pyplot as plt
from torchvision import transforms

class LensingDataset(Dataset):
    """Loading Galaxy10 DECals test dataset from .h5 file.
    Test dataset has original images roated at random angles.
    Args:
        dataset_path (string) : path to h5 file
    """
    def __init__(self,dataset_path : str, transform = None) :
        self.dataset_path = dataset_path
        self.dataset = None
        self.transform = transform
        with h5py.File(self.dataset_path,"r") as f:
            self.length = len(f['labels'][()])

    def __getitem__(self, idx):

        if self.dataset is None:
            self.dataset = h5py.File(self.dataset_path,"r")

        original_image = self.dataset['original_images'][idx]
        lensed_image = self.dataset['lensed_images'][idx]
        label = torch.tensor(self.dataset['labels'][idx],dtype=torch.long)
        params = torch.tensor(self.dataset['lens_params_list'][idx], dtype=torch.float)
        
        if self.transform:
            original_image = self.transform(original_image)
            lensed_image = self.transform(lensed_image)
        return original_image, lensed_image, label, params

    def __len__(self):
        return self.length

if __name__ == '__main__':

    transform = transforms.ToTensor()
    PATH = '/Users/snehpandya/Projects/DE-2-Lensing/data/lensing_dataset_SIS.h5'
    dataset = LensingDataset(PATH,transform=transform)
    img, lensed_img, label, params = dataset[12342]
    print(img.shape)
    print(lensed_img.shape)
    print(params)