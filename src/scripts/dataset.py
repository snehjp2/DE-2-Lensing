import sys
import os
import torch
import h5py
import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms as T
import matplotlib.pyplot as plt
from torchvision import transforms
import time

class LensingDataset(Dataset):
    """Loading Galaxy10 DECals test dataset from .h5 file.
    Test dataset has original images roated at random angles.
    Args:
        dataset_path (string) : path to h5 file
    """
    def __init__(self,dataset_path : str, transform = None) :
        self.dataset_path = dataset_path
        #self.dataset = None
        self.transform = transform
        with h5py.File(self.dataset_path,"r") as f:
            self.original_images = f['original_images'][()]
            self.lensed_images = f['lensed_images'][()]
            self.labels = f['labels'][()]
            self.lens_params = f['lens_params_list'][()]
            self.length = len(f['labels'][()])

    def __getitem__(self, idx):

        #if self.dataset is None:
        #    self.dataset = h5py.File(self.dataset_path,"r")

        original_image = self.original_images[idx]
        lensed_image = self.lensed_images[idx]
        label = torch.tensor(self.labels[idx],dtype=torch.long)
        params = torch.tensor(self.lens_params[idx], dtype=torch.float)
        
        if self.transform:
            original_image = self.transform(original_image)
            lensed_image = self.transform(lensed_image)
        return original_image, lensed_image, label, params

    def __len__(self):
        return self.length

if __name__ == '__main__':

    transform = transforms.ToTensor()
    PATH = '/scratch/pandya.sne/E2_Lensing/lensing_dataset_SIS.h5'
    print("Loading dataset...")
    start = time.time()
    dataset = LensingDataset(PATH,transform=transform)
    end = time.time()
    print(f"Dataset loaded in {end-start} seconds")
    img, lensed_img, label, params = dataset[12342]
    print(img.shape)
    print(lensed_img.shape)
    print(params)