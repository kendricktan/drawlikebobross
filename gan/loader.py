import os
import glob
import h5py
import numpy as np
import torch.utils.data as data

from PIL import Image


class BobRossDataset(data.Dataset):
    def __init__(self, filedir, transform=None):
        self.data = h5py.File(filedir, 'r')

        self.images = self.data['images']
        self.smoothen = self.data['smoothen']

        # Transforms for images
        self.transform = transform

        # Length of dataset
        self._length = len(self.images)

    def __getitem__(self, index):
        img = self.images[index]
        smth_img = self.smoothen[index]

        if self.transform is not None:
            img = self.transform(img)
            smth_img = self.transform(smth_img)

        return (img, smth_img)

    def __len__(self):
        return self._length
