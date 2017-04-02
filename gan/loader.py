import os
import glob
import numpy as np
import torch.utils.data as data

from PIL import Image


class BobRossDataset(data.Dataset):
    def __init__(self, filedir, transform=None, ext='*.png'):
        self.filelist = list(glob.iglob(os.path.join(filedir, ext)))
        self.images = []

        # Convert images to numpy
        for f in self.filelist:
            im = Image.open(f)
            im = im.resize((256, 256), Image.ANTIALIAS)
            self.images.append(np.asarray(im))

        # Transforms for images
        self.transform = transform

        # Length of dataset
        self._length = len(self.filelist)

    def __getitem__(self, index):
        img = self.images[index]

        if self.transform is not None:
            img = self.transform(img)

        return (img, )

    def __len__(self):
        return self._length
