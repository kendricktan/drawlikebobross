import os
import cv2
import h5py
import glob
import numpy as np

from PIL import Image
from tqdm import tqdm


if __name__ == "__main__":
    filelist = list(glob.iglob(os.path.join('images/', '*.png')))
    dataset_len = len(filelist)

    if not os.path.exists('dataset/'):
        os.makedirs('dataset/')
    h5dset = h5py.File('dataset/bobross.h5py')

    imgs = h5dset.create_dataset("images", (dataset_len, 256, 256, 3))
    sm_imgs = h5dset.create_dataset("smoothen", (dataset_len, 256, 256, 3))

    # Convert images to numpy
    for i in tqdm(range(dataset_len)):
        # Get file
        f = filelist[i]

        # Open file and convert to numpy
        im = Image.open(f)
        im = im.resize((256, 256), Image.ANTIALIAS)
        im_np = np.asarray(im)

        # Smoothen seg
        sim_np = cv2.pyrMeanShiftFiltering(im_np, 20, 45, 3)

        imgs[i] = im_np
        sm_imgs[i] = sim_np

    h5dset.close()
