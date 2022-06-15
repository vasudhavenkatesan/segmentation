from pathlib import Path

import h5py
import numpy
import numpy as np
import torch
from torch.utils.data import Dataset
import logging
from augmentation import transforms


def get_file_list_from_dir(filepath):
    p = Path(filepath)
    assert (p.is_dir())
    files = list(p.glob('**/*img.h5'))
    if len(files) < 1:
        logging.debug('Could not find hdf5 datasets')
        raise RuntimeError('No hdf5 datasets found')
    return files


class Hdf5Dataset(Dataset):
    def __init__(self, filepath, image_dim, transform=None, contains_mask: bool = True):
        logging.info('Initialising dataset from HDF5 files')
        self.images = []
        self.masks = []
        self.contains_mask = contains_mask
        self.create_dataset(self, filepath)
        self.data_size = self.images.__len__()
        self.transform = transform
        self.dimension = image_dim
        self.images = self.transform_fn(self.images)
        self.masks = self.transform_fn(self.masks, is_mask=True)

    def __getitem__(self, index):
        return self.images[index], self.masks[index]

    def __len__(self):
        return self.data_size

    def transform_fn(self, data, is_mask: bool = False):
        for i in range(0, len(data)):
            data[i] = transforms.resize_image(self.dimension, data[i], is_mask)
        return data

    @staticmethod
    def create_dataset(self, dirpath):
        paths = get_file_list_from_dir(dirpath)
        for file in paths:
            mask = dirpath + '/' + file.name.rpartition('img')[0] + 'mask.h5'
            with h5py.File(file, "r") as image_file:
                group = image_file['ITKImage']
                subgroup = group['0']
                self.images.append(torch.from_numpy(np.array(subgroup['VoxelData'])))
            if self.contains_mask:
                with h5py.File(mask, "r") as mask_file:
                    group = mask_file['ITKImage']
                    subgroup = group['0']
                    mask_with_ch = np.array(subgroup['VoxelData']).astype(numpy.float32)
                    mask_labels = torch.from_numpy(mask_with_ch)
                    # replace mask label of 255 with 2
                    mask_labels[mask_labels == 255] = 2
                    self.masks.append(mask_labels)

        logging.info('Completed initialisation')
