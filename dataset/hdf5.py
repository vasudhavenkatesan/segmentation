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
    def __init__(self, filepath, transform=None):
        logging.info('Initialising dataset from HDF5 files')
        self.images = []
        self.masks = []
        self.create_dataset(self, filepath)
        self.data_size = self.images.__len__()
        self.transform = transform

    def __getitem__(self, index):
        if self.transform:
            self.images[index] = transforms.padding(self.images[index], self.dimension, 0)
            self.masks[index] = transforms.padding((self.masks[index], self.dimension, 0))
            return self.images[index], self.masks[index]
        else:
            return self.images[index], self.masks[index]

    def __len__(self):
        return self.data_size

    @staticmethod
    def create_dataset(self, dirpath):
        paths = get_file_list_from_dir(dirpath)
        for file in paths:
            mask = dirpath + '/' + file.name.rpartition('img')[0] + 'mask.h5'
            with h5py.File(file, "r") as image_file:
                group = image_file['ITKImage']
                subgroup = group['0']
                self.images.append(torch.from_numpy(np.array(subgroup['VoxelData'])))
            with h5py.File(mask, "r") as mask_file:
                group = mask_file['ITKImage']
                subgroup = group['0']
                self.masks.append(torch.from_numpy(np.array(subgroup['VoxelData']).astype(numpy.float32)))
        logging.info('Completed initialisation')
