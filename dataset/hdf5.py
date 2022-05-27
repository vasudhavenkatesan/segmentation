from pathlib import Path

import h5py
import numpy as np
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader


def get_file_list_from_dir(filepath):
    p = Path(filepath)
    assert (p.is_dir())
    files = list(p.glob('**/*img.h5'))
    if len(files) < 1:
        raise RuntimeError('No hdf5 datasets found')
    return files


class Hdf5Dataset(Dataset):
    def __init__(self, filepath, phase, load_data, data_cache_size=3):
        print('checking 1')
        self.images, self.masks = self.initialise_data(self, filepath)
        self.data_size = self.images.__len__()
        self.load_data = load_data
        self.cache_size = data_cache_size
        print(f'Size of data {self.data_size}')

    def __getitem__(self, index):
        return self.images[index], self.masks[index]

    def __len__(self):
        return self.data_size

    @staticmethod
    def initialise_data(self, dirpath):
        images = []
        masks = []
        paths = get_file_list_from_dir(dirpath)
        for file in paths:
            mask = dirpath + '/' + file.name.rpartition('img')[0] + 'mask.h5'
            with h5py.File(file, "r") as image_file:
                group = image_file['ITKImage']
                subgroup = group['0']
                images.append(np.array(subgroup['VoxelData']))
            with h5py.File(mask, "r") as mask_file:
                group = mask_file['ITKImage']
                subgroup = group['0']
                masks.append(np.array(subgroup['VoxelData']))
        return images, masks
