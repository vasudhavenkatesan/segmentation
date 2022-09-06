from pathlib import Path

import h5py
import torch
from torchvision import transforms
from torch.utils.data import Dataset
import logging

import config
from augmentation.transforms import *
from sklearn.utils.class_weight import compute_class_weight


def get_file_list_from_dir(filepath):
    p = Path(filepath)
    assert (p.is_dir())
    files = list(p.glob('**/*img.h5'))
    if len(files) < 1:
        logging.debug('Could not find hdf5 datasets')
        raise RuntimeError('No hdf5 datasets found')
    return files


class Hdf5Dataset(Dataset):
    def __init__(self, filepath, image_dim, contains_mask: bool = True):
        logging.info('Initialising dataset from HDF5 files')
        self.image_id = {}
        self.contains_mask = contains_mask
        # stores the image and label ids only
        self.get_image_id(self, filepath)
        self.dirpath = filepath
        self.rand_crop = RandomCrop3D((60, 506, 506), image_dim)

        mean, std = self.compute_mean_and_std()
        self.transform_norm = transforms.Compose([
            transforms.Normalize(mean=mean, std=std)
        ])

    def __getitem__(self, index):
        # lazy loading of data
        image, label = self.get_image_and_label(self, index)
        image, label = self.rand_crop(image, label)
        image = self.transform_norm(image)
        return image, label

    def __len__(self):
        return self.image_id.__len__()

    @staticmethod
    def get_image_id(self, dirpath):
        paths = get_file_list_from_dir(dirpath)
        i = 0
        for file in paths:
            self.image_id[i] = file
            i += 1
        return self.image_id

    @staticmethod
    def get_image_and_label(self, id):
        file = self.image_id[id]
        with h5py.File(file, "r") as image_file:
            group = image_file['ITKImage']
            subgroup = group['0']
            image = torch.from_numpy(np.array(subgroup['VoxelData']))
        if self.contains_mask:
            mask = self.dirpath + '/' + file.name.rpartition('img')[0] + 'mask.h5'
            with h5py.File(mask, "r") as mask_file:
                group = mask_file['ITKImage']
                subgroup = group['0']
                label = torch.from_numpy(np.array(subgroup['VoxelData']).astype(np.float32))
        logging.info(f'Loaded image {id} - {file}')
        return image, label

    def compute_mean_and_std(self):
        mean = 0.0
        std = 0.0
        for id_val, i in zip(self.image_id, range(0, len(self.image_id))):
            image, _ = self.get_image_and_label(self=self, id=id_val)
            mean += image.mean()
            std += image.std()

        mean /= len(self.image_id)
        std /= len(self.image_id)

        return mean, std
