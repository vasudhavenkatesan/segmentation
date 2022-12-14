import os
from glob import glob
from pathlib import Path

import h5py
import nrrd
import logging

import torch, itk
import tqdm
from skimage.transform import rescale
from torchvision import transforms
from torch.utils.data import Dataset
import config
import numpy as np
from augmentation.transforms import RandomCrop3D


def get_file_list_from_dir(filepath):
    p = Path(filepath)
    assert (p.is_dir())
    files = list(p.glob('embl*.h5'))
    if len(files) < 1:
        logging.debug('Could not find hdf5 datasets')
        raise RuntimeError('No hdf5 datasets found')
    return files


def load_file_chunkwise(path_to_file, crop_indices_zyx):
    with h5py.File(path_to_file, "r") as image_file:
        file = np.array(image_file['ITKImage']['0']['VoxelData'][crop_indices_zyx[0][0]:crop_indices_zyx[0][1],
                        crop_indices_zyx[1][0]:crop_indices_zyx[1][1],
                        crop_indices_zyx[2][0]:crop_indices_zyx[2][1]], dtype=np.float32)
    return torch.from_numpy(file)


def get_image_shape(path_to_image):
    with h5py.File(path_to_image, "r") as image_file:
        shape = image_file['ITKImage']['0']['VoxelData'].shape
    return shape


class Hdf5Dataset(Dataset):
    def __init__(self, filepath, reqd_image_dim, contains_mask: bool = True, mask_file_type: str = "h5",
                 is_test: bool = False, max_images_per_iteration: int = 250):
        logging.info('Initialising dataset from HDF5 files')
        self.image_id = {}
        self.contains_mask = contains_mask
        # stores the image and label ids only
        self.get_image_id(self, filepath)
        self.dirpath = filepath
        self.reqd_dim = reqd_image_dim
        self.num_images_per_iteration = max_images_per_iteration
        self.rand_crop = RandomCrop3D(reqd_image_dim)
        self.mask_file_type = mask_file_type
        self.is_test = is_test
        mean, std = self.compute_mean_and_std()
        self.transform_norm = transforms.Compose([
            transforms.Normalize(mean=mean, std=std)
        ])

    def __getitem__(self, index):
        random_image_index = np.random.random_integers(0, self.__len__() - 1)
        path_to_image = self.image_id[random_image_index]
        if self.is_test:
            image, label = self.get_full_image_and_label(path_to_image)
        else:
            image_shape = get_image_shape(path_to_image)
            crop_size = self.rand_crop(image_shape)
            image = load_file_chunkwise(path_to_image, crop_size)
            path_to_label = self.dirpath + '/pred_' + path_to_image.name.rpartition('rec')[0] + 'rec.h5'
            label = load_file_chunkwise(path_to_label, crop_size)
            label = (label > 0).float()
            logging.info(f'Loaded image and label chunkwise- {path_to_image}')

        image = self.transform_norm(image)
        return image, label

    def __len__(self):
        if self.image_id.__len__() > self.num_images_per_iteration:
            return self.num_images_per_iteration
        else:
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
    def get_full_image_and_label(self, path_to_image, binarize: bool = True):
        with h5py.File(path_to_image, "r") as image_file:
            image = torch.from_numpy(np.array(image_file['ITKImage']['0']['VoxelData']))
            if self.contains_mask:
                path_to_label = self.dirpath + '/pred_' + path_to_image.name.rpartition('rec')[0] + 'rec.h5'
                with h5py.File(path_to_label, "r") as mask_file:
                    label = torch.from_numpy(
                        np.array(mask_file['ITKImage']['0']['VoxelData'], dtype=np.float32))
                if binarize:
                    label = (label > 0).float()
        logging.info(f'Loaded full image and label - {path_to_image}')
        return image, label

    def compute_mean_and_std(self):
        mean = 0.0
        std = 0.0
        for id_val, i in zip(self.image_id, range(0, len(self.image_id))):
            path_to_image = self.image_id[id_val]
            image, _ = self.get_full_image_and_label(self, path_to_image)
            mean += image.mean()
            std += image.std()

        mean /= len(self.image_id)
        std /= len(self.image_id)

        return mean, std


def downsample_data():
    filepath = '../' + config.dataset_path

    list_of_all_mask_files = glob(os.path.join(filepath, '*.nrrd'))
    for path_mask in tqdm.tqdm(list_of_all_mask_files):
        export_folder = path_mask.rpartition('pred_')[0] + 'downsampled_mask'
        if not os.path.isdir(export_folder):
            os.makedirs(export_folder)
        mask = itk.GetArrayFromImage(itk.imread(path_mask))
        downsampled_mask = rescale(image=mask, scale=(1, 0.5, 0.5), order=0)
        output_filename = os.path.join(export_folder, f'{os.path.basename(path_mask)[:-4]}h5')

        itk.imwrite(itk.GetImageFromArray(downsampled_mask), output_filename)


def test():
    filepath = '../' + config.dataset_path
    dataset = Hdf5Dataset(filepath, reqd_image_dim=[64, 128, 128], contains_mask=True,
                          mask_file_type="nrrd")
    print(dataset.__getitem__(0))


if __name__ == "__main__":
    test()
