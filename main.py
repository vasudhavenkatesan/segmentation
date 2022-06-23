import torch

from torch.utils.data import DataLoader

from dataset import hdf5
from augmentation import transforms

if __name__ == '__main__':
    # C,H,W of the image
    img_dimensions = [60, 506, 506]
    img_filepath = 'dataset/data/2_2_2_downsampled'
    training_data = hdf5.Hdf5Dataset(img_filepath, [16, 506, 506], True)
    train_dataloader = DataLoader(training_data, batch_size=4, shuffle=True, num_workers=2)
    val = training_data.__getitem__(0)
    print(val[1].shape)
