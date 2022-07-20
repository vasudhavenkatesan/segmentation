import os.path

import torch

from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from dataset import hdf5
from augmentation import transforms
from torch.utils.tensorboard import SummaryWriter
from unet.unet import UNET
from datetime import datetime
from utils import plot_image
import numpy as np

if __name__ == '__main__':
    # C,H,W of the image
    img_dimensions = [60, 506, 506]
    img_filepath = 'dataset/data/2_2_2_downsampled'
    training_data = hdf5.Hdf5Dataset(img_filepath, (16, 506, 506), True)
    # train_dataloader = DataLoader(training_data, batch_size=1, shuffle=True, num_workers=1)
    print('Initialised training')
    val1 = training_data.__getitem__(1)
    print(f'image - {val1[0]}')
    print(f'mask - {val1[1]}')
