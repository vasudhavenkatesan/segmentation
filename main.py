import os.path

import torch

from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from dataset import hdf5
from augmentation import transforms
from torch.utils.tensorboard import SummaryWriter
from monai.visualize import blend_images, matshow3d, plot_2d_or_3d_image
from datetime import datetime
from utils import plot_image

if __name__ == '__main__':
    # C,H,W of the image
    img_dimensions = [60, 506, 506]
    img_filepath = 'dataset/data/2_2_2_downsampled'
    training_data = hdf5.Hdf5Dataset(img_filepath, (16, 506, 506), True)
    train_dataloader = DataLoader(training_data, batch_size=4, shuffle=True, num_workers=2)
    val = training_data.__getitem__(0)
    print(val[0].shape)
    print(val[1].shape)

    # sw = SummaryWriter()
    # date = datetime.now().strftime("%d_%m_%I_%M_%S_%p")
    # plt.figure(f'Segmentation_{date}', (18, 6))
    # # plt.figure("check", (18, 6))
    # plt.subplot(1, 3, 1)
    # plt.title('Image 1')
    # plt.imshow(val[0][12, :, :], cmap="gray")
    # plt.subplot(1, 3, 2)
    # plt.title(f'Label 1')
    # plt.imshow(val[1][12, :, :], cmap="gray")
    # plt.savefig(f'Segmentation_{date}')
    # # print(val[0])
    # # plot_2d_or_3d_image(data=val[0], step=0, writer=sw, frame_dim=-1, tag='image')
    # # plot_2d_or_3d_image(data=val[1], step=0, writer=sw, frame_dim=-1, tag='mask')
