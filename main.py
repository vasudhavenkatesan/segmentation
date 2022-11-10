import os.path
import argparse

import torch

from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

import config
from dataset import hdf5
from augmentation import transforms
from torch.utils.tensorboard import SummaryWriter
from unet.unet import UNET
from unetr.unetr import UNETR
from train import training_fn
from datetime import datetime
from utils import plot_image
import numpy as np


def get_param_arguments():
    parser = argparse.ArgumentParser(description="Unet parammeters")
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs(training cycles)")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size - Number of datasets in each training batch")
    parser.add_argument("--learning_rate", type=float, default=3e-5,
                        help="Learning rate for optimizer")
    parser.add_argument("--validation_perc", type=float, default=0.1,
                        help="Percent of validation set")
    parser.add_argument("--n_channels", type=int, default=1,
                        help="Number of channels in the image")
    parser.add_argument("--n_classes", type=int, default=2,
                        help="Number of output classes")
    parser.add_argument("--model_name", default="unetr", type=str,
                        help="model name used for training")
    parser.add_argument("--image_sizex", default=256, type=int,
                        help="size of image in x axis")
    parser.add_argument("--image_sizey", default=256, type=int,
                        help="size of image in y axis")
    parser.add_argument("--image_sizez", default=64, type=int,
                        help="size of image in z axis")
    parser.add_argument("--mask_type", default="h5", type=str,
                        help="Type of mask - h5 or nrrd")
    parser.add_argument("--load_cp", default=False, type=bool,
                        help="Load model from check point ")
    parser.add_argument("--save_cp", default=True, type=bool,
                        help="Save model at check point location")
    return parser.parse_args()


def main():
    param_arg = get_param_arguments()
    device = config.device
    logger = config.get_logger()
    in_channels = param_arg.n_channels
    out_channels = param_arg.n_classes
    img_size = [param_arg.image_sizez, param_arg.image_sizex, param_arg.image_sizey]
    logger.debug(f'Using device - {device}')
    if (param_arg.model_name == "unetr"):
        model = UNETR(in_channels, out_channels, img_size)
        logger.debug(f'UNETR model initialised')
    else:
        model = UNET(param_arg.n_channels, param_arg.n_classes)
        logger.debug(f'UNET model initialised')
    try:
        training_fn(model=model, device=device, epochs=param_arg.epochs,
                    batch_size=param_arg.batch_size,
                    learning_rate=param_arg.learning_rate, valiation_percent=param_arg.validation_perc,
                    input_dim=img_size, load_checkpoint=param_arg.load_cp, model_name=param_arg.model_name,
                    mask_type=param_arg.mask_type)
    except:
        logger.exception('Got exception on main handler')
        raise
    logger.debug('Process completed')


if __name__ == '__main__':
    main()
