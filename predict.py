import argparse
import string

import torch
import config
from unet.unet import UNET


def predict(net, image, input_dim, device):
    net.eval()


def get_param_arguments():
    parser = argparse.ArgumentParser(description='Unet parammeters')
    parser.add_argument('--image_path', '-i_path', metavar='IPATH', type=string,
                        help='Path of the image to be predicted')
    return parser.parse_args()


if __name__ == '__main__':
    parameter_arguments = get_param_arguments()
    device = config.device
    logger.info(f'Using device - {device}')
    net = UNET(config.n_channels, config.n_classes)
    input_image = parameter_arguments.image_path;
    predict(net, input_image, )
