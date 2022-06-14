import argparse
import string

import torch
import torch.nn.functional as F

import config
from unet.unet import UNET
from dataset import hdf5

logger = config.get_logger()
import config
from unet.unet import UNET


def predict(net, image, input_dim, device):
    net.eval()

    image = hdf5.Hdf5Dataset(image, input_dim, contains_mask=False).unsqueeze(0)
    image = image.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        prediction = net(image)
        probabilties = F.softmax(prediction)
        mask = probabilties.cpu().squeeze()

    return mask.numpy()


def get_param_arguments():
    parser = argparse.ArgumentParser(description='Unet parammeters')
    parser.add_argument('--input_path', '-i_path', metavar='IPATH', nargs='+',
                        help='Path of the image to be predicted', required=True)
    parser.add_argument('--viz', '-v', action='store_true', help='Help tp visualise the images')
    parser.add_argument('--model', '-m', default='MODEL.pth', metavar='FILE',
                        help='File in which the model is stored')


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
    net.load_state_dict(torch.load(parameter_arguments.model, map_location=device))

    logger.info('Model loaded')

    input_image = parameter_arguments.image_path;
    predict(net, input_image, config.input_dimension, device)

    net = UNET(config.n_channels, config.n_classes)
    input_image = parameter_arguments.image_path;
    predict(net, input_image, )
