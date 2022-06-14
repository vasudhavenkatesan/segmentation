import argparse
import string
from pathlib import Path

import torch
import torch.nn.functional as F

import config
from unet.unet import UNET
from dataset import hdf5
from evaluate import mIoU

logger = config.get_logger()


def predict(net, image, input_dim, device):
    net.eval()

    img = hdf5.Hdf5Dataset(image, input_dim, mask_with_channel=True)
    n_preds = img.images.__len__()
    pred_masks = []
    for i in range(0, n_preds):
        image = torch.unsqueeze(img.images[i], 0).to(device=device, dtype=torch.float32)
        with torch.no_grad():
            prediction = net(image)
            probabilties = F.softmax(prediction)
            mask = probabilties.cpu().squeeze()

        print(f'Mean IoU -  {img.masks[i].shape}')
        pred_masks.append(mask.numpy())
    return pred_masks


def get_param_arguments():
    parser = argparse.ArgumentParser(description='Unet parammeters')
    parser.add_argument('--viz', '-v', action='store_true', help='Help tp visualise the images')
    parser.add_argument('--model', '-m', default='Model.pth', metavar='FILE',
                        help='File in which the model is stored')
    parser.add_argument('--mo', '-mo', default='Model.pth', metavar='FILE',
                        help='File in which the model is stored')
    parser.add_argument('--input', '-ip', default='dataset/data/2_2_2_downsampled/test', metavar='FILE',
                        help='Input File')
    return parser.parse_args()


if __name__ == '__main__':
    parameter_arguments = get_param_arguments()
    device = config.device
    logger.info(f'Using device - {device}')
    net = UNET(config.n_channels, config.n_classes)
    net.load_state_dict(torch.load(parameter_arguments.model, map_location=device))
    print('Model loaded')
    logger.info('Model loaded')
    print(f'Model path {parameter_arguments.model}')
    input_image = parameter_arguments.input
    print(input_image)
    output = predict(net, input_image, config.input_dimension, device)

    # net = UNET(config.n_channels, config.n_classes)
    # input_image = parameter_arguments.image_path;
    # predict(net, input_image)
