import argparse

import torch
import torch.nn.functional as F

import config
from unet.unet import UNET
from dataset import hdf5
import matplotlib.pyplot as plt

logger = config.get_logger()


def predict(net, image, input_dim, device):
    net.eval()
    plt.figure("Segmentation", (18, 6))
    img = hdf5.Hdf5Dataset(image, input_dim)
    n_preds = img.__len__()
    pred_masks = []
    for i in range(0, n_preds):
        image = img.__getitem__(i)[0].to(device=device, dtype=torch.float32)
        mask = img.__getitem__(i)[1].to(device=device, dtype=torch.float32)
        plt.subplot(1, 3, 1)
        plt.title(f'Image')
        plt.imshow(image[10, :, :], cmap="gray")
        plt.subplot(1, 3, 2)
        plt.title(f'Mask')
        plt.imshow(mask[10, :, :], cmap="gray")
        image = image[:, None, :, :]
        # 0, 2, 3)
        with torch.no_grad():
            prediction = net(image)
            print(f'After prediction {prediction.shape}')
            pred_for_plot = prediction.detach().cpu().numpy()
            predic = torch.from_numpy(pred_for_plot)
            pred_for_plot = torch.unsqueeze(predic.argmax(dim=1), 1)
            print('Plotting')
            plt.subplot(1, 3, 3)
            plt.title('Predicted Mask')
            plt.imshow(pred_for_plot[10, -1, :, :], cmap="gray")
    print('saving plot ---------')
    plt.savefig('Segmentation')
    plt.show()
    # print(f'Mean IoU -  {img.masks[i].shape}')
    # pred_masks.append(mask.numpy())
    return pred_masks


def get_param_arguments():
    parser = argparse.ArgumentParser(description='Unet parammeters')
    parser.add_argument('--viz', '-v', action='store_true', help='Help tp visualise the images')
    parser.add_argument('--model', '-m', default='Model1.pth', metavar='FILE',
                        help='File in which the model is stored')
    parser.add_argument('--mo', '-mo', default='Model1.pth', metavar='FILE',
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
    predict(net, input_image, config.image_dim, device)

# net = UNET(config.n_channels, config.n_classes)
# input_image = parameter_arguments.image_path;
# predict(net, input_image)
