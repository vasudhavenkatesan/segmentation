import torch.nn.functional as F
from datetime import datetime
import matplotlib.pyplot as plt
import torch
import config
import numpy as np

logger = config.get_logger()


def one_hot_encoding(input, n_classes):
    target = F.one_hot(input, n_classes)
    # Reorder the target with depth as batch size followed by num of classes, height, width
    # target = target.permute(0, 1, 5, 2, 3, 4)
    target = target[-1, :, :, :, :, -1]
    return target


def plot_image(image, gt, pred, type='val', i=0):
    date = datetime.now().strftime("%d_%m_%I_%M_%S_%p")
    filename = 'Segm_' + type + '_' + str(i) + '_' + date

    logger.info(f'Plotting image -  {filename} saved!')
    plt.figure(filename, (18, 6))
    plt.subplot(1, 3, 1)
    plt.title(f'Image')
    plt.imshow(image[-1, 12, :, :], cmap='gray')
    plt.subplot(1, 3, 2)
    plt.title(f'GT')
    gt[gt == 2] = 0.5
    plt.imshow(gt[-1, 12, :, :], cmap='gray')
    predic = torch.from_numpy(pred.detach().cpu().numpy())
    pred_for_plot = predic.argmax(dim=1)
    pred_for_plot[pred_for_plot == 2] = 0.5
    plt.subplot(1, 3, 3)
    plt.title('Predicted Mask')
    plt.imshow(pred_for_plot[12, :, :], cmap='gray')
    plt.savefig(filename)


def test():
    input = 3 * torch.rand([1, 1, 16, 256, 256])
    input = input.to(device=config.device, dtype=torch.int64)
    one_hot_encoding(input, 3)
