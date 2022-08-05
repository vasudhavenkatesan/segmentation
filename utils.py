import torch.nn.functional as F
from datetime import datetime
import matplotlib.pyplot as plt
import config
import sys
import numpy as np

np.set_printoptions(threshold=sys.maxsize)

logger = config.get_logger()


def one_hot_encoding(input, n_classes):
    target = F.one_hot(input, n_classes)
    target = target.permute(0, 1, 4, 2, 3)
    target = target[-1, :, -1, :, :]
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
    # normalise gt and prediction for a grayscale image
    gt = normalise_values(gt)
    plt.imshow(gt[-1, 12, :, :], cmap='gray')
    pred = normalise_values(pred)
    pred = pred.argmax(axis=1)
    plt.subplot(1, 3, 3)
    plt.title('Predicted Mask')
    plt.imshow(pred[12, :, :], cmap='gray')
    plt.savefig(filename)
    print_tensor_values(gt, pred)


def normalise_values(unnormalised_input):
    np_input = unnormalised_input.cpu().detach().numpy()
    return np_input / 2


def print_tensor_values(gt, pred):
    a = np.ravel(gt[-1, 12, :, :])
    b = np.ravel(pred[12, :, :])
    print(f'Size : {a.size} , Difference : {np.count_nonzero(a - b)}')
