import torch.nn.functional as F
from datetime import datetime
import matplotlib.pyplot as plt
import torch
import config
import numpy as np
from sklearn import preprocessing

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
    pred = (torch.from_numpy(pred.detach().cpu().numpy())).argmax(dim=1)
    pred = normalise_values(pred)
    plt.subplot(1, 3, 3)
    plt.title('Predicted Mask')
    plt.imshow(pred[12, :, :], cmap='gray')
    plt.savefig(filename)
    print_tensor_values(gt, pred)


def normalise_values(unnormalised_input):
    min_max_scaler = preprocessing.MinMaxScaler()
    return min_max_scaler.fit_transform(unnormalised_input)


def print_tensor_values(gt, pred):
    a = torch.unsqueeze(gt[-1, 12, :, :], 0)
    predic = torch.from_numpy(pred.detach().cpu().numpy())
    b = torch.unsqueeze(predic[12, :, :], 0)
    print(a - b)
