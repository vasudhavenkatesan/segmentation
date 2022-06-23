import numpy as np

import os
import torch
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, hamming_loss, multilabel_confusion_matrix
import matplotlib.pyplot as plt
from dataset import hdf5


def mIoU(y_pred, y_true):
    y_pred = y_pred.data.cpu().numpy()
    y_true = y_true.cpu().numpy()
    y_pred = y_pred.flatten()
    y_true = y_true.flatten()
    # Accuracy Score
    val = accuracy_score(y_true, y_pred, normalize=True, sample_weight=None)
    ((y_pred == y_true).all(axis=1).sum() / y_pred.shape[0])

    # Hamming Loss
    hamming_loss(y_true, y_pred)
    scores = (y_pred != y_true).sum(axis=1)
    numerator = scores.sum()
    denominator = ((scores != 0).sum() * y_true.shape[1])
    hl = (numerator / denominator)

    current = multilabel_confusion_matrix(y_true, y_pred.round(), labels=[0, 1, 2])
    intersection = np.diag(current)
    ground_truth_set = current.sum(axis=1)
    predicted_set = current.sum(axis=0)
    union = ground_truth_set + predicted_set - intersection
    IoU = intersection / union.astype(np.float32)
    # return np.mean(IoU)

    return val, hl, np.mean(IoU)


def visualise(img, mask):
    classes = mask.shape[0] if len(mask.shape) > 2 else 1
    fig, ax = plt.subplots(1, classes + 1)
    ax[0].set_title('Input image')
    ax[0].imshow(img)
    if classes > 1:
        for i in range(classes):
            ax[i + 1].set_title(f'Output mask (class {i + 1})')
            ax[i + 1].imshow(mask[1, :, :])
    else:
        ax[1].set_title(f'Output mask')
        ax[1].imshow(mask)
    plt.xticks([]), plt.yticks([])
    plt.show()


if __name__ == '__main__':
    img_filepath = '../dataset/data/2_2_2_downsampled/test'
    training_data = hdf5.Hdf5Dataset(filepath=img_filepath, image_dim=[60, 506, 506], contains_mask=True)
    visualise(training_data.images[0], training_data.masks[0])
