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


def dice_loss(true, logits, eps=1e-7):
    """Computes the Sørensen–Dice loss.
    Note that PyTorch optimizers minimize a loss. In this
    case, we would like to maximize the dice loss so we
    return the negated dice loss.
    Args:
        true: a tensor of shape [B, 1, H, W].
        logits: a tensor of shape [B, C, H, W]. Corresponds to
            the raw output or logits of the model.
        eps: added to the denominator for numerical stability.
    Returns:
        dice_loss: the Sørensen–Dice loss.
    """
    num_classes = logits.shape[1]
    if num_classes == 1:
        true_1_hot = torch.eye(num_classes + 1)[true.squeeze(1)]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        true_1_hot_f = true_1_hot[:, 0:1, :, :]
        true_1_hot_s = true_1_hot[:, 1:2, :, :]
        true_1_hot = torch.cat([true_1_hot_s, true_1_hot_f], dim=1)
        pos_prob = torch.sigmoid(logits)
        neg_prob = 1 - pos_prob
        probas = torch.cat([pos_prob, neg_prob], dim=1)
    else:
        true_1_hot = torch.eye(num_classes)[true.squeeze(1)]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        probas = F.softmax(logits, dim=1)
    true_1_hot = true_1_hot.type(logits.type())
    dims = (0,) + tuple(range(2, true.ndimension()))
    intersection = torch.sum(probas * true_1_hot, dims)
    cardinality = torch.sum(probas + true_1_hot, dims)
    dice_loss = (2. * intersection / (cardinality + eps)).mean()
    return (1 - dice_loss)


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
    img_filepath = 'dataset/data/2_2_2_downsampled/test'
    training_data = hdf5.Hdf5Dataset(filepath=img_filepath, image_dim=[60, 506, 506], contains_mask=True)
    visualise(training_data.images[0], training_data.masks[0])
