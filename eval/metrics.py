import numpy as np
from torchmetrics import Dice, Accuracy, JaccardIndex
from torchmetrics.functional import dice_score

from config import device


def dice(test=None, reference=None):
    """2TP / (2TP + FP + FN)"""

    jaccard = JaccardIndex(num_classes=2).to(device)
    j = jaccard(test, reference)
    return (2 * j) / (j + 1)
    # return state.metrics['dice']
    # return dice(test, reference)


def accuracy(test=None, reference=None):
    """(TP + TN) / (TP + FP + FN + TN)"""

    # if confusion_matrix is None:
    #     confusion_matrix = ConfusionMatrix(test, reference)
    #
    # tp, fp, tn, fn = confusion_matrix.get_matrix()
    #
    # return float((tp + tn) / (tp + fp + tn + fn))

    accuracy = Accuracy(mdmc_average='global').to(device)
    return accuracy(test, reference)


def test_():
    y_true = [0, 0, 1, 0, 0]
    y_pred = [0, 1, 1, 0, 1]
    print(f'Accuracy - {accuracy(np.array(y_pred), np.array(y_true))}')
    print(f'Dice score - {dice(np.array(y_pred), np.array(y_true))}')
