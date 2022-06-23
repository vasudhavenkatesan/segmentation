import torch.nn.functional as F


def one_hot_encoding(input, n_classes):
    target = F.one_hot(input, n_classes)
    # Reorder the target with depth as batch size followed by num of classes, height, width
    target = target.permute(0, 1, 4, 2, 3)
    target = target[-1, :, -1, :, :]
    return target
