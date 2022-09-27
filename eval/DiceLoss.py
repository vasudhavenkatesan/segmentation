import torch
from config import device


def flatten(tensor):
    """Flattens a given tensor such that the channel axis is first.
    The shapes are transformed as follows:
       (N, C, D, H, W) -> (C, N * D * H * W)
    """
    # number of channels
    C = tensor.size(1)
    # new axis order
    axis_order = (1, 0) + tuple(range(2, tensor.dim()))
    # Transpose: (N, C, D, H, W) -> (C, N, D, H, W)
    transposed = tensor.permute(axis_order)
    # Flatten: (C, N, D, H, W) -> (C, N * D * H * W)
    return transposed.contiguous().view(C, -1)


def dice(input, target, weight=None, epsilon=1e-6):
    input = input.to(device=device, dtype=torch.long)
    target = torch.sigmoid(target[:, 1, :])
    assert input.size() == target.size(), "'input' and 'target' must have the same shape"

    input = flatten(input)
    target = flatten(target)
    target = target.float()

    if input.size(0) == 1:
        input = torch.cat((input, 1 - input), dim=0)
        target = torch.cat((target, 1 - target), dim=0)

    # GDL weighting: the contribution of each label is corrected by the inverse of its volume
    w_l = target.sum(-1)
    w_l = 1 / (w_l * w_l).clamp(min=epsilon)

    intersect = (input * target).sum(-1)
    intersect = intersect * w_l

    denominator = (input + target).sum(-1)
    denominator = (denominator * w_l).clamp(epsilon)

    return 2 * (intersect.sum() / denominator.sum())
