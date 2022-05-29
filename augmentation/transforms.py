import numpy as np
import torch
import math


# Add padding to the 3D data
def padding(data_to_be_padded, dimensions, padding_value):
    x = torch.tensor(np.array([[[1., 2., 13], [3., 4., 14], [19, 20, 21]],
                               [[5., 6., 15], [7., 8., 16], [22, 23, 24]],
                               [[9., 10., 17], [11., 12., 18], [25, 26, 27]]]))
    dimension = [4, 4, 4]
    c = math.ceil(((dimension[0] - x.shape[0]) / 2))
    h = math.ceil(((dimension[1] - x.shape[1]) / 2))
    w = math.ceil(((dimension[2] - x.shape[2]) / 2))
    print(f'shape - {x.shape} ')
    print(f'c = {c}, h = {h}, w = {w}')
    out = np.pad(x, ((c, c), (h, h), (w, w)), 'constant')
    print(f'After padding {out}')
    return data_to_be_padded
