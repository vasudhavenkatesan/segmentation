import numpy as np
import torch


# Add padding to the 3D data
def padding(data_to_be_padded, dimensions, padding_value):
    side_1, side_2 = calc_size_for_padding(np.array(dimensions), np.array(data_to_be_padded.shape))
    out = np.pad(data_to_be_padded, (
        (side_1[0], side_2[0]),
        (side_1[1], side_2[1]),
        (side_1[2], side_2[2])), "constant")
    print(f'After padding {out.shape}\n')
    return out


def calc_size_for_padding(reqd_dim, data_dim):
    l_size = (reqd_dim - data_dim) // 2
    r_size = (reqd_dim - (data_dim + l_size))
    return l_size, r_size


def resize_image(reqd_dim, input):
    return input[0:reqd_dim[0], 0:reqd_dim[1], 0:reqd_dim[2]]
