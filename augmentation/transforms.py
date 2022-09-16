import numpy as np
import torch
from scipy.ndimage import zoom


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
    resize_values = (reqd_dim[0] / input.shape[0],
                     reqd_dim[1] / input.shape[1],
                     reqd_dim[2] / input.shape[2])
    return zoom(input, resize_values, mode='nearest')


class RandomCrop3D:
    def __init__(self, crop_sz):
        self.crop_sz = tuple(crop_sz)

    def __call__(self, image, mask):
        self.img_sz = image.shape
        slice_dhw = [self._get_slice(i, k) for i, k in zip(self.img_sz, self.crop_sz)]
        # slice_dhw = [[0, 16], [0, 256], [0, 256]]
        img = self._crop(image, *slice_dhw)
        msk = self._crop(mask, *slice_dhw)
        return img, msk

    @staticmethod
    def _get_slice(sz, crop_sz):
        try:
            lower_bound = torch.randint(sz - crop_sz, (1,)).item()
            return lower_bound, lower_bound + crop_sz
        except:
            return None, None

    @staticmethod
    def _crop(x, slice_d, slice_w, slice_h):
        return x[slice_d[0]:slice_d[1], slice_h[0]:slice_h[1], slice_w[0]:slice_w[1]]


def test():
    crop_dim = [64, 256, 256]
    image_dim = [90, 512, 512]
    rand_crop = RandomCrop3D(image_dim, crop_dim)
    print(rand_crop)
