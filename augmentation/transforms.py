import numpy as np
import torch
from scipy.ndimage import zoom


def resize_image(reqd_dim, image, label):
    resize_values = (reqd_dim[0] / image.shape[0],
                     reqd_dim[1] / image.shape[1],
                     reqd_dim[2] / image.shape[2])
    img = torch.from_numpy(zoom(image, resize_values, mode='nearest').astype(np.float32))
    lbl = np.asarray(zoom(label, resize_values, order=0, mode='nearest'))
    return img, lbl


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
