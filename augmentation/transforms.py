import numpy as np
import torch
from scipy.ndimage import zoom


def resize_image(reqd_dim, image, label):
    resize_values = (reqd_dim[0] // image.shape[0],
                     reqd_dim[1] // image.shape[1],
                     reqd_dim[2] // image.shape[2])
    img = torch.from_numpy(zoom(image, resize_values, mode='nearest').astype(np.float32))
    lbl = np.asarray(zoom(label, resize_values, order=0, mode='nearest'))
    return img, lbl


# Returns the indices of random crop
class RandomCrop3D:
    def __init__(self, crop_sz):
        self.crop_sz = crop_sz

    def __call__(self, original_img_sz):
        self.img_sz = original_img_sz
        slice_dhw = [self._get_slice(i, k) for i, k in zip(self.img_sz, self.crop_sz)]
        return slice_dhw

    @staticmethod
    def _get_slice(sz, crop_sz):
        lower_bound = torch.randint(sz - crop_sz, (1,)).item()
        return lower_bound, lower_bound + crop_sz


def test():
    crop_dim = (67, 127, 127)
    img = (68, 128, 128)
    rand_crop = RandomCrop3D(crop_dim)
    print(rand_crop(img))
