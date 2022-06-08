from torch.utils.data import DataLoader

from dataset import hdf5
from augmentation import transforms

if __name__ == '__main__':
    # C,H,W of the image
    img_dimensions = [96, 512, 512]
    img_filepath = 'dataset/data/2_2_2_downsampled'
    training_data = hdf5.Hdf5Dataset(img_filepath, True)
    train_dataloader = DataLoader(training_data, batch_size=4, shuffle=True, num_workers=2)

    # test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)
