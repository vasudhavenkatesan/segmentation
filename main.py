from dataset import hdf5

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    fileRead = hdf5.Hdf5Dataset('dataset/data/2_2_2_downsampled', 'train', False, 2)
    # fileRead.create_data_loader()


