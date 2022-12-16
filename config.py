import torch
import os
import logging
from datetime import datetime

# smaller images(h5) -"dataset/data/2_2_2_downsampled/train"
# larger images(nrrd) - "dataset/data/nrrd_data/train"

dataset_path = "/misc/lmbssd/venkatev/data/data/train"

output_path = os.path.join("output")

device = "cuda" if torch.cuda.is_available() else "cpu"

max_image_dim = [80, 512, 512]

n_channels = 1

n_classes = 2

ignore_label = [2]

checkpoint_dir = os.path.join("checkpoints")


def get_logger():
    logging.basicConfig(handlers=[
        logging.FileHandler(filename=datetime.now().strftime('logs/training_log_%H_%M_%d_%m_%Y.log'),
                            encoding='utf-8')], format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
    # Creating an object
    logger = logging.getLogger()
    # Setting the threshold of logger to DEBUG
    logger.setLevel(logging.DEBUG)
    return logger
