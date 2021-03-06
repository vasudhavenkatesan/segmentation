import torch
import os
import logging
from datetime import datetime

dataset_path = "dataset/data/2_2_2_downsampled/train"

device = "cuda" if torch.cuda.is_available() else "cpu"

image_dim = (16, 256, 256)

n_channels = 1

n_classes = 3

ignore_label = [2]

checkpoint_dir = os.path.join("checkpoints", "Model1.pth")


def get_logger():
    logging.basicConfig(handlers=[
        logging.FileHandler(filename=datetime.now().strftime('logs/training_log_%H_%M_%d_%m_%Y.log'),
                            encoding='utf-8')],
        format='%(asctime)s %(message)s',
        datefmt='%m/%d/%Y %I:%M:%S %p')

    # Creating an object
    logger = logging.getLogger()

    # Setting the threshold of logger to DEBUG
    logger.setLevel(logging.DEBUG)
    return logger
