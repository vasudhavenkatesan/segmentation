import torch
import os
import logging
from datetime import datetime

path = 'C:/Users/vasud/Documents/Subjects/Project/segmentation'
DATASET_PATH = os.path.join("dataset", "data", "")

TEST_SPLIT = 0.15

device = "cuda" if torch.cuda.is_available() else "cpu"

PIN_MEMORY = True if device == "cuda" else False

n_channels = 60

n_classes = 3

input_dimension = [60,506,506]


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
