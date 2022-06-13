import torch
import os

path = 'C:/Users/vasud/Documents/Subjects/Project/segmentation'
DATASET_PATH = os.path.join("dataset", "data", "train")

TEST_SPLIT = 0.15

device = "cuda" if torch.cuda.is_available() else "cpu"

PIN_MEMORY = True if device == "cuda" else False

n_channels = 60

n_classes = 3
