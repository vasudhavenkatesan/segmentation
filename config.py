import torch
import os

DATASET_PATH = os.path.join("dataset", "data", "train")

TEST_SPLIT = 0.15

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

PIN_MEMORY = True if DEVICE == "cuda" else False

N_CHANNELS = 95

N_CLASSES = 3

N_LEVELS = 3
