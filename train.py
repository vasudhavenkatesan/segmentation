import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

import logging
import sys

from unet.unet import UNET
from dataset import hdf5

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

data_file_path = 'dataset/data/2_2_2_downsampled'


def training_fn(net,
                device,
                epochs: int = 1,
                batch_size: int = 1,
                learning_rate: float = 1e-5,
                valiation_percent=0.1,
                input_dim: int = [60, 506, 506],
                save_checkpoint: bool = True):
    # create dataset
    dataset = hdf5.Hdf5Dataset(data_file_path, input_dim)

    # create training and validation dataset
    n_dataset = dataset.data_size
    n_val = int(n_dataset * valiation_percent)
    n_train = n_dataset - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val])

    # create dataloaders
    train_dataloader = DataLoader(train_set, shuffle=True, batch_size=batch_size, num_workers=4, pin_memory=True)
    val_dataloader = DataLoader(val_set, shuffle=False, batch_size=batch_size, num_workers=2, pin_memory=True)

    # specify loss functions, optimizers
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)

    for epoch in range(1, epochs + 1):
        net.train()
        loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
            # training
            for batch in enumerate(train_dataloader):
                train_features, train_labels = next(iter(train_dataloader))
                image = train_features[0]
                true_mask = train_labels[0]

                logging.info(f'Image size {image.shape}')

                image = image.to(device=device, dtype=torch.float64)
                true_mask = true_mask.to(device=device, dtype=torch.float32)

                pred = model(image)
                loss = loss_fn(pred, true_mask)

                # Backpropagation
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

                loss, current = loss.item(), (batch * n_train)
                logging.info(f"loss: {loss:>7f}  [{current:>5d}/{n_train:>5d}]")

        # validation
        num_batches = len(val_dataloader)
        test_loss, correct = 0, 0

        with torch.no_grad():
            for batch in enumerate(train_dataloader):
                val_image, val_label = next(iter(train_dataloader))
                image = val_image[0]
                true_mask = val_label[0]
                pred = model(image)
                test_loss += loss_fn(pred, true_mask).item()
                correct += (pred.argmax(1) == true_mask).type(torch.float).sum().item()

        test_loss /= num_batches
        correct /= n_val
        print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


def get_param_arguments():
    parser = argparse.ArgumentParser(description='Unet parammeters')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=1, help='Number of epochs(training cycles)')
    parser.add_argument('--batch_size', '-b', type=int, metavar='B', default=1,
                        help='Batch size - Number of datasets in each training batch')
    parser.add_argument('--learning_rate', '-lr', metavar='LR', type=float, default=1e-5,
                        help='Learning rate for optimizer')
    parser.add_argument('--validation_perc', '-val', metavar='VALPERC', type=float, default=0.1,
                        help='Percent of validation set')
    parser.add_argument('--n_channels', '-n_chan', metavar='NCHANNEL', type=int, default=60,
                        help='Number of channels in the image')
    parser.add_argument('--n_classes', '-n_class', metavar='NCLASS', type=int, default=3,
                        help='Number of output classes')
    return parser.parse_args()


if __name__ == '__main__':
    parameter_arguments = get_param_arguments()
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    logging.info(f'Using device - {device}')
    model = UNET(parameter_arguments.n_channels, parameter_arguments.n_classes)
    logging.info(f'UNET model initialised')

    training_fn(net=model, device=device, batch_size=parameter_arguments.batch_size,
                learning_rate=parameter_arguments.learning_rate, valiation_percent=parameter_arguments.validation_perc)

    logging.info('Training completed')
