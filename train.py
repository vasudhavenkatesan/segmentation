import argparse
import os.path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from unet.unet import UNET
from dataset import hdf5
from utils import plot_image, plot_3d_image
import config
import tqdm

# Logger
logger = config.get_logger()

data_file_path = config.dataset_path

checkpoint_path = config.checkpoint_dir

writer = SummaryWriter()


def training_fn(net,
                device,
                input_dim,
                epochs: int = 1,
                batch_size: int = 1,
                learning_rate: float = 1e-3,
                valiation_percent=0.1,
                load_checkpoint: bool = True):
    # create dataset
    dataset = hdf5.Hdf5Dataset(data_file_path, image_dim=input_dim, contains_mask=True)

    # create training and validation dataset
    n_dataset = dataset.__len__()
    n_val = round(n_dataset * valiation_percent)
    n_train = n_dataset - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val])

    # create dataloaders
    train_dataloader = DataLoader(train_set, shuffle=True, batch_size=batch_size, num_workers=1, pin_memory=True)
    val_dataloader = DataLoader(val_set, shuffle=False, batch_size=batch_size, num_workers=1, pin_memory=True)

    # criterion = DiceLoss(ignore_index=[2], reduction='mean')

    # specify loss functions, optimizers
    criterion = nn.CrossEntropyLoss(ignore_index=255, reduction='none')
    optimizer = Adam(net.parameters(), lr=learning_rate)
    # scheduler = MultiStepLR(optimizer, milestones=[50, 100, 150], gamma=0.1)

    if load_checkpoint:
        # load model if it exists
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path)
            net.load_state_dict(checkpoint)

    val_loss = 0
    best_validation_loss = 10.0

    for epoch in tqdm.tqdm(range(epochs)):
        logger.info('Epoch {}/{}'.format(epoch + 1, epochs))
        logger.info('-' * 15)
        print('Epoch {}/{}'.format(epoch + 1, epochs))
        print('-' * 10)

        net.to(device)
        net.train()

        i = 0
        running_loss = 0.0
        # training
        for batch in train_dataloader:

            image = torch.unsqueeze(batch[0], dim=0)
            image = image.to(device=device, dtype=torch.float32)
            gt = batch[1].to(device=device, dtype=torch.long)

            optimizer.zero_grad()

            pred = net(image)
            loss = criterion(pred, gt)
            i += 1
            # Backpropagation
            loss.mean().backward()
            optimizer.step()
            # scheduler.step()

            running_loss += loss.mean()

            if epoch == (epochs - 1):
                plot_image(batch[0], batch[1], pred, 'train', i)

            plot_3d_image(batch[0], batch[1], pred, loss, step=epoch, writer=writer)

        writer.add_scalar("Loss/train", (running_loss / i), epoch)
        print(f'Epoch : {epoch}, running loss : {running_loss}, loss: {(running_loss / i):.4f}')
        logger.info(f'Epoch : {epoch}, running loss : {running_loss}, loss: {(running_loss / i)}')

        # validation
        logger.info('Validation step')
        net.eval()
        val_loss = 0.0
        for batch in val_dataloader:
            image = torch.unsqueeze(batch[0], dim=0)
            image = image.to(device=device, dtype=torch.float32)
            gt = batch[1].to(device=device, dtype=torch.long)

            with torch.no_grad():
                # predict the mask
                pred = net(image)
                loss = criterion(pred, gt)

                val_loss += loss.mean()

                if epoch == (epochs - 1):
                    plot_image(batch[0], batch[1], pred, 'val', 0)

            plot_3d_image(batch[0], batch[1], pred, loss, step=epoch, writer=writer)

        print(f'Validation loss : {val_loss:.4f}')
        logger.info(f'Validation loss : {val_loss}')
        writer.add_scalar("Validation Loss", val_loss, epoch)

    torch.cuda.empty_cache()
    writer.flush()

    # save checkpoint
    if val_loss < best_validation_loss:
        if os.path.exists(checkpoint_path):  # checking if there is a file with this name
            os.remove(checkpoint_path)  # deleting the file
        checkpoint = net.state_dict()
        torch.save(checkpoint, checkpoint_path)
        logger.info(f'Best checkpoint at epoch - {epoch} saved!')
        logger.info(f'Best validation loss - {best_validation_loss}')
    writer.close()


def get_param_arguments():
    parser = argparse.ArgumentParser(description='Unet parammeters')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=1, help='Number of epochs(training cycles)')
    parser.add_argument('--batch_size', '-b', type=int, metavar='B', default=1,
                        help='Batch size - Number of datasets in each training batch')
    parser.add_argument('--learning_rate', '-lr', metavar='LR', type=float, default=3e-5,
                        help='Learning rate for optimizer')
    parser.add_argument('--validation_perc', '-val', metavar='VALPERC', type=float, default=0.1,
                        help='Percent of validation set')
    parser.add_argument('--n_channels', '-n_chan', metavar='NCHANNEL', type=int, default=1,
                        help='Number of channels in the image')
    parser.add_argument('--n_classes', '-n_class', metavar='NCLASS', type=int, default=2,
                        help='Number of output classes')
    parser.add_argument('--load_cp', '-l_cp', metavar='LOADCHECKPOINT', type=bool, default=False,
                        help='Load model from check point ')
    return parser.parse_args()


if __name__ == '__main__':
    parameter_arguments = get_param_arguments()
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    logger.info(f'Using device - {device}')
    model = UNET(parameter_arguments.n_channels, parameter_arguments.n_classes)
    logger.info(f'UNET model initialised')

    training_fn(net=model, device=device, epochs=parameter_arguments.epochs, batch_size=parameter_arguments.batch_size,
                learning_rate=parameter_arguments.learning_rate, valiation_percent=parameter_arguments.validation_perc,
                input_dim=config.image_dim, load_checkpoint=parameter_arguments.load_cp)

    logger.info('Process completed')
