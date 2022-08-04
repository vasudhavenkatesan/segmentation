import argparse
import os.path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.tensorboard import SummaryWriter
from unet.unet import UNET
from dataset import hdf5
from utils import one_hot_encoding, plot_image
import config
from eval.DiceLoss import DiceLoss

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
                save_checkpoint: bool = True,
                load_checkpoint: bool = False):
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
    # class_weights = dataset.compute_class_weights(dataset)
    class_weights = [1.0, 1000.0]
    c_weights = torch.tensor(class_weights, dtype=torch.float)
    c_weights = c_weights.to(device=device, dtype=torch.float32)

    # specify loss functions, optimizers
    criterion = nn.CrossEntropyLoss(weight=c_weights, ignore_index=2)
    optimizer = Adam(net.parameters(), lr=learning_rate)
    # scheduler = MultiStepLR(optimizer, milestones=[50, 100, 150], gamma=0.1)

    if load_checkpoint:
        # load model if it exists
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path)
            net.load_state_dict(checkpoint)

    for epoch in range(1, epochs + 1):
        logger.info('Epoch {}/{}'.format(epoch, epochs))
        logger.info('-' * 15)
        print('Epoch {}/{}'.format(epoch, epochs))
        print('-' * 10)

        net.to(device)
        net.train()
        running_loss = 0
        i = 0
        # training
        for batch in train_dataloader:
            image = batch[0]
            image = image.permute(1, 0, 2, 3)
            true_mask = batch[1]

            image = image.to(device=device, dtype=torch.float32)
            true_mask = true_mask.to(device=device, dtype=torch.int64)
            true_mask = one_hot_encoding(true_mask, config.n_classes)
            true_mask = true_mask.type(torch.long)

            optimizer.zero_grad()

        #     pred = net(image)
        #     # pred = pred[:, -1, :, :]
        #     loss = criterion(pred, true_mask)
        #     i += 1
        #     # Backpropagation
        #     loss.backward()
        #     optimizer.step()
        #     # scheduler.step()
        #
        #     running_loss += loss.item()
        #
        #     if epoch == epochs:
        #         plot_image(batch[0], batch[1], pred, 'train', i)
        #
        # print(f'Epoch : {epoch}, running loss : {running_loss}, loss: {(running_loss / i):.4f}')
        # logger.info(f'Epoch : {epoch}, running loss : {running_loss}, loss: {(running_loss / i)}')
        # writer.add_scalar("Loss/train", (running_loss / i), epoch)
        #
        # if i == 1:
        #     break;
        # # validation
        # logger.info('Validation step')
        # net.eval()
        #
        # for batch in val_dataloader:
        #     image = batch[0]
        #     image = image.permute(1, 0, 2, 3)
        #     mask = batch[1]
        #     image = image.to(device=device, dtype=torch.float32)
        #     true_mask = mask.to(device=device, dtype=torch.int64)
        #
        #     true_mask = one_hot_encoding(true_mask, config.n_classes)
        #     true_mask = true_mask.type(torch.long)
        #
        #     val_loss = 0
        #     with torch.no_grad():
        #         # predict the mask
        #         pred = net(image)
        #         # pred = pred[:, -1, :, :]
        #         loss = criterion(pred, true_mask)
        #
        #         val_loss += loss
        #
        #         if epoch == epochs:
        #             plot_image(batch[0], batch[1], pred, 'val', 0)
        #
        # print(f'Validation loss : {val_loss:.4f}')
        # logger.info(f'Validation loss : {val_loss}')
        # writer.add_scalar("Validation Loss", val_loss, epoch)

    torch.cuda.empty_cache()
    writer.flush()

    # save checkpoint
    if save_checkpoint:
        if os.path.exists(checkpoint_path):  # checking if there is a file with this name
            os.remove(checkpoint_path)  # deleting the file
        checkpoint = net.state_dict()
        torch.save(checkpoint, checkpoint_path)
        logger.info(f'Checkpoint {epoch} saved!')
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
