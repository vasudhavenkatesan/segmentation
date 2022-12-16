import os.path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import GradScaler
from torch.cuda.amp import autocast
from dataset import hdf5
from utils import plot_image, plot_3d_image
from eval.metrics import dice, accuracy
import config
import tqdm
from pytorchcheckpoint.checkpoint import CheckpointHandler

# Logger
logger = config.get_logger()

data_file_path = config.dataset_path

checkpoint_path = config.checkpoint_dir

writer = SummaryWriter()


def training_fn(model,
                device,
                input_dim,
                model_name,
                epochs: int = 1,
                batch_size: int = 1,
                learning_rate: float = 1e-3,
                valiation_percent=0.1,
                load_checkpoint: bool = False,
                save_checkpoint: bool = True,
                mask_type="h5"):
    # create dataset
    dataset = hdf5.Hdf5Dataset(data_file_path, reqd_image_dim=input_dim, contains_mask=True, mask_file_type=mask_type)

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
    optimizer = Adam(model.parameters(), lr=learning_rate)
    # scheduler = MultiStepLR(optimizer, milestones=[50, 100, 150], gamma=0.1)

    scaler = GradScaler()
    checkpoint_handler = CheckpointHandler()
    checkpoint_handler.store_var(var_name="Model type", value=model_name)
    checkpoint_handler.store_var(var_name="learning_rate", value=learning_rate)
    checkpoint_handler.store_var(var_name="batch_size", value=batch_size)
    checkpoint_handler.store_var(var_name="epochs", value=epochs)
    if load_checkpoint:
        # load model if it exists
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path)
            model.load_state_dict(checkpoint)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    best_validation_loss = 10.0
    parallel_net = nn.DataParallel(model, device_ids=[0, 1, 2, 3])
    for epoch in tqdm.tqdm(range(epochs)):
        logger.info('Epoch {}/{}'.format(epoch + 1, epochs))
        logger.info('-' * 15)
        print('Epoch {}/{}'.format(epoch + 1, epochs))
        print('-' * 10)

        # model.to(device)
        parallel_net.train()

        running_loss = 0.0
        dice_loss = 0.0
        accuracy_score = 0.0
        n = n_train / batch_size
        # training
        for index, batch in enumerate(train_dataloader):

            image = batch[0].unsqueeze(1).to(device=device, dtype=torch.float32)
            gt = batch[1].to(device=device, dtype=torch.int64)
            parallel_net.to(device)
            optimizer.zero_grad()

            with autocast():
                pred = parallel_net(image)
                loss = criterion(pred, gt)

            # Backpropagation
            scaler.scale(loss.mean()).backward()
            scaler.step(optimizer)

            running_loss += float(loss.mean())
            scaler.update()

            if epoch == (epochs - 1):
                plot_image(batch[0], batch[1], pred, 'train', index)

            # plot in tensorboard
            plot_3d_image(batch[0], batch[1], pred, loss, step=epoch, writer=writer)
            dice_loss += dice(test=pred.argmax(1), reference=gt)
            accuracy_score += accuracy(test=pred.argmax(1), reference=gt)

        writer.add_scalar('Loss/train', (running_loss / n), epoch)
        writer.add_scalar('Accuracy', (accuracy_score / n), epoch)
        writer.add_scalar('Dice score', (dice_loss / n), epoch)

        print(f'Epoch : {epoch}, running loss : {running_loss}, loss: {(running_loss / n):.4f}')
        print(f'Accuracy : {accuracy_score / n}, Dice score:{dice_loss / n}')
        logger.info(f'Epoch : {epoch}, running loss : {running_loss}, loss: {(running_loss / n)}')
        logger.info(f'Dice score : {dice_loss / n}')

        # validation
        logger.info('Validation step')
        parallel_net.eval()
        val_loss = 0.0
        val_dice_loss = 0.0
        accuracy_score = 0.0
        i = 0
        for index, batch in enumerate(val_dataloader):
            image = batch[0].unsqueeze(1).to(device=device, dtype=torch.float32)
            gt = batch[1].to(device=device, dtype=torch.long)

            with torch.no_grad():
                # predict the mask
                pred = parallel_net(image)
                loss = criterion(pred, gt)

                val_loss += loss.mean()
                i += 1

                if epoch == (epochs - 1):
                    plot_image(batch[0], batch[1], pred, 'val', index)

            val_dice_loss += dice(test=pred.argmax(1), reference=gt)
            accuracy_score += accuracy(test=pred.argmax(1), reference=gt)
        if n_val > 0:
            n_val = n_val / batch_size
            val_dice_loss = val_dice_loss / n_val
            accuracy_score = accuracy_score / n_val
            print(f'Validation loss : {val_loss:.4f}')
            print(f'Accuracy - {accuracy_score}, dice score - {val_dice_loss}')
            logger.info(f'Validation loss : {val_loss}')
            logger.info(f'Validation Dice score : {val_dice_loss}')
            writer.add_scalar("Validation Loss", val_loss, epoch)
            writer.add_scalar('Val Accuracy', accuracy_score, epoch)
            writer.add_scalar('Val Dice score', val_dice_loss, epoch)
        torch.cuda.empty_cache()
        writer.flush()

        # save metrics checkpoint
        path = checkpoint_handler.generate_checkpoint_path(path2save=checkpoint_path)
        checkpoint_handler.save_checkpoint(checkpoint_path + '/metrics.pth', iteration=epoch, model=model,
                                           optimizer=optimizer)

        # save checkpoint
        if save_checkpoint and best_validation_loss > val_dice_loss:
            best_validation_loss = val_dice_loss
            best_model_name = 'best_model_' + model_name + '.pth'
            model_path = os.path.join(checkpoint_path, best_model_name)
            if os.path.exists(model_path):  # checking if there is a file with this name
                os.remove(model_path)  # deleting the file
            checkpoint = model.state_dict()
            torch.save(checkpoint, model_path)
            logger.info(f'Best checkpoint at epoch - {epoch} saved')
            logger.info(f'Best validation loss - {best_validation_loss}')

        writer.close()

    logger.info('Training completed')
