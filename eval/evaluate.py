import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, hamming_loss
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from pytorchcheckpoint.checkpoint import CheckpointHandler
import config


def save_metrics(epoch, loss, dice_loss, accuracy, checkpoint_handler, type):
    checkpoint_handler.store_running_var_with_header(header=type, var_name='loss', iteration=epoch, value=loss)
    checkpoint_handler.store_running_var_with_header(header=type, var_name='dice_score', iteration=epoch,
                                                     value=dice_loss)
    checkpoint_handler.store_running_var_with_header(header=type, var_name='accuracy', iteration=epoch,
                                                     value=accuracy)


def evaluate_model(checkpoint_name):
    checkpoint_handler = CheckpointHandler()

    checkpoint_handler = checkpoint_handler.load_checkpoint(checkpoint_name)

    model_name = checkpoint_handler.get_var(var_name='Model type')
    epochs = checkpoint_handler.get_var(var_name='epochs')
    batch_size = checkpoint_handler.get_var(var_name='batch_size')
    learning_rate = checkpoint_handler.get_var(var_name='learning_rate')
    train_loss = []
    dice_loss = []
    validation_loss = []
    accuracy = []
    for i in range(0, epochs):
        train_loss.append(
            checkpoint_handler.get_running_var_with_header(header='train', var_name='loss',
                                                           iteration=i))
        dice_loss.append(
            checkpoint_handler.get_running_var_with_header(header='validation', var_name='dice_score',
                                                           iteration=i))
        accuracy.append(
            checkpoint_handler.get_running_var_with_header(header='validation', var_name='accuracy',
                                                           iteration=i))
        validation_loss.append(
            checkpoint_handler.get_running_var_with_header(header='validation', var_name='loss',
                                                           iteration=i))

    plot_metrics(model_name, epochs, batch_size, learning_rate, train_loss, dice_loss, accuracy, validation_loss)


def plot_metrics(model_name, epochs, batch_size, learning_rate, train_loss, dice_loss, accuracy, validation_loss):
    cell_text = [[f'Model name:', f'{model_name}'], [f'Epochs', f'{epochs}'], [f'Batch size', f'{batch_size}'],
                 [f'Learning rate', f'{learning_rate}'], [f'Average Dice loss', f'{dice_loss[-1]}'],
                 [f'Accuracy', f'{accuracy[-1]}']]

    date = datetime.now().strftime("%d_%m")
    filename = model_name + '_metrics_' + date
    fig = plt.figure(filename)
    plt.title('Model metrics')
    axs = fig.subplots(2, 1)
    epochs = range(0, epochs)
    the_table = axs[0].table(cellText=cell_text, loc='top')

    axs[0].axis('tight')
    axs[0].axis('off')

    axs[1].plot(epochs, train_loss, 'g', label='Training loss')
    axs[1].plot(epochs, validation_loss, 'b', label='validation loss')

    axs[1].set_xlabel('Epochs')
    axs[1].set_ylabel('Loss')
    axs[1].legend()
    plt.show()
    fig.savefig(filename)


if __name__ == '__main__':
    checkpoint_path = Path('../checkpoints/unet/metrics.pth')
    evaluate_model(checkpoint_path)
