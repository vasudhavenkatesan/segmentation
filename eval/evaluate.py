import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, hamming_loss
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from pytorchcheckpoint.checkpoint import CheckpointHandler
import config


def mIoU(y_pred, y_true):
    y_pred = (y_pred.data.cpu().numpy()).argmax(axis=1)
    y_true = y_true.cpu().numpy()
    y_pred = y_pred.flatten()
    y_true = y_true.flatten()
    # Accuracy Score
    val = accuracy_score(y_true, y_pred, normalize=True, sample_weight=None)
    ((y_pred == y_true).all(axis=0).sum() / y_pred.shape[0])
    print(f'Accuract score - {val}')
    # Hamming Loss
    hamming_loss(y_true, y_pred)
    scores = (y_pred != y_true).sum(axis=0)
    numerator = scores.sum()
    denominator = ((scores != 0).sum() * y_true.shape[0])
    hl = (numerator / denominator)
    print(f'Hamming loss - {hl}')
    current = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    intersection = np.diag(current)
    ground_truth_set = current.sum(axis=1)
    predicted_set = current.sum(axis=0)
    union = ground_truth_set + predicted_set - intersection
    IoU = np.divide(intersection, union.astype(np.float32), where=union != 0)
    return np.mean(IoU)

    # return val, hl, np.mean(IoU)


def visualise(img, mask):
    classes = mask.shape[0] if len(mask.shape) > 2 else 1
    fig, ax = plt.subplots(1, classes + 1)
    ax[0].set_title('Input image')
    ax[0].imshow(img)
    if classes > 1:
        for i in range(classes):
            ax[i + 1].set_title(f'Output mask (class {i + 1})')
            ax[i + 1].imshow(mask[1, :, :])
    else:
        ax[1].set_title(f'Output mask')
        ax[1].imshow(mask)
    plt.xticks([]), plt.yticks([])
    plt.show()


def save_metrics(epoch, loss, dice_loss, checkpoint_handler):
    checkpoint_handler.store_running_var_with_header(header='train', var_name='loss', iteration=epoch, value=loss)
    checkpoint_handler.store_running_var_with_header(header='train', var_name='dice_score', iteration=epoch,
                                                     value=dice_loss)


def evaluate_model(checkpoint_name):
    checkpoint_handler = CheckpointHandler()

    checkpoint_handler = checkpoint_handler.load_checkpoint(checkpoint_name)

    model_name = checkpoint_handler.get_var(var_name='Model type')
    epochs = checkpoint_handler.get_var(var_name='epochs')
    batch_size = checkpoint_handler.get_var(var_name='batch_size')
    learning_rate = checkpoint_handler.get_var(var_name='learning_rate')
    train_loss = []
    dice_loss = []
    for i in range(0, epochs):
        train_loss.append(
            checkpoint_handler.get_running_var_with_header(header='train', var_name='loss',
                                                           iteration=i).detach().numpy())
        dice_loss.append(
            checkpoint_handler.get_running_var_with_header(header='train', var_name='dice_score',
                                                           iteration=i).detach().numpy())

    plot_metrics(model_name, epochs, batch_size, learning_rate, train_loss, dice_loss)


def plot_metrics(model_name, epochs, batch_size, learning_rate, train_loss, dice_loss):
    date = datetime.now().strftime("%d_%m_%I_%M_%S_%p")
    filename = model_name + '_' + date
    epochs = range(0, epochs)
    plt.figure(filename)
    plt.plot(epochs, train_loss, 'g', label='Training loss')
    plt.plot(epochs, dice_loss, 'b', label='Dice loss')
    plt.title('Training and Dice loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    plt.savefig(filename)


if __name__ == '__main__':
    checkpoint_path = Path('../checkpoints/model1.pth')
    evaluate_model(checkpoint_path)
