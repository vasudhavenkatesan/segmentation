import torch.nn.functional as F
from datetime import datetime
import matplotlib.pyplot as plt
import torch
import config

logger = config.get_logger()


def one_hot_encoding(input, n_classes):
    target = F.one_hot(input, n_classes)
    # Reorder the target with depth as batch size followed by num of classes, height, width
    target = target.permute(0, 1, 4, 2, 3)
    target = target[-1, :, -1, :, :]
    return target


def plot_image(image, gt, pred, type='val'):
    date = datetime.now().strftime("%d_%m_%I_%M_%p")
    filename = 'Segm_' + type + '_' + date
    logger.info(f'Plotting image -  {filename} saved!')
    plt.figure(filename, (18, 6))
    plt.subplot(1, 3, 1)
    plt.title(f'Image')
    plt.imshow(image[-1, 12, :, :], cmap='gray')
    plt.subplot(1, 3, 2)
    plt.title(f'GT')
    plt.imshow(gt[-1, 12, :, :], cmap='gray')
    pred_for_plot = pred.detach().cpu().numpy()
    predic = torch.from_numpy(pred_for_plot)
    pred_for_plot = predic.argmax(dim=1)
    print(predic.shape)
    plt.subplot(1, 3, 3)
    plt.title('Predicted Mask')
    plt.imshow(pred_for_plot[12, :, :], cmap='gray')
    plt.savefig(filename)
