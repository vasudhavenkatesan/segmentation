import argparse

import torch
from torch.utils.data import DataLoader
import config
from unet.unet import UNET
from unetr.unetr import UNETR
from dataset import hdf5
import matplotlib.pyplot as plt
from eval.metrics import dice, accuracy
from patchify import patchify, unpatchify

logger = config.get_logger()


def predict(net, input_path, input_dim, device):
    net.eval()
    plt.figure('Segmentation', (18, 6))

    dataset = hdf5.Hdf5Dataset(input_path, reqd_image_dim=input_dim, contains_mask=True, is_test=True)
    dataloader = DataLoader(dataset, shuffle=False, batch_size=1, num_workers=1, pin_memory=True)
    n_preds = dataset.__len__()
    pred_masks = []
    dice_loss = 0.0
    accuracy_score = 0.0
    for index, batch in enumerate(dataloader):
        image = batch[0].unsqueeze(0)
        gt = batch[1]
        plt.subplot(1, 3, 1)
        plt.title(f'Image')
        plt.imshow(batch[0][-1, 12, :, :], cmap="gray")
        plt.subplot(1, 3, 2)
        plt.title(f'Mask')
        plt.imshow(gt[-1, 12, :, :], cmap="gray")
        # 0, 2, 3)
        predicted_patches = []
        patches = patchify(image, (256, 256), step=64)
        for i in range(patches.shape[0]):
            for j in range(patches.shape[1]):
                print(i, j)

                single_patch = patches[i, j, :, :]
                single_patch_norm = np.expand_dims(normalize(np.array(single_patch), axis=1), 2)
                single_patch_input = np.expand_dims(single_patch_norm, 0)

                # Predict and threshold for values above 0.5 probability
                single_patch_prediction = (model.predict(single_patch_input)[0, :, :, 0] > 0.5).astype(np.uint8)
                predicted_patches.append(single_patch_prediction)

        predicted_patches = np.array(predicted_patches)

        predicted_patches_reshaped = np.reshape(predicted_patches, (patches.shape[0], patches.shape[1], 256, 256))
        reconstructed_image = unpatchify(predicted_patches_reshaped, large_image.shape)
        with torch.no_grad():
            prediction = net(image)
            plt.subplot(1, 3, 3)
            plt.title('Predicted Mask')
            pred_for_plot = prediction.argmax(dim=1)
            plt.imshow(pred_for_plot[-1, 12, :, :], cmap='gray')
            dice_loss += dice(test=pred_for_plot, reference=gt)
            accuracy_score += accuracy(test=pred_for_plot, reference=gt)
            plt.savefig('Segmentation')
    print('saving plot ---------')
    print(f'Accuracy - {accuracy_score / n_preds}, dice score - {dice_loss / n_preds}')
    plt.savefig('Segmentation')
    plt.show()

    return pred_masks


def get_param_arguments():
    parser = argparse.ArgumentParser(description='Unet parammeters')
    parser.add_argument('--viz', '-v', action='store_true', help='Help tp visualise the images')
    parser.add_argument("--model_name", default='unet', type=str,
                        help="model name used for predicting")
    parser.add_argument('--model', '-m', default='checkpoints/unet/best_model.pth', metavar='FILE',
                        help='File in which the model is stored')
    parser.add_argument('--input', '-ip', default='dataset/data/2_2_2_downsampled/test', metavar='FILE',
                        help='Input File')
    parser.add_argument("--image_sizex", default=384, type=int,
                        help="size of image in x axis")
    parser.add_argument("--image_sizey", default=384, type=int,
                        help="size of image in y axis")
    parser.add_argument("--image_sizez", default=64, type=int,
                        help="size of image in z axis")
    return parser.parse_args()


if __name__ == '__main__':
    parameter_arguments = get_param_arguments()
    device = config.device
    logger.info(f'Using device - {device}')
    img_size = [parameter_arguments.image_sizez, parameter_arguments.image_sizex, parameter_arguments.image_sizey]
    if parameter_arguments.model_name == "unetr":
        net = UNETR(config.n_channels, config.n_classes, img_size)
        logger.debug(f'UNETR model initialised')
    else:
        net = UNET(config.n_channels, config.n_classes)
        logger.debug(f'UNET model initialised')
    net.load_state_dict(torch.load(parameter_arguments.model, map_location=device))
    print('Model loaded')
    logger.info('Model loaded')
    print(f'Model path {parameter_arguments.model}')
    input_path = parameter_arguments.input
    print(input_path)
    predict(net, input_path, img_size, config.device)
