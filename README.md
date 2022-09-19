# Semantic Segmentation
This is an implementaion of volumetric segmentation of 3D medical images of heart using a standard Unet(Learning Dense Volumetric Segmentation from Sparse Annotation Özgün Çiçek et al.. )
This code can be used for binary and multiclass semantic segmentation of images.
## Setup
1. Install CUDA

2. Install PyTorch

3. Install dependencies

    ```pip install -r requirements.txt```

4. Download the data in dataset/data folder and train the model with respective model parameters.

    ``` python train.py --epochs 500 --batch_size 5 --learning_rate 1e-5```

5. Predict test data with saved model in models path.

    ```python predict.py --model best_model.pth --input filename```

## Results
We obtained excellent segmentation results for EM cell images. The loss function converged well in 200 iterations.
### T-tubule segmentation on cell images
This displays the segmentation of the EM cell images.
![Segm_train_1_18_08_12_36_54_PM](https://user-images.githubusercontent.com/46302072/189919255-445583c5-3850-4fab-a2fa-8511ce86e077.png)

![Segm_val_0_18_08_12_37_09_PM](https://user-images.githubusercontent.com/46302072/189921589-38ab0a9f-9f77-46eb-8f1d-27d2f34b24c9.png)
