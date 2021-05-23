# Polyp segmentation

Scripts for training and testing convolutional neural network for images segmentation.

This script builds models using this library "https://github.com/qubvel/segmentation_models". Using this library made 
it possible to quickly test various models and choose the best option for solving the segmentation task.

You can view or set parameters in config.py
    
To train the model, the "Hyper-Kvasir Dataset" was used.
> https://datasets.simula.no/hyper-kvasir/

## 1. Dataset preparation
My dataset has this configuration:
```
data/
    masks/
            cju0qkwl35piu0993l0dewei2.jpg
            cju0qoxqj9q6s0835b43399p4.jpg
            ...    
    images/
            cju0qkwl35piu0993l0dewei2.jpg
            cju0qoxqj9q6s0835b43399p4.jpg
            ...
``` 
Run this script to prepare dataset for training and testing:
```shell script
python data_preparation.py
```
## 2. Training
Run training script with default parameters:
```shell script
python train.py
```
## 3. Plotting graphs
If you want to build graphs from saved logs, you can use tens or board by passing the path to the logs folder.
```shell script
tensorboard --logdir models_data/tensorboard_logs/Unet_imagenet_2021-05-19_00-20-00_False
```
## 4. Testing
You can test the model using a webcam.
For visualization, you need to pass two arguments "--weights ". In the output image, you will see the segmentation 
mask.
```shell script
python visualization.py --weights save_models/Unetresnet18imagenet/Unetresnet18.h5 --path_video test_video/vid_2.avi
```
Image segmentation, example.
 
![example_1](examples_for_github/image1.jpg)
![example_2](examples_for_github/image2.jpg)
![example_3](examples_for_github/image3.jpg)
![example_4](examples_for_github/image4.jpg)
![example_5](examples_for_github/image5.jpg)
![example_6](examples_for_github/image6.jpg)


The frame from the test video is shown below.

![example_7](examples_for_github/image7.png)

## Results
### Model Unet with backbone efficientnetb0.
The graphs show metrics during model training with loaded "imagenet" weights. The orange line is train, the blue 
line is val
Prediction examples:

This is iou_score metric.
![example_8](examples_for_github/image8.png)

This is loss.
![example_9](examples_for_github/image9.png)
