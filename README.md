# Image-Classifier

This contains the code for training a multi-class image classifier using transfer learning for `VGG16`, `ResNet50`, `Inception ResNet V2`, `MobileNet V2` with tensorflow.

## Scripts

- `data_loader.py` : Contains the code for loading the dataset from a folder structured as `class_name/*.jpg` . 
- `models.py` : Class for tensorflow model loading.
- `train.py` : Code for training the model.
- `test.py` : Code for testing the model.

## Dataset and Models

The dataset used for training and the pre-trained models can be downloaded from [here](https://drive.google.com/drive/folders/12zQMGN6USFDHPAu3O2wnaNycII-SXuVK?usp=share_link). 

## Usage

Create a conda enviroment
```sh
conda create -n image_classifier python=3.6.3
conda activate image_classifier
```

Install the required libraries
```sh
pip3 install -r requirements.txt
```

### For training

Unzip the downloaded dataset in `./dataset` folder and in the `config.yaml` set the required parameters.

```sh
python train.py --config_path <path to config file>
```
This repository currently supports the following models:
- `VGG16`
- `ResNet50`
- `Inception Resnet V2`
- `Mobilenet V2`

### For testing

```sh
python test.py --config_path <path to config file> 
```
The output of test will be stored in `./figures` folder

## Results

| Model Name | VGG16    | ResNet50    | Inception ResNet V2    | MobileNet V2    |
| :---:   | :---: | :---: | :---: | :---: |
| Accuracy (%) | 88.89    | 283   | 88.89   | 33.33%   |

The Inception-Resnet-v2 (although similar performance to VGG16) can be considered the best due to its computational efficiency owing to it being lightweight and its ability to train on a single GPU (less resource intensive).
