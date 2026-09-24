# Skin Cancer Classification using Transfer Learning

## Project Description

This project focuses on skin cancer classification using deep learning and transfer learning.

## Dataset

The project uses the Skin Cancer ISIC dataset from Kaggle.

Total Images: 2357

Number of Classes: 9

## Models

The following pretrained deep learning models are evaluated:

- AlexNet
- VGG16
- VGG19
- ResNet18
- ResNet50
- ResNet101
- DenseNet121
- EfficientNet-B0

## Preprocessing

Images are resized to 224 × 224 pixels.

Training data augmentation includes:

- Random Horizontal Flip
- Random Vertical Flip
- Random Rotation
- Color Jitter

## Training

Framework: PyTorch

Optimizer: AdamW

Batch Size: 32

Learning Rate: 0.0001

Hardware: Google Colab GPU

## Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1-Score
- AUC

## Notebook

The complete implementation is available in the Jupyter Notebook included in this repository.
