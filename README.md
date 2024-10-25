# Handwritten Digit Recognition

This project implements a Convolutional Neural Network (CNN) for recognizing handwritten digits using the MNIST dataset. The model is trained to classify digits from 0 to 9 based on input images.

## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Evaluation](#evaluation)
- [Testing with Custom Images](#testing-with-custom-images)
- [Results](#results)

## Overview
The goal of this project is to classify images of handwritten digits using a CNN model. The model is trained on the MNIST dataset, which contains 28x28 grayscale images of digits.


## Dataset
We use the MNIST dataset from `tensorflow.keras.datasets`, which contains:
- **60,000 training images**
- **10,000 test images**

Each image is 28x28 pixels in grayscale.

## Model Architecture
The CNN model architecture is as follows:
1. **Input Layer**: 28x28 grayscale images reshaped to (28, 28, 1).
2. **Convolutional Layer 1**: 32 filters, kernel size 3x3, ReLU activation.
3. **Max Pooling Layer 1**: Pool size 2x2, stride 2.
4. **Convolutional Layer 2**: 64 filters, kernel size 3x3, ReLU activation.
5. **Max Pooling Layer 2**: Pool size 2x2, stride 2.
6. **Flatten Layer**: Flattens the output for fully connected layers.
7. **Dense Layer 1**: 128 units, ReLU activation, L2 regularization.
8. **Dropout Layer**: Dropout rate of 0.5 for regularization.
9. **Output Layer**: 10 units (for 10 classes), Softmax activation.

![mlp mnist](https://github.com/user-attachments/assets/38e91fe3-deb0-4b90-ac02-88b4a84bbae3)


## Training
The model is compiled using:
- **Optimizer**: Adam
- **Loss Function**: Categorical Crossentropy
- **Metrics**: Accuracy
- **Model File**: `my_model.keras` (saved in Keras format)

## App Features

- **Drawing Canvas**: Draw a digit on a black canvas for the model to predict.
- **Image Upload**: Upload an image (PNG format) of a handwritten digit for classification.
- **Real-time Prediction**: Displays the predicted digit on the screen.


### Training Process
The model is trained for 10 epochs with a batch size of 64. The model's performance is tracked on both training and validation datasets.

## Evaluation
After training, the model's accuracy on the test set is evaluated. The test accuracy achieved is printed in the output.

![image](https://github.com/user-attachments/assets/80c3c35e-15b8-4590-94c4-de494481cb28)

![image](https://github.com/user-attachments/assets/ecc6472d-a2ee-4e7b-b664-c74f7a2fae18)


## Results
Training and validation accuracy and loss are visualized after the training process to track performance over epochs.

## Using the App

### Draw a Digit: Use the canvas to draw a digit, then view the model's prediction.

### Example Screenshots

![image](https://github.com/user-attachments/assets/ff099c55-8542-47ba-91c5-4510b0a482a2)

### [CLICK to Visit Site](https://handwritten-digit-recognition-cnn.streamlit.app/)

## Testing with Custom Images
The trained model is tested on custom images. To test:
1. Load a grayscale image of a digit and resize it to 28x28 pixels.
2. Preprocess the image by normalizing it and reshaping it to the format expected by the model.
3. The model predicts the digit, and the result is displayed.

![image](https://github.com/user-attachments/assets/7d598a67-69ee-44e0-8c93-744bd23cbbb6)

## Contribution
Everyone feel free to contribute and Enhance the Project.

## Author
Siddharth Jaiswal

