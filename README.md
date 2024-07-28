# Image Classification using Convolutional Neural Networks (CNNs) on CIFAR-10

## Overview
This project demonstrates the use of Convolutional Neural Networks (CNNs) for image classification on the CIFAR-10 dataset. The CIFAR-10 dataset consists of 60,000 32x32 color images in 10 classes, with 6,000 images per class. There are 50,000 training images and 10,000 test images.

## Dataset
The CIFAR-10 dataset is a widely used benchmark in machine learning and computer vision. It consists of 10 classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, and truck. Each class has 6,000 images.

## Data Preprocessing
- Normalization: The pixel values of the images are scaled to the range [0, 1] by dividing by 255.
- One-Hot Encoding: The class labels are converted to one-hot encoded vectors.

## Data Augmentation
To improve the generalization of the model, data augmentation is applied using the ImageDataGenerator class from Keras. The following augmentations are used:
- Rotation range: 15 degrees
- Width shift range: 0.1
- Height shift range: 0.1
- Horizontal flip

## Model Architecture
The CNN model is built using the Keras Sequential API. The architecture consists of the following layers:
- Convolutional Layer 1: 32 filters, kernel size 3x3, ReLU activation, input shape (32, 32, 3)
- Batch Normalization
- Convolutional Layer 2: 32 filters, kernel size 3x3, ReLU activation
- Max Pooling Layer: Pool size 2x2
- Dropout Layer: Dropout rate 0.25
- Convolutional Layer 3: 64 filters, kernel size 3x3, ReLU activation
- Batch Normalization
- Convolutional Layer 4: 64 filters, kernel size 3x3, ReLU activation
- Max Pooling Layer: Pool size 2x2
- Dropout Layer: Dropout rate 0.25
- Flatten Layer
- Dense Layer: 512 units, ReLU activation
- Batch Normalization
- Dropout Layer: Dropout rate 0.5
- Output Layer: 10 units (one for each class), Softmax activation

## Compilation
The model is compiled using the Adam optimizer and categorical crossentropy loss. The performance metric is accuracy.

## Training
The model is trained for 10 epochs with a batch size of 64. Early stopping is used to prevent overfitting, monitoring the validation loss with a patience of 10 epochs.

## Evaluation
After training, the model's performance is evaluated on the test set. Accuracy and loss are plotted for both training and validation sets. The final accuracy on the test set is printed.

## Results
The final test accuracy is reported, along with a plot showing the training and validation accuracy and loss over epochs.

## Conclusion
This project demonstrates a CNN model capable of classifying images from the CIFAR-10 dataset with high accuracy. Data augmentation and regularization techniques like dropout and batch normalization contribute to the model's performance and generalization.

## Acknowledgments
This project utilizes the CIFAR-10 dataset, provided by the Canadian Institute For Advanced Research.
