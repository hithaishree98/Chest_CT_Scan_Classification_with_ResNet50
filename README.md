### ResNet50-ChestCancer-Detection
This project implements the ResNet50 architecture from scratch and uses it to classify chest cancer from CT scan images. It leverages both a fully custom residual network and transfer learning approaches to achieve high classification accuracy on the Chest CT-scan Images dataset.

## Project Highlights
 Custom ResNet50 architecture implemented from scratch using TensorFlow/Keras
 Transfer Learning support using pretrained ImageNet weights (optional)
 Classification of Chest CT scan images into 4 categories
 Training Monitoring with accuracy/loss visualization
 Includes L2 Regularization, Dropout, EarlyStopping, and ReduceLROnPlateau

## Introduction to ResNet
Residual Networks (ResNets) are deep convolutional neural networks that solve the vanishing gradient problem by introducing skip connections, allowing gradients to flow more effectively in deep architectures.

## Dataset
Source: Kaggle - Chest CT-scan Images

Structure: Pre-organized into train, valid, and test folders

Classes: 4 chest-related conditions

![download](https://github.com/user-attachments/assets/aa59085c-9ec2-4043-ae61-8f0d70f20d0e)


## Model Architecture
Custom ResNet50 implementation with:

identity_block() and convolutional_block() modules

Batch normalization, ReLU activations, and skip connections

Final classification head:

Global average pooling + fully connected dense layers

Dropout (0.3–0.5) and L2 regularization applied

Softmax output for multi-class classification

## Model Evaluation
Training and validation accuracy was plotted to observe performance trends. Below is a sample output:

python
Copy code
Test loss: 0.29
Test accuracy: 0.89
📌 The model showed:

High training accuracy with stable validation accuracy

Good generalization on test data

![download (1)](https://github.com/user-attachments/assets/021b9da8-7607-45fa-a228-033a508988df)


## Conclusion
This project demonstrates the efficacy of a ResNet50 architecture from scratch in classifying CT scan images for chest cancer detection. The combination of residual learning, data augmentation, and regularization techniques makes it robust for real-world medical imaging tasks.
