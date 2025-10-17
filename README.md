# Chest CT Scan Classification using Transfer Learning with ResNet50

## Overview
This project demonstrates the use of **deep transfer learning** for **multi-class classification** of **chest CT scan images**.  
Using a pre-trained **ResNet-50** architecture (trained on ImageNet) as a feature extractor, the model classifies images into **four categories**:  
**adenocarcinoma**, **large cell carcinoma**, **squamous cell carcinoma**, and **normal lung tissue**.

The project showcases key deep learning practices such as **data augmentation**, **feature extraction**, and **fine-tuning** using **TensorFlow and Keras**.  
It highlights how transfer learning drastically improves model performance when working with **small medical imaging datasets**.

---

## Why Transfer Learning for Medical Imaging?
Medical image datasets are typically small and expensive to collect. Training a deep CNN from scratch would:
- Require **millions of labeled samples**.
- Lead to **slow convergence** and potential **overfitting**.

**Transfer learning** solves this by leveraging representations learned from large datasets like **ImageNet**.  
The ResNet-50 backbone already understands textures, edges, and shapes, allowing the final layers to specialize in detecting disease-specific patterns in CT scans.

---

##  Key Concepts Demonstrated
- **Convolutional Neural Networks (CNNs):** Hierarchical visual feature extraction from pixels to patterns.  
- **Residual Learning (ResNet):** Skip connections that prevent vanishing gradients and enable very deep networks.  
- **Transfer Learning:** Reusing a pre-trained model for a new task to improve accuracy and speed.  
- **Data Augmentation:** Synthetic expansion of training data to improve generalization.  
- **Callbacks:** EarlyStopping, ModelCheckpoint, and learning rate scheduling to ensure optimal training.  

---

##  Experimentation
**Training from Scratch (Random Weights):**
- Model initialized with random weights.
- Accuracy remained low (~50–60%) due to insufficient data.
- Model struggled to generalize and converged slowly.
  
**Feature Extractor (Top Layers Only):**
- Loaded pre-trained ResNet-50 base with all convolutional layers frozen.
- Trained only a small dense classification head on top.
  
**Full Transfer Learning (Current Implementation):**
- Used ResNet-50 pretrained backbone + custom classification head with multiple Dense and Dropout layers.
- Achieved higher test accuracy with stable validation performance.
