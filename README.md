Knowledge Distillation for Image Classification (ViT Teacher, ResNet Student)

![Python](https://img.shields.io/badge/Python-3.x-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

Table of Contents
- [Introduction](#introduction)
- [Project Overview](#project-overview)
- [Features](#features)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Model Architectures](#model-architectures)
- [Knowledge Distillation Loss](#knowledge-distillation-loss)


Introduction

This repository presents an implementation of Knowledge Distillation for image classification. The goal is to transfer the knowledge from a large, powerful Vision Transformer (ViT) teacher model to a smaller, more efficient ResNet50 student model. This technique aims to achieve comparable performance with the student model while significantly reducing its computational footprint and inference time.

This project was developed as a final year project focusing on advanced deep learning techniques for practical applications in computer vision.

 Project Overview

The core of this project involves:
1.  Loading pre-trained Vision Transformer (DeiT Small) as the teacher model.
2.  Loading a pre-trained ResNet50 as the student model.
3.  Adapting both models for a 2-class image classification task (classes 'B' and 'M' based on the dataset used).
4.  Implementing a custom Knowledge Distillation loss function that combines the Kullback-Leibler (KL) divergence between teacher and student logits with the standard Cross-Entropy loss.
5.  Training the student model using the distilled knowledge from the teacher, along with traditional supervision.

 Features

* Knowledge Distillation: Effective transfer of knowledge from a large teacher model to a compact student model.
* Pre-trained Models: Utilizes pre-trained `deit_small_distilled_patch16_224` (ViT) as teacher and `ResNet50` as student.
* Data Augmentation: Includes `RandomHorizontalFlip`, `RandomRotation`, and `ColorJitter` for robust model training.
* Custom Loss Function:Implements a combined KD and Cross-Entropy loss for optimized training.
* Learning Rate Scheduling: Uses `StepLR` to adjust the learning rate during training for better convergence.
* Performance Visualization: Plots training and validation loss and accuracy curves to monitor model performance.
* GPU Acceleration: Configured to utilize CUDA if available for faster training.

Dataset

* Dataset Path: `/kaggle/input/enhanced-images/Attention_Dataset`
* Classes: 'B' (index 0) and 'M' (index 1).
* Preprocessing: Images are resized to 224x224 and normalized.

 Installation

To set up the project environment, follow these steps:

1.  Clone the repository:
    ```bash
    git clone [https://github.com/YourUsername/Knowledge-Distillation-ViT-ResNet-Image-Classification.git](https://github.com/YourUsername/Knowledge-Distillation-ViT-ResNet-Image-Classification.git)
    cd Knowledge-Distillation-ViT-ResNet-Image-Classification
    ```
2.  Create a virtual environment (recommended):
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: `venv\Scripts\activate`
    ```
3.  Install dependencies:
    ```bash
    pip install torch torchvision timm matplotlib torchsummary
    ```

Usage

To run the knowledge distillation training and evaluation, execute the Jupyter Notebook:

1. Ensure your dataset is correctly placed at ` /kaggle/input/enhanced-images/Attention_Dataset` or update the `dataset_path` variable in the notebook to your dataset's location.
2. Open the Jupyter Notebook:
    ```bash
    jupyter notebook Final_Year_Project.ipynb
    ```
3.  Run all cells in the notebook. The script will:
    * Load and preprocess the dataset.
    * Initialize and adapt the teacher and student models.
    * Perform knowledge distillation training over 20 epochs.
    * Display training and validation metrics for each epoch.
    * Generate plots for student model's training/validation loss and accuracy.

Results

After 20 epochs of training, the student model achieved the following performance metrics:

* Final Training Loss: ~0.1813
* Final Training Accuracy:~0.9724
* Final Validation Loss: ~0.1793
* Final Validation Accuracy:~0.9729

Model Architectures

Teacher Model: Vision Transformer (ViT)
* Model: `deit_small_distilled_patch16_224`
* Framework: `timm`
* Key Modification: Output head adapted for 2 classification classes.

Student Model: ResNet50
* Model:`resnet50`
* Framework:`torchvision.models`
* Key Modification: Fully connected layer replaced with a `nn.Sequential` block including Dropout and a Linear layer for 2 classification classes.

Knowledge Distillation Loss

The knowledge distillation loss function used in this project is a weighted sum of two components:

1.  Kullback-Leibler (KL) Divergence: Measures the difference between the softened probabilities of the teacher and student models. A higher temperature `T` (set to 5) is used to soften the logits, providing more information about the teacher's "dark knowledge".
2.  Cross-Entropy Loss: The standard supervised loss calculated between the student's predictions and the true ground-truth labels.
The combined loss is calculated as:
```python
loss = alpha * kd_loss + (1 - alpha) * ce_loss
