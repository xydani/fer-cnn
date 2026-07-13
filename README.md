# Facial Emotion Recognition: A Comparative Study between Custom CNNs and Fine-Tuned Xception

## Project Overview
This repository contains the code and resources for a comparative study on Facial Emotion Recognition (FER). The goal is to classify human faces into 7 standard emotional categories (Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral). 

The project evaluates two distinct deep learning approaches:
1. **Custom CNN**: A Convolutional Neural Network built and trained from scratch.
2. **Fine-Tuned Xception**: A pre-trained state-of-the-art architecture adapted via Transfer Learning.

A key focus of this research is assessing **Cross-Dataset Generalization**. The models are trained primarily on the [FER-2013 Dataset](https://www.kaggle.com/datasets/msambare/fer2013/data) and subsequently tested on the [FANE Dataset](https://www.kaggle.com/datasets/furcifer/fane-facial-expressions-and-emotion-dataset) to analyze performance drops due to domain shift.

---

## Project Structure

```text
├── data/
│   ├── raw/                  # Original datasets (FER-2013, FANE)
│   └── processed/            # Preprocessed images/tensors
├── models_saved/             # Saved weights (.h5 or .keras) of trained models
├── results/
│   ├── figures/              # EDA plots, confusion matrices, loss curves
│   └── metrics/              # Evaluation reports and CSV tables
├── src/                      # Source code
│   ├── config.py             # Global parameters and path definitions
│   ├── data_loader.py        # Dataset loading and augmentation pipelines
│   ├── eda.py                # Exploratory Data Analysis script
│   ├── custom_cnn.py         # Custom CNN architecture definition
│   ├── xception_finetune.py  # Transfer learning architecture definition
│   ├── train.py              # Training script
│   ├── evaluate.py           # Evaluation and benchmarking script
│   └── utils.py              # Helper functions (seeds, plotters)
└── README.md                 # Project documentation
