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
fer-cnn/
├── data/
│   ├── raw/                        # Original datasets (FER-2013, FANE)
│   └── processed/                  # Preprocessed images/tensors
├── models_saved/                   # Saved weights (.h5 or .keras) of trained models
├── results/
│   ├── figures/                    # EDA plots, confusion matrices, loss curves
│   └── metrics/                    # Evaluation reports and CSV tables
├── src/                            # Source code
│   ├── models/                     # Original datasets (FER-2013, FANE)
│   │   ├── custom_cnn.py           # Custom CNN architecture definition
│   │   └── xception_finetune.py    # Transfer learning architecture definition
│   ├── data_loader.py              # Dataset loading and augmentation pipelines
│   ├── eda.py                      # Exploratory Data Analysis script
│   ├── train.py                    # Training script
│   ├── evaluate.py                 # Evaluation and benchmarking script
│   └── utils.py                    # Helper functions (seeds, plotters)
├── config.py                       # Global parameters and path definitions
├── requirements.txt 
└── README.md                       # Project documentation
```

## Installation & Setup

> **NOTE:** This project has been done using **Python 3.11**.


**1. Clone the repository**
```bash
git clone https://github.com/xydani/fer-cnn.git
cd fer-cnn
```

**2. Create a virtual environment (Recommended)**

On Linux/macOS:
```bash
python -m venv venv
source venv/bin/activate
```

On Windows:
```bash
python -m venv venv
venv\Scripts\activate
```

**3. Install dependencies**

Install the required packages using the generated requirements.txt file:
```bash
pip install -r requirements.txt
```

**4. Data Preparation**

Ensure your folder structure matches the paths defined in the root config.py file. Place the downloaded FER-2013 and FANE datasets inside the data/raw/ directory:

```text
fer-cnn/
├── config.py                 <-- Global parameters and paths
├── data/
│   ├── raw/                  <-- Extract FER-2013 and FANE here
│   └── processed/            
...
```
