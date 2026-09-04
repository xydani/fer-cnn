# Facial Emotion Recognition: A Comparative Study between a Custom CNN and Fine-Tuned Xception

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
│   └── raw/                        # Original datasets (FER-2013, FANE)
├── docs/                           # Report
├── models_saved/                   # Saved weights (.keras) of the trained runs
├── notebooks/
│   └── run_experiments.ipynb       # Colab notebook, runs every experiment on GPU
├── results/
│   ├── figures/                    # EDA plots, confusion matrices, curves
│   └── metrics/                    # Per-class reports and CSV tables
├── src/
│   ├── models/
│   │   ├── custom_cnn.py           # Custom CNN architecture definition
│   │   └── xception_finetune.py    # Transfer learning architecture definition
│   ├── data_loader.py              # Dataset loading and augmentation pipelines
│   ├── eda.py                      # Exploratory Data Analysis script
│   ├── train.py                    # Training script
│   ├── evaluate.py                 # Evaluation on both test sets
│   ├── compare_runs.py             # Cross-run tables and comparison figures
│   ├── plot_results.py             # Training curves and generalization gap
│   └── utils.py                    # Helper functions (seeds, parameter counts)
├── config.py                       # Global parameters and path definitions
├── requirements.txt
└── README.md
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
└── data/
    └── raw/                  <-- Extract FER-2013 and FANE here
        ├── fer-2013/{train,test}/<class>/
        └── fane_data/<class>/
```

---

## Experiments

Five configurations are trained, all on FER-2013 and all evaluated on both
FER-2013 test and FANE.

| Run | Model | Batch | Warm-up | Retrained part |
|---|---|---|---|---|
| `custom_cnn_bs32` | Custom CNN | 32 | — | whole network, from scratch |
| `custom_cnn_bs64` | Custom CNN | 64 | — | whole network, from scratch |
| `xception_1phase_bs32` | Xception | 32 | no | from `block14` on (24.6%) |
| `xception_2phase_bs32` | Xception | 32 | 5 epochs | from `block14` on (24.6%) |
| `xception_full_bs32` | Xception | 32 | 5 epochs | whole base (99.5%) |

Each pair isolates one variable, so every comparison has a single possible cause.

## Usage

Run the scripts from the repository root.

```bash
python src/eda.py                                    # dataset figures and counts
python src/train.py --model custom_cnn --run_name custom_cnn_bs64 \
                    --epochs 100 --batch_size 64
python src/evaluate.py --runs custom_cnn_bs64        # both test sets
python src/compare_runs.py                           # cross-run tables and figures
python src/plot_results.py --runs custom_cnn xception
```

Every artifact of a run is keyed by its `--run_name`, so configurations that
differ only in batch size or fine-tuning depth never overwrite each other.

Training on CPU is slow. The whole study was run on a Colab GPU through
`notebooks/run_experiments.ipynb`, which clones this repository, downloads both
datasets from Kaggle and archives the output to Google Drive in a copy of this
folder structure.

## Results

Best run of each architecture, selected on validation loss alone.

| Model | Accuracy FER-2013 | Macro-F1 | Accuracy FANE | Macro-F1 |
|---|---|---|---|---|
| Custom CNN | 61.12% | 0.5203 | 34.62% | 0.2941 |
| Xception, full fine-tuning | 65.85% | 0.6210 | 37.62% | 0.3393 |

Both models lose more than 25 accuracy points crossing to FANE, and the
difference between the two drops is not statistically significant. Pretraining
on ImageNet raises the absolute level on both datasets without reducing the
fraction lost to domain shift.

Full details in the report under `docs/`.
