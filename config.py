"""Configurazione centrale del progetto: percorsi e iperparametri condivisi."""

from pathlib import Path

# Percorsi
ROOT_DIR = Path(__file__).resolve().parent
DATA_RAW_DIR = ROOT_DIR / "data" / "raw"
DATA_PROCESSED_DIR = ROOT_DIR / "data" / "processed"
MODELS_DIR = ROOT_DIR / "models_saved"
RESULTS_DIR = ROOT_DIR / "results"

# Dataset
IMG_SIZE = (48, 48)          # FER-2013 e' nativamente 48x48 in scala di grigi
NUM_CLASSES = 7              # angry, disgust, fear, happy, sad, surprise, neutral
CLASS_NAMES = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]

# Training
BATCH_SIZE = 64
EPOCHS = 50
LEARNING_RATE = 1e-3
SEED = 42
