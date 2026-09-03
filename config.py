"""Central configuration for paths and hyperparameters."""

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
DATA_RAW_DIR = ROOT_DIR / "data" / "raw"
FER_DIR = DATA_RAW_DIR / "fer-2013"  
FANE_DIR = DATA_RAW_DIR / "fane_data"

MODELS_DIR = ROOT_DIR / "models_saved"
RESULTS_DIR = ROOT_DIR / "results"

IMG_SIZE = (48, 48)  # Native FER-2013 resolution
NUM_CLASSES = 7

# Anything in FANE that is NOT in this list will be ignored later.
CLASS_NAMES = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

# --- TRAINING HYPERPARAMETERS ---
BATCH_SIZE = 64
EPOCHS = 50

LEARNING_RATE = 1e-3           # training from scratch
LEARNING_RATE_FINETUNE = 1e-4  # smaller, otherwise it ruins the ImageNet weights

# xception only: epochs spent training just the head with the base still frozen,
# so the random head does not push large gradients into the pretrained weights
WARMUP_EPOCHS = 5

# validation set taken from the train folder, so the test set is used only at the end
VAL_SPLIT = 0.1

SEED = 42