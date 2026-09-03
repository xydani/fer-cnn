from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
DATA_RAW_DIR = ROOT_DIR / "data" / "raw"


def _dataset_root(root, expected):
    if (root / expected).is_dir():
        return root
    for pattern in (f"*/{expected}", f"*/*/{expected}"):
        for candidate in sorted(root.glob(pattern)):
            if candidate.is_dir():
                return candidate.parent
    return root


FER_DIR = _dataset_root(DATA_RAW_DIR / "fer-2013", "train")
FANE_DIR = _dataset_root(DATA_RAW_DIR / "fane_data", "happy")

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