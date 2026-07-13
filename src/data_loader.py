import tensorflow as tf
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import FER_DIR, FANE_DIR, IMG_SIZE, BATCH_SIZE, SEED, CLASS_NAMES

def get_fer_datasets(model_type="custom_cnn"):
    if model_type == "custom_cnn":
        color_mode = "grayscale"
        target_size = IMG_SIZE
    elif model_type == "xception":
        color_mode = "rgb"
        target_size = (71, 71)
    else:
        raise ValueError("Model type must be 'custom_cnn' or 'xception'")

    print(f"Loading FER-2013 for {model_type.upper()}...")
    
    train_dir = FER_DIR / "train"
    test_dir = FER_DIR / "test"

    train_dataset = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        labels="inferred",
        label_mode="categorical",
        class_names=CLASS_NAMES,
        color_mode=color_mode,
        batch_size=BATCH_SIZE,
        image_size=target_size,
        shuffle=True,
        seed=SEED
    )

    test_dataset = tf.keras.utils.image_dataset_from_directory(
        test_dir,
        labels="inferred",
        label_mode="categorical",
        class_names=CLASS_NAMES,
        color_mode=color_mode,
        batch_size=BATCH_SIZE,
        image_size=target_size,
        shuffle=False,
        seed=SEED
    )

    AUTOTUNE = tf.data.AUTOTUNE
    return (train_dataset.cache().prefetch(buffer_size=AUTOTUNE), 
            test_dataset.cache().prefetch(buffer_size=AUTOTUNE))


def get_fane_test_dataset(model_type="custom_cnn"):
    if model_type == "custom_cnn":
        color_mode = "grayscale"
        target_size = IMG_SIZE
    elif model_type == "xception":
        color_mode = "rgb"
        target_size = (71, 71)
    else:
        raise ValueError("Model type must be 'custom_cnn' or 'xception'")

    print(f"Loading FANE (Test Only) for {model_type.upper()}...")

    fane_dataset = tf.keras.utils.image_dataset_from_directory(
        FANE_DIR, 
        labels="inferred",
        label_mode="categorical",
        class_names=CLASS_NAMES,
        color_mode=color_mode,
        batch_size=BATCH_SIZE,
        image_size=target_size,
        shuffle=False,
        seed=SEED
    )

    AUTOTUNE = tf.data.AUTOTUNE
    return fane_dataset.cache().prefetch(buffer_size=AUTOTUNE)