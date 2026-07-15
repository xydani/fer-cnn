import tensorflow as tf
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import FER_DIR, FANE_DIR, IMG_SIZE, BATCH_SIZE, SEED, CLASS_NAMES

AUTOTUNE = tf.data.AUTOTUNE
SHUFFLE_BUFFER = 8192


def _get_target_config(model_type):
    if model_type == "custom_cnn":
        return "grayscale", IMG_SIZE
    elif model_type == "xception":
        return "rgb", (71, 71)
    else:
        raise ValueError("Model type must be 'custom_cnn' or 'xception'")


def _get_normalizer(model_type):
    # Xception expects its own [-1, 1] preprocessing; the custom CNN just needs [0, 1].
    if model_type == "xception":
        return lambda x, y: (tf.keras.applications.xception.preprocess_input(x), y)
    return lambda x, y: (x / 255.0, y)


def get_fer_datasets(model_type="custom_cnn", batch_size=BATCH_SIZE):
    color_mode, target_size = _get_target_config(model_type)
    normalize = _get_normalizer(model_type)

    train_dir = FER_DIR / "train"
    test_dir = FER_DIR / "test"

    train_dataset = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        labels="inferred",
        label_mode="categorical",
        class_names=CLASS_NAMES,
        color_mode=color_mode,
        batch_size=None,
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
        batch_size=batch_size,
        image_size=target_size,
        shuffle=False,
        seed=SEED
    )

    train_dataset = (
        train_dataset
        .map(normalize, num_parallel_calls=AUTOTUNE)
        .cache()
        .shuffle(SHUFFLE_BUFFER, seed=SEED)
        .batch(batch_size)
        .prefetch(AUTOTUNE)
    )

    test_dataset = (
        test_dataset
        .map(normalize, num_parallel_calls=AUTOTUNE)
        .cache()
        .prefetch(AUTOTUNE)
    )

    return train_dataset, test_dataset


def get_fane_test_dataset(model_type="custom_cnn"):
    color_mode, target_size = _get_target_config(model_type)
    normalize = _get_normalizer(model_type)

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

    return (
        fane_dataset
        .map(normalize, num_parallel_calls=AUTOTUNE)
        .cache()
        .prefetch(AUTOTUNE)
    )
