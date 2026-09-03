import tensorflow as tf
from tensorflow.keras import layers
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import FER_DIR, FANE_DIR, IMG_SIZE, BATCH_SIZE, SEED, CLASS_NAMES, VAL_SPLIT

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
    if model_type == "xception":
        return lambda x, y: (tf.keras.applications.xception.preprocess_input(x), y)
    return lambda x, y: (x / 255.0, y)


def _drop_undecodable(dataset):
    return dataset.ignore_errors(log_warning=True)


def _finalize_eval_dataset(dataset, normalize, batch_size):
    return (
        _drop_undecodable(dataset)
        .map(normalize, num_parallel_calls=AUTOTUNE)
        .cache()
        .batch(batch_size)
        .prefetch(AUTOTUNE)
    )


def _build_augmenter():
    return tf.keras.Sequential(
        [
            layers.RandomFlip("horizontal", seed=SEED),
            layers.RandomRotation(0.03, fill_mode="reflect", seed=SEED),
            layers.RandomZoom(0.1, fill_mode="reflect", seed=SEED),
            layers.RandomTranslation(0.1, 0.1, fill_mode="reflect", seed=SEED),
        ],
        name="augmentation",
    )


def get_fer_datasets(model_type="custom_cnn", batch_size=BATCH_SIZE):
    color_mode, target_size = _get_target_config(model_type)
    normalize = _get_normalizer(model_type)
    augment = _build_augmenter()

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
        seed=SEED,
        validation_split=VAL_SPLIT,
        subset="training"
    )

    val_dataset = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        labels="inferred",
        label_mode="categorical",
        class_names=CLASS_NAMES,
        color_mode=color_mode,
        batch_size=None,
        image_size=target_size,
        shuffle=True,
        seed=SEED,
        validation_split=VAL_SPLIT,
        subset="validation"
    )

    test_dataset = tf.keras.utils.image_dataset_from_directory(
        test_dir,
        labels="inferred",
        label_mode="categorical",
        class_names=CLASS_NAMES,
        color_mode=color_mode,
        batch_size=None,
        image_size=target_size,
        shuffle=False,
        seed=SEED
    )

    train_dataset = (
        _drop_undecodable(train_dataset)
        .map(normalize, num_parallel_calls=AUTOTUNE)
        .cache()
        .shuffle(SHUFFLE_BUFFER, seed=SEED)
        .batch(batch_size)
        .map(lambda x, y: (augment(x, training=True), y), num_parallel_calls=AUTOTUNE)
        .prefetch(AUTOTUNE)
    )

    val_dataset = _finalize_eval_dataset(val_dataset, normalize, batch_size)
    test_dataset = _finalize_eval_dataset(test_dataset, normalize, batch_size)

    return train_dataset, val_dataset, test_dataset


def get_fane_test_dataset(model_type="custom_cnn"):
    color_mode, target_size = _get_target_config(model_type)
    normalize = _get_normalizer(model_type)

    fane_dataset = tf.keras.utils.image_dataset_from_directory(
        FANE_DIR,
        labels="inferred",
        label_mode="categorical",
        class_names=CLASS_NAMES,
        color_mode=color_mode,
        batch_size=None,
        image_size=target_size,
        shuffle=False,
        seed=SEED
    )

    return _finalize_eval_dataset(fane_dataset, normalize, BATCH_SIZE)
