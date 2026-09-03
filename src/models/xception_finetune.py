import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import NUM_CLASSES

L2 = regularizers.l2(1e-4)

FINE_TUNE_FROM = "block14_sepconv1"

FINE_TUNE_ALL = "all"


def build_finetuned_model(input_shape=(71, 71, 3), fine_tune_from=FINE_TUNE_FROM):
    base = tf.keras.applications.Xception(
        weights="imagenet",
        include_top=False,
        input_shape=input_shape,
    )

    base.trainable = False

    x = layers.GlobalAveragePooling2D()(base.output)

    x = layers.Dense(256, use_bias=False, kernel_regularizer=L2)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(negative_slope=0.1)(x)
    x = layers.Dropout(0.5)(x)

    outputs = layers.Dense(NUM_CLASSES, activation="softmax", kernel_regularizer=L2)(x)

    model = models.Model(base.input, outputs, name="Xception_FineTuned")

    if fine_tune_from is not None:
        unfreeze_from(model, fine_tune_from)
    return model


def unfreeze_from(model, fine_tune_from=FINE_TUNE_FROM):
    reached = fine_tune_from == FINE_TUNE_ALL
    for layer in model.layers:
        if layer.name == fine_tune_from:
            reached = True
        if reached and not isinstance(layer, layers.BatchNormalization):
            layer.trainable = True
    if not reached:
        raise ValueError(f"{fine_tune_from} is not a layer of {model.name}")
    return model


if __name__ == "__main__":
    test_model = build_finetuned_model()
    test_model.summary()
