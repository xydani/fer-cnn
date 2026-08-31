"""Xception pretrained on ImageNet, fine-tuned, used for the comparison."""

import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import NUM_CLASSES

L2 = regularizers.l2(1e-4)

# last conv block of Xception: we train only from here on, FER-2013 is too
# small to retrain 21M parameters
FINE_TUNE_FROM = "block14_sepconv1"


def build_finetuned_model(input_shape=(71, 71, 3), fine_tune_from=FINE_TUNE_FROM):
    base = tf.keras.applications.Xception(
        weights="imagenet",
        include_top=False,
        input_shape=input_shape,
    )

    base.trainable = False
    reached = False
    for layer in base.layers:
        if layer.name == fine_tune_from:
            reached = True
        # BatchNorm stays frozen anyway: if it recomputes the statistics on our
        # small batches it ruins the pretrained weights
        if reached and not isinstance(layer, layers.BatchNormalization):
            layer.trainable = True

    # same head as fer_resnet, so the comparison is only about the feature extractor
    x = layers.GlobalAveragePooling2D()(base.output)

    x = layers.Dense(256, use_bias=False, kernel_regularizer=L2)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(negative_slope=0.1)(x)
    x = layers.Dropout(0.5)(x)

    outputs = layers.Dense(NUM_CLASSES, activation="softmax", kernel_regularizer=L2)(x)

    return models.Model(base.input, outputs, name="Xception_FineTuned")


if __name__ == "__main__":
    test_model = build_finetuned_model()
    test_model.summary()
