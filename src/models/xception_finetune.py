import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import NUM_CLASSES

# penalty added to the loss for large weights, keeps them small
L2 = regularizers.l2(1e-4)

# last conv block of Xception: we train only from here on, FER-2013 is too
# small to retrain 21M parameters
FINE_TUNE_FROM = "block14_sepconv1"


def build_finetuned_model(input_shape=(71, 71, 3), fine_tune_from=FINE_TUNE_FROM):
    base = tf.keras.applications.Xception(
        # start from the ImageNet weights instead of random ones
        weights="imagenet",
        # drop the original classifier for the 1000 ImageNet classes, we add ours
        include_top=False,
        # 71 is the smallest input Xception accepts, and it needs 3 channels
        input_shape=input_shape,
    )

    # freeze the whole pretrained base. the head built below stays trainable
    base.trainable = False

    # same head as fer_resnet, so the comparison is only about the feature extractor

    # turns each feature map of the base into a single number
    x = layers.GlobalAveragePooling2D()(base.output)

    # one hidden dense layer that mixes those values before the decision.
    # no bias because the BatchNorm right after already adds its own shift
    x = layers.Dense(256, use_bias=False, kernel_regularizer=L2)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(negative_slope=0.1)(x)
    # turns off half the units at each training step, only during training
    x = layers.Dropout(0.5)(x)

    # one output per class, softmax turns the scores into probabilities summing to 1
    outputs = layers.Dense(NUM_CLASSES, activation="softmax", kernel_regularizer=L2)(x)

    # one single model that goes from the base input straight to our own output
    model = models.Model(base.input, outputs, name="Xception_FineTuned")

    # with fine_tune_from=None the base stays completely frozen, which is the
    # feature extraction setup used for the warm-up phase
    if fine_tune_from is not None:
        unfreeze_from(model, fine_tune_from)
    return model


def unfreeze_from(model, fine_tune_from=FINE_TUNE_FROM):
    reached = False
    for layer in model.layers:
        if layer.name == fine_tune_from:
            reached = True
        if reached and not isinstance(layer, layers.BatchNormalization):
            layer.trainable = True
    return model


if __name__ == "__main__":
    test_model = build_finetuned_model()
    test_model.summary()
