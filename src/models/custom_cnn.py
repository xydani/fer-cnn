import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import IMG_SIZE, NUM_CLASSES

L2 = regularizers.l2(1e-4)


def build_improved_fer_cnn(input_shape=(IMG_SIZE[0], IMG_SIZE[1], 1)):

    def conv_bn_act(x, filters):
        x = layers.Conv2D(filters, (3, 3), padding='same', use_bias=False, kernel_regularizer=L2)(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU(alpha=0.1)(x)
        return x

    def residual_block(x, filters, dropout_rate):
        shortcut = x
        x = conv_bn_act(x, filters)
        x = conv_bn_act(x, filters)

        if shortcut.shape[-1] != filters:
            shortcut = layers.Conv2D(filters, (1, 1), padding='same', use_bias=False, kernel_regularizer=L2)(shortcut)
            shortcut = layers.BatchNormalization()(shortcut)

        x = layers.Add()([x, shortcut])
        x = layers.LeakyReLU(alpha=0.1)(x)
        x = layers.MaxPooling2D(pool_size=(2, 2))(x)
        x = layers.SpatialDropout2D(dropout_rate)(x)
        return x

    inputs = layers.Input(shape=input_shape)

    # --- Block 1: Initial Feature Extraction ---
    x = residual_block(inputs, 64, dropout_rate=0.2)

    # --- Block 2: Intermediate Features ---
    x = residual_block(x, 128, dropout_rate=0.3)

    # --- Block 3: High-Level Abstraction ---
    x = residual_block(x, 256, dropout_rate=0.4)

    # --- Classifier Head ---
    x = layers.GlobalAveragePooling2D()(x)

    x = layers.Dense(256, use_bias=False, kernel_regularizer=L2)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.1)(x)
    x = layers.Dropout(0.5)(x)

    outputs = layers.Dense(NUM_CLASSES, activation='softmax', kernel_regularizer=L2)(x)

    return models.Model(inputs, outputs, name="Improved_FER_CNN")


if __name__ == "__main__":
    test_model = build_improved_fer_cnn()
    test_model.summary()
