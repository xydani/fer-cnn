import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import IMG_SIZE, NUM_CLASSES

# penalty added to the loss for large weights, keeps them small
L2 = regularizers.l2(1e-4)


def fer_resnet(input_shape=(IMG_SIZE[0], IMG_SIZE[1], 1)):

    def conv_bn_act(x, filters):
        # one convolution step: 3x3 filters that look at each pixel and its neighbours.
        # no bias because the BatchNorm right after already adds its own shift
        x = layers.Conv2D(filters, (3, 3), padding='same', use_bias=False, kernel_regularizer=L2)(x)
        # rescales the outputs to mean 0 and variance 1, so training is more stable
        x = layers.BatchNormalization()(x)
        # activation. unlike ReLU it keeps a small slope for negative values,
        # so a unit that goes negative can still recover
        x = layers.LeakyReLU(alpha=0.1)(x)
        return x

    def residual_block(x, filters, dropout_rate):
        # save the input, we add it back at the end of the block
        shortcut = x

        # the actual work of the block, two convolutions in a row
        x = conv_bn_act(x, filters)
        x = conv_bn_act(x, filters)

        # the two branches must have the same number of channels to be summed.
        # a 1x1 convolution changes only the channels and leaves width and height alone
        if shortcut.shape[-1] != filters:
            shortcut = layers.Conv2D(filters, (1, 1), padding='same', use_bias=False, kernel_regularizer=L2)(shortcut)
            shortcut = layers.BatchNormalization()(shortcut)

        # the skip connection. it gives the gradient a direct path back and lets the
        # block learn a correction to add to its input instead of the whole transform
        x = layers.Add()([x, shortcut])
        x = layers.LeakyReLU(alpha=0.1)(x)

        # halves width and height, keeping the strongest value of each 2x2 window
        x = layers.MaxPooling2D(pool_size=(2, 2))(x)

        # drops entire feature maps instead of single pixels. on images neighbouring
        # pixels are similar, so dropping them one by one would not remove much
        x = layers.SpatialDropout2D(dropout_rate)(x)
        return x

    inputs = layers.Input(shape=input_shape)

    # three blocks. the filters double each time because after pooling the images are
    # smaller and we can afford more of them, and the dropout grows because deeper
    # blocks have more parameters and overfit more easily

    # --- Block 1: Initial Feature Extraction ---
    x = residual_block(inputs, 64, dropout_rate=0.2)   # 48x48 -> 24x24

    # --- Block 2: Intermediate Features ---
    x = residual_block(x, 128, dropout_rate=0.3)       # 24x24 -> 12x12

    # --- Block 3: High-Level Abstraction ---
    x = residual_block(x, 256, dropout_rate=0.4)       # 12x12 -> 6x6

    # --- Classifier Head ---
    # averages each 6x6 feature map down to a single number, so we go from
    # 6x6x256 to 256 values. flattening instead would give a huge dense layer
    x = layers.GlobalAveragePooling2D()(x)

    # one hidden dense layer that mixes those 256 values before the decision
    x = layers.Dense(256, use_bias=False, kernel_regularizer=L2)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(alpha=0.1)(x)
    # turns off half the units at each training step, only during training
    x = layers.Dropout(0.5)(x)

    # one output per class, softmax turns the scores into probabilities summing to 1
    outputs = layers.Dense(NUM_CLASSES, activation='softmax', kernel_regularizer=L2)(x)

    return models.Model(inputs, outputs, name="FER_ResNet")


if __name__ == "__main__":
    test_model = fer_resnet()
    test_model.summary()
