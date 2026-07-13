import tensorflow as tf
from tensorflow.keras import layers, models
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import IMG_SIZE, NUM_CLASSES

def build_improved_fer_cnn(input_shape=(IMG_SIZE[0], IMG_SIZE[1], 1)):
    model = models.Sequential(name="Improved_FER_CNN")

    def add_conv_block(filters, dropout_rate=0.25):
        model.add(layers.Conv2D(filters, (3, 3), padding='same', use_bias=False))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU(alpha=0.1))
        
        model.add(layers.Conv2D(filters, (3, 3), padding='same', use_bias=False))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU(alpha=0.1))
        
        model.add(layers.MaxPooling2D(pool_size=(2, 2)))
        model.add(layers.SpatialDropout2D(dropout_rate))

    # --- Block 1: Initial Feature Extraction ---
    model.add(layers.InputLayer(input_shape=input_shape))
    add_conv_block(64)

    # --- Block 2: Intermediate Features ---
    add_conv_block(128)

    # --- Block 3: High-Level Abstraction ---
    add_conv_block(256)

    # --- Classifier Head ---
    model.add(layers.GlobalAveragePooling2D())
    
    # Dense layer
    model.add(layers.Dense(256, use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(alpha=0.1))
    model.add(layers.Dropout(0.5))

    # Output
    model.add(layers.Dense(NUM_CLASSES, activation='softmax'))

    return model

if __name__ == "__main__":
    test_model = build_improved_fer_cnn()
    test_model.summary()