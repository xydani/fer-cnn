import random
import numpy as np


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass


def count_parameters(model):
    total = int(model.count_params())
    trainable = int(sum(np.prod(weight.shape) for weight in model.trainable_weights))
    return total, trainable
