"""Shared helper functions (seed, plotting, saving metrics)."""

import random
import numpy as np


def set_seed(seed: int = 42) -> None:
    """Sets the seed for numpy and random (and tensorflow if available)."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass
