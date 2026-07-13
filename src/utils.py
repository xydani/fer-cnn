"""Funzioni di supporto condivise (plotting, seed, salvataggio metriche)."""

import random
import numpy as np


def set_seed(seed: int = 42) -> None:
    """Imposta il seed per numpy e random (e tensorflow se disponibile)."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass
