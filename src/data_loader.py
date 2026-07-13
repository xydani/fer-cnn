"""Caricamento e preprocessing del dataset (FER-2013 o FANE).

TODO:
- load_dataset(): legge le immagini da data/raw/ (o il CSV di FER-2013) e
  restituisce array numpy (o tf.data.Dataset) di immagini + etichette.
- get_data_generators(): crea train/val/test generators con augmentation
  (rotazione, flip, zoom) per il training della CNN.
"""

from config import DATA_RAW_DIR, IMG_SIZE, BATCH_SIZE


def load_dataset():
    raise NotImplementedError


def get_data_generators():
    raise NotImplementedError
