"""Rete pre-addestrata (es. Xception) con fine-tuning per il confronto.

TODO:
- build_finetuned_model(): caricare Xception (o altra rete) pre-addestrata
  su ImageNet senza il top, aggiungere un nuovo classificatore per
  NUM_CLASSES, e sbloccare gli ultimi layer per il fine-tuning.
  Nota: Xception richiede input RGB (non scala di grigi) e dimensioni
  minime maggiori di 48x48, quindi le immagini andranno convertite e
  ridimensionate rispetto a data_loader.py.
"""

from config import NUM_CLASSES


def build_finetuned_model():
    raise NotImplementedError
