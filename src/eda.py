"""Analisi esplorativa del dataset (EDA), eseguibile come script.

Uso: python src/eda.py

Genera grafici (distribuzione delle classi, esempi di immagini per
categoria) e li salva in results/figures/, senza bisogno di un notebook.

TODO:
- plot_class_distribution(): bar chart del numero di immagini per emozione.
- plot_sample_images(): griglia di esempi di immagini per ciascuna classe.
"""

from config import RESULTS_DIR


def plot_class_distribution():
    raise NotImplementedError


def plot_sample_images():
    raise NotImplementedError


if __name__ == "__main__":
    plot_class_distribution()
    plot_sample_images()
