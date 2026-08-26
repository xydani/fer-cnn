"""Analisi esplorativa del dataset (EDA), eseguibile come script.

Uso: python src/eda.py

Genera grafici (distribuzione delle classi, esempi di immagini per
categoria) e li salva in results/figures/, senza bisogno di un notebook.
"""

import os
import random
import sys

import matplotlib
matplotlib.use("Agg")  # No display in Colab/headless runs.

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import CLASS_NAMES, FANE_DIR, FER_DIR, RESULTS_DIR, SEED

from utils import set_seed

# Slot 1 and 2 of the categorical palette, in fixed order.
SERIES = ("#2a78d6", "#eb6834")
INK = "#0b0b0b"
INK_MUTED = "#52514e"

SAMPLES_PER_CLASS = 5

SPLITS = {
    "FER-2013 train": FER_DIR / "train",
    "FER-2013 test": FER_DIR / "test",
    "FANE": FANE_DIR,
}


def _class_paths(split_dir, class_name):
    """JPEG files for one class, skipping anything that is not really a JPEG.

    FANE ships a Jupyter notebook renamed to `happy/happy1283.jpg`; sampling it
    into the grid would crash, and counting it would overstate the class by one.
    Checking the magic bytes is enough here - both datasets are JPEG-only.
    """
    paths = []
    for path in sorted((split_dir / class_name).glob("*.jpg")):
        with open(path, "rb") as f:
            if f.read(3) == b"\xff\xd8\xff":
                paths.append(path)
    return paths


def _class_counts(split_dir):
    return {name: len(_class_paths(split_dir, name)) for name in CLASS_NAMES}


def _style_axis(ax):
    """Recessive axes: the bars carry the data, the frame should not compete."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="both", colors=INK_MUTED, length=0)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color="#e5e4e0", linewidth=1)


def plot_class_distribution():
    """Class counts per split, plus the FER-2013 vs FANE balance comparison."""
    counts = {name: _class_counts(path) for name, path in SPLITS.items()}

    fig, axes = plt.subplots(1, len(counts), figsize=(15, 4.5), dpi=150)
    for ax, (split_name, split_counts) in zip(axes, counts.items()):
        values = [split_counts[name] for name in CLASS_NAMES]
        ax.bar(CLASS_NAMES, values, color=SERIES[0], width=0.65)
        # One series per panel, so label the bars directly instead of a legend.
        for x, value in enumerate(values):
            ax.text(x, value, f"{value:,}", ha="center", va="bottom",
                    fontsize=8, color=INK_MUTED)
        ax.set_title(f"{split_name}  (n={sum(values):,})", color=INK, fontsize=11)
        ax.set_xticks(range(len(CLASS_NAMES)), CLASS_NAMES, rotation=45, ha="right")
        ax.set_ylim(0, max(values) * 1.15)
        _style_axis(ax)
    axes[0].set_ylabel("Images")

    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "figures" / "class_distribution.png", bbox_inches="tight")
    plt.close(fig)

    # Proportions, not counts: this is the prior shift that makes raw accuracy
    # incomparable between the two datasets, and the reason evaluate.py reports
    # macro-F1 alongside it.
    fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
    compared = ["FER-2013 train", "FANE"]
    width = 0.38
    positions = np.arange(len(CLASS_NAMES))
    for offset, (split_name, color) in enumerate(zip(compared, SERIES)):
        total = sum(counts[split_name].values())
        shares = [100 * counts[split_name][name] / total for name in CLASS_NAMES]
        ax.bar(positions + (offset - 0.5) * width, shares, width=width,
               color=color, label=split_name)

    ax.set_xticks(positions, CLASS_NAMES, rotation=45, ha="right")
    ax.set_ylabel("Share of dataset (%)")
    ax.set_title("Class balance: FER-2013 vs FANE", color=INK, fontsize=12)
    ax.legend(frameon=False, labelcolor=INK_MUTED)
    _style_axis(ax)

    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "figures" / "class_balance_comparison.png", bbox_inches="tight")
    plt.close(fig)

    header = f"{'class':<10}" + "".join(f"{name:>18}" for name in SPLITS)
    print(header)
    for name in CLASS_NAMES:
        row = f"{name:<10}" + "".join(f"{counts[s][name]:>18,}" for s in SPLITS)
        print(row)
    print(f"{'total':<10}" + "".join(f"{sum(counts[s].values()):>18,}" for s in SPLITS))


def plot_sample_images(samples_per_class=SAMPLES_PER_CLASS):
    """One row of random examples per class, for each dataset.

    Kept side by side because the visual gap between FER-2013's tight 48x48
    grayscale crops and FANE's larger colour photographs *is* the domain shift
    the study measures.
    """
    for split_name in ("FER-2013 train", "FANE"):
        split_dir = SPLITS[split_name]
        fig, axes = plt.subplots(
            len(CLASS_NAMES), samples_per_class,
            figsize=(samples_per_class * 1.5, len(CLASS_NAMES) * 1.6), dpi=150,
        )

        for row, class_name in enumerate(CLASS_NAMES):
            paths = _class_paths(split_dir, class_name)
            chosen = random.sample(paths, min(samples_per_class, len(paths)))
            for column in range(samples_per_class):
                ax = axes[row][column]
                ax.set_xticks([])
                ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_visible(False)
                if column < len(chosen):
                    image = Image.open(chosen[column])
                    ax.imshow(np.asarray(image), cmap="gray" if image.mode == "L" else None)
            axes[row][0].set_ylabel(class_name, rotation=0, ha="right", va="center",
                                    fontsize=10, color=INK, labelpad=12)

        fig.suptitle(f"{split_name} samples", color=INK, fontsize=12)
        fig.tight_layout()
        slug = split_name.split()[0].lower().replace("-", "")
        fig.savefig(RESULTS_DIR / "figures" / f"samples_{slug}.png", bbox_inches="tight")
        plt.close(fig)


if __name__ == "__main__":
    set_seed(SEED)
    (RESULTS_DIR / "figures").mkdir(parents=True, exist_ok=True)
    plot_class_distribution()
    plot_sample_images()
