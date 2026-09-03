"""Exploratory data analysis, runnable as a script.

Usage: python src/eda.py

Makes the plots (class distribution, sample images per class) and saves them
in results/figures/, without needing a notebook.
"""

import os
import random
import sys

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import CLASS_NAMES, FANE_DIR, FER_DIR, RESULTS_DIR, SEED

from utils import set_seed

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
    """Returns the jpg files of one class, skipping the ones that are not jpg.

    In FANE there is a notebook renamed to happy/happy1283.jpg: if we pick it for
    the grid the script crashes, and it would also add 1 to the count. Checking
    the first bytes is enough because both datasets only contain jpg files.
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
    """Removes the extra spines and leaves only a light horizontal grid."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="both", colors=INK_MUTED, length=0)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color="#e5e4e0", linewidth=1)


def plot_class_distribution():
    """Class counts per split, plus the FER-2013 vs FANE balance comparison."""
    counts = {name: _class_counts(path) for name, path in SPLITS.items()}

    for split_name, split_counts in counts.items():
        if sum(split_counts.values()) == 0:
            raise SystemExit(
                f"no images found for {split_name} in {SPLITS[split_name]}, "
                f"the folder should hold one subfolder per class"
            )

    fig, axes = plt.subplots(1, len(counts), figsize=(15, 4.5), dpi=150)
    for ax, (split_name, split_counts) in zip(axes, counts.items()):
        values = [split_counts[name] for name in CLASS_NAMES]
        ax.bar(CLASS_NAMES, values, color=SERIES[0], width=0.65)
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
    """One row of random examples per class, for FER-2013 and for FANE.

    Useful to see the difference between the two datasets: FER-2013 is 48x48
    grayscale, FANE has bigger colour photos.
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
