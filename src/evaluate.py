"""Evaluates and compares the models on FER-2013 and FANE.

Usage: python src/evaluate.py [--models custom_cnn xception]
"""

import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")  # no display on Colab

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import CLASS_NAMES, MODELS_DIR, RESULTS_DIR, SEED

from data_loader import get_fane_test_dataset, get_fer_datasets
from utils import set_seed

MODEL_TYPES = ["custom_cnn", "xception"]
LABELS = list(range(len(CLASS_NAMES)))


def _predict(model, dataset):
    """Returns the true labels and the predicted ones.

    Done batch by batch instead of model.predict because after dropping the
    undecodable files Keras does not know how many batches there are and prints
    a warning. This way labels and predictions also stay aligned for sure.
    """
    y_true, y_pred = [], []
    for images, labels in dataset:
        probabilities = model(images, training=False)
        y_true.append(labels.numpy().argmax(axis=1))
        y_pred.append(np.asarray(probabilities).argmax(axis=1))
    return np.concatenate(y_true), np.concatenate(y_pred)


def _per_class_report(y_true, y_pred):
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=LABELS, zero_division=0
    )
    return {
        name: {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }
        for i, name in enumerate(CLASS_NAMES)
    }


def plot_confusion_matrix(y_true, y_pred, title, output_path):
    """Confusion matrix normalized by row, so every row sums to 1.

    With the raw counts we would mostly see that the two datasets have a
    different number of images per class.
    """
    matrix = confusion_matrix(y_true, y_pred, labels=LABELS)
    row_totals = matrix.sum(axis=1, keepdims=True)
    normalised = np.divide(
        matrix, row_totals, out=np.zeros(matrix.shape, dtype=float), where=row_totals != 0
    )

    fig, ax = plt.subplots(figsize=(7.5, 6.5), dpi=150)
    image = ax.imshow(normalised, cmap="Blues", vmin=0, vmax=1)

    ax.set_xticks(LABELS, CLASS_NAMES, rotation=45, ha="right")
    ax.set_yticks(LABELS, CLASS_NAMES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title, pad=12)

    # white lines between the cells
    ax.set_xticks(np.arange(-0.5, len(CLASS_NAMES), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(CLASS_NAMES), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=2)
    ax.tick_params(which="minor", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    for i in range(len(CLASS_NAMES)):
        for j in range(len(CLASS_NAMES)):
            value = normalised[i, j]
            ax.text(
                j, i, f"{value:.2f}",
                ha="center", va="center", fontsize=9,
                # white text on the dark cells, otherwise it is unreadable
                color="white" if value > 0.5 else "#1f2933",
            )

    colorbar = fig.colorbar(image, ax=ax, shrink=0.85)
    colorbar.set_label("Proportion of true class")
    colorbar.outline.set_visible(False)

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def evaluate_model(model_type, figures_dir, metrics_dir):
    """Evaluates one model on the FER-2013 test set and on FANE."""
    model = tf.keras.models.load_model(MODELS_DIR / f"{model_type}.keras")

    _, _, fer_test = get_fer_datasets(model_type=model_type)
    datasets = {"fer2013": fer_test, "fane": get_fane_test_dataset(model_type)}

    rows = []
    for dataset_name, dataset in datasets.items():
        y_true, y_pred = _predict(model, dataset)

        report = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            # macro-F1 gives the same weight to every class, so we can compare
            # it between the two datasets
            "macro_f1": float(f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0)),
            "weighted_f1": float(f1_score(y_true, y_pred, labels=LABELS, average="weighted", zero_division=0)),
            "per_class": _per_class_report(y_true, y_pred),
        }

        with open(metrics_dir / f"{model_type}_{dataset_name}_report.json", "w") as f:
            json.dump(report, f, indent=2)

        plot_confusion_matrix(
            y_true,
            y_pred,
            f"{model_type} on {dataset_name}",
            figures_dir / f"{model_type}_{dataset_name}_confusion.png",
        )

        rows.append({
            "model": model_type,
            "dataset": dataset_name,
            "accuracy": report["accuracy"],
            "macro_f1": report["macro_f1"],
            "weighted_f1": report["weighted_f1"],
        })

    return rows


def build_gap_table(comparison):
    """Drop from FER-2013 to FANE for each model."""
    gaps = []
    for model_type, group in comparison.groupby("model", sort=False):
        scores = group.set_index("dataset")
        if not {"fer2013", "fane"}.issubset(scores.index):
            continue
        gaps.append({
            "model": model_type,
            "accuracy_fer2013": scores.loc["fer2013", "accuracy"],
            "accuracy_fane": scores.loc["fane", "accuracy"],
            "accuracy_drop": scores.loc["fer2013", "accuracy"] - scores.loc["fane", "accuracy"],
            "macro_f1_fer2013": scores.loc["fer2013", "macro_f1"],
            "macro_f1_fane": scores.loc["fane", "macro_f1"],
            "macro_f1_drop": scores.loc["fer2013", "macro_f1"] - scores.loc["fane", "macro_f1"],
        })
    return pd.DataFrame(gaps)


def main():
    parser = argparse.ArgumentParser(description="Evaluate and compare FER models.")
    parser.add_argument("--models", nargs="+", choices=MODEL_TYPES, default=MODEL_TYPES)
    args = parser.parse_args()

    set_seed(SEED)

    figures_dir = RESULTS_DIR / "figures"
    metrics_dir = RESULTS_DIR / "metrics"
    figures_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for model_type in args.models:
        if not (MODELS_DIR / f"{model_type}.keras").exists():
            print(f"skipping {model_type}: no checkpoint in {MODELS_DIR}")
            continue
        rows.extend(evaluate_model(model_type, figures_dir, metrics_dir))

    if not rows:
        raise SystemExit(f"no trained models found in {MODELS_DIR}; run src/train.py first")

    comparison = pd.DataFrame(rows)
    comparison.to_csv(metrics_dir / "comparison.csv", index=False)
    print(comparison.to_string(index=False, float_format="%.4f"))

    gaps = build_gap_table(comparison)
    if not gaps.empty:
        gaps.to_csv(metrics_dir / "generalization_gap.csv", index=False)
        print()
        print(gaps.to_string(index=False, float_format="%.4f"))


if __name__ == "__main__":
    main()
