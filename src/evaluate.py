import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")

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

METRICS_DIR = RESULTS_DIR / "metrics"
FIGURES_DIR = RESULTS_DIR / "figures"


def load_run_config(run_name):
    path = METRICS_DIR / f"{run_name}_history.json"
    if not path.exists():
        return None
    with open(path) as f:
        history = json.load(f)
    model_type = history.get("model", run_name)
    if model_type not in MODEL_TYPES:
        return None
    return {
        "run": run_name,
        "model": model_type,
        "batch_size": history.get("batch_size"),
        "fine_tune_from": history.get("fine_tune_from"),
        "warmup_epochs": history.get("warmup_epochs", 0),
        "trainable_params": history.get("trainable_params"),
        "total_params": history.get("total_params"),
    }


def discover_runs():
    return sorted(
        path.stem for path in MODELS_DIR.glob("*.keras")
        if (METRICS_DIR / f"{path.stem}_history.json").exists()
    )


def _predict(model, dataset):
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
                color="white" if value > 0.5 else "#1f2933",
            )

    colorbar = fig.colorbar(image, ax=ax, shrink=0.85)
    colorbar.set_label("Proportion of true class")
    colorbar.outline.set_visible(False)

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def get_eval_datasets(model_type, cache):
    if model_type not in cache:
        _, _, fer_test = get_fer_datasets(model_type=model_type)
        cache[model_type] = {
            "fer2013": fer_test,
            "fane": get_fane_test_dataset(model_type),
        }
    return cache[model_type]


def evaluate_run(config, cache):
    run_name = config["run"]
    model = tf.keras.models.load_model(MODELS_DIR / f"{run_name}.keras")

    rows = []
    for dataset_name, dataset in get_eval_datasets(config["model"], cache).items():
        y_true, y_pred = _predict(model, dataset)

        report = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "macro_f1": float(f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0)),
            "weighted_f1": float(f1_score(y_true, y_pred, labels=LABELS, average="weighted", zero_division=0)),
            "per_class": _per_class_report(y_true, y_pred),
        }

        with open(METRICS_DIR / f"{run_name}_{dataset_name}_report.json", "w") as f:
            json.dump(report, f, indent=2)

        plot_confusion_matrix(
            y_true,
            y_pred,
            f"{run_name} on {dataset_name}",
            FIGURES_DIR / f"{run_name}_{dataset_name}_confusion.png",
        )

        rows.append({
            "run": run_name,
            "model": config["model"],
            "batch_size": config["batch_size"],
            "dataset": dataset_name,
            "accuracy": report["accuracy"],
            "macro_f1": report["macro_f1"],
            "weighted_f1": report["weighted_f1"],
        })

    return rows


def build_gap_table(comparison):
    gaps = []
    for run_name, group in comparison.groupby("run", sort=False):
        scores = group.set_index("dataset")
        if not {"fer2013", "fane"}.issubset(scores.index):
            continue
        gaps.append({
            "run": run_name,
            "model": group["model"].iloc[0],
            "batch_size": group["batch_size"].iloc[0],
            "accuracy_fer2013": scores.loc["fer2013", "accuracy"],
            "accuracy_fane": scores.loc["fane", "accuracy"],
            "accuracy_drop": scores.loc["fer2013", "accuracy"] - scores.loc["fane", "accuracy"],
            "macro_f1_fer2013": scores.loc["fer2013", "macro_f1"],
            "macro_f1_fane": scores.loc["fane", "macro_f1"],
            "macro_f1_drop": scores.loc["fer2013", "macro_f1"] - scores.loc["fane", "macro_f1"],
        })
    return pd.DataFrame(gaps)


def main():
    parser = argparse.ArgumentParser(description="Evaluate and compare FER runs.")
    parser.add_argument("--runs", nargs="+", default=None)
    args = parser.parse_args()

    set_seed(SEED)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    run_names = args.runs if args.runs is not None else discover_runs()

    rows = []
    cache = {}
    for run_name in run_names:
        if not (MODELS_DIR / f"{run_name}.keras").exists():
            print(f"skipping {run_name}: no checkpoint in {MODELS_DIR}")
            continue
        config = load_run_config(run_name)
        if config is None:
            print(f"skipping {run_name}: no usable history in {METRICS_DIR}")
            continue
        rows.extend(evaluate_run(config, cache))

    if not rows:
        raise SystemExit(f"no trained runs found in {MODELS_DIR}; run src/train.py first")

    comparison = pd.DataFrame(rows)
    comparison.to_csv(METRICS_DIR / "comparison.csv", index=False)
    print(comparison.to_string(index=False, float_format="%.4f"))

    gaps = build_gap_table(comparison)
    if not gaps.empty:
        gaps.to_csv(METRICS_DIR / "generalization_gap.csv", index=False)
        print()
        print(gaps.to_string(index=False, float_format="%.4f"))


if __name__ == "__main__":
    main()
