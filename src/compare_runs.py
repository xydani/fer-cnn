import json
import math
import os
import sys

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RESULTS_DIR

METRICS_DIR = RESULTS_DIR / "metrics"
FIGURES_DIR = RESULTS_DIR / "figures"

DATASETS = ("fer2013", "fane")
DATASET_COLOURS = {"fer2013": "#2a78d6", "fane": "#eb6834"}
DATASET_LABELS = {"fer2013": "FER-2013 test", "fane": "FANE"}
BATCH_COLOURS = {32: "#8b5cd6", 64: "#2f9e6f"}
INK = "#0b0b0b"
INK_MUTED = "#52514e"


def _style_axis(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="both", colors=INK_MUTED, length=0)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color="#e5e4e0", linewidth=1)


def _load_json(path):
    with open(path) as f:
        return json.load(f)


def _family(run_name, batch_size):
    suffix = f"_bs{batch_size}"
    if batch_size is not None and run_name.endswith(suffix):
        return run_name[: -len(suffix)]
    return run_name


def _confidence_interval(accuracy, n):
    if not n:
        return float("nan")
    return 1.96 * math.sqrt(accuracy * (1 - accuracy) / n)


def collect_runs():
    rows = []
    for path in sorted(METRICS_DIR.glob("*_history.json")):
        history = _load_json(path)
        run_name = history.get("run_name")
        if run_name is None or path.name != f"{run_name}_history.json":
            continue

        best = min(range(len(history["val_loss"])), key=lambda i: history["val_loss"][i])
        batch_size = history.get("batch_size")
        total = history.get("total_params") or 0
        trainable = history.get("trainable_params") or 0

        row = {
            "run": run_name,
            "family": _family(run_name, batch_size),
            "model": history.get("model"),
            "batch_size": batch_size,
            "fine_tune_from": history.get("fine_tune_from", ""),
            "warmup_epochs": history.get("warmup_epochs", 0),
            "epochs_run": history.get("epochs_run", len(history["loss"])),
            "best_epoch": best + 1,
            "best_val_loss": history["val_loss"][best],
            "best_val_accuracy": history["val_accuracy"][best],
            "train_val_gap": history["accuracy"][best] - history["val_accuracy"][best],
            "total_params": total,
            "trainable_params": trainable,
            "trainable_pct": 100 * trainable / total if total else float("nan"),
        }

        complete = True
        for dataset in DATASETS:
            report_path = METRICS_DIR / f"{run_name}_{dataset}_report.json"
            if not report_path.exists():
                complete = False
                break
            report = _load_json(report_path)
            n = sum(c["support"] for c in report["per_class"].values())
            row[f"accuracy_{dataset}"] = report["accuracy"]
            row[f"accuracy_ci95_{dataset}"] = _confidence_interval(report["accuracy"], n)
            row[f"macro_f1_{dataset}"] = report["macro_f1"]
            row[f"weighted_f1_{dataset}"] = report["weighted_f1"]
        if not complete:
            print(f"skipping {run_name}: it has not been evaluated yet")
            continue

        row["accuracy_drop"] = row["accuracy_fer2013"] - row["accuracy_fane"]
        row["macro_f1_drop"] = row["macro_f1_fer2013"] - row["macro_f1_fane"]
        row["macro_f1_drop_pct"] = 100 * row["macro_f1_drop"] / row["macro_f1_fer2013"]
        rows.append(row)

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["model", "family", "batch_size"])


def build_batch_size_table(runs):
    metrics = ["accuracy_fer2013", "macro_f1_fer2013", "accuracy_fane",
               "macro_f1_fane", "best_val_loss"]
    rows = []
    for family, group in runs.groupby("family", sort=False):
        sizes = group.set_index("batch_size")
        if not {32, 64}.issubset(sizes.index):
            continue
        row = {"family": family}
        for metric in metrics:
            row[f"{metric}_bs32"] = sizes.loc[32, metric]
            row[f"{metric}_bs64"] = sizes.loc[64, metric]
            row[f"{metric}_delta"] = sizes.loc[32, metric] - sizes.loc[64, metric]
        rows.append(row)
    return pd.DataFrame(rows)


def _bar_panel(ax, families, values, labels, colours, title, tick_labels=None):
    positions = np.arange(len(families))
    width = 0.38
    for i, (label, colour) in enumerate(zip(labels, colours)):
        offset = (i - (len(labels) - 1) / 2) * width
        bars = ax.bar(positions + offset, values[label], width, color=colour, label=label)
        for bar, value in zip(bars, values[label]):
            ax.annotate(f"{value:.3f}", (bar.get_x() + bar.get_width() / 2, value),
                        textcoords="offset points", xytext=(0, 3),
                        ha="center", fontsize=8, color=INK_MUTED)
    if tick_labels is None:
        tick_labels = [f.replace("_", "\n") for f in families]
    ax.set_xticks(positions, tick_labels, fontsize=9)
    ax.set_ylim(0, max(max(series) for series in values.values()) * 1.25)
    ax.set_title(title, color=INK, fontsize=12)
    _style_axis(ax)


def plot_batch_size_comparison(runs):
    paired = runs[runs.groupby("family")["batch_size"].transform("nunique") == 2]
    if paired.empty:
        print("no family was trained at both batch sizes, skipping the batch size figure")
        return

    families = list(dict.fromkeys(paired["family"]))
    panel_width = max(3.2, 1.6 * len(families))
    fig, axes = plt.subplots(1, 2, figsize=(2 * panel_width + 1.5, 4.8), dpi=150)

    for ax, dataset in zip(axes, DATASETS):
        values = {}
        for size in (32, 64):
            subset = paired[paired["batch_size"] == size].set_index("family")
            values[f"batch {size}"] = [subset.loc[f, f"macro_f1_{dataset}"] for f in families]
        _bar_panel(ax, families, values, [f"batch {s}" for s in (32, 64)],
                   [BATCH_COLOURS[32], BATCH_COLOURS[64]],
                   f"Macro-F1 on {DATASET_LABELS[dataset]}")
    axes[0].set_ylabel("Macro-F1")
    axes[0].legend(frameon=False, fontsize=9, labelcolor=INK_MUTED)

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "batch_size_comparison.png", bbox_inches="tight")
    plt.close(fig)


def strategy_subset(runs):
    xception = runs[runs["model"] == "xception"]
    if xception.empty:
        return None, None
    batch_size = xception.groupby("batch_size")["best_val_loss"].mean().idxmin()
    subset = xception[xception["batch_size"] == batch_size].sort_values("trainable_pct")
    return subset, batch_size


def plot_finetune_strategy(runs):
    subset, batch_size = strategy_subset(runs)
    if subset is None or len(subset) < 2:
        return

    families = list(subset["family"])
    values = {DATASET_LABELS[d]: list(subset[f"macro_f1_{d}"]) for d in DATASETS}
    tick_labels = [
        f"{family.replace('xception_', '')}\n{pct:.1f}% trainable"
        for family, pct in zip(families, subset["trainable_pct"])
    ]

    fig, ax = plt.subplots(figsize=(9, 5), dpi=150)
    _bar_panel(ax, families, values, [DATASET_LABELS[d] for d in DATASETS],
               [DATASET_COLOURS[d] for d in DATASETS],
               f"Xception macro-F1 by fine-tuning strategy (batch {batch_size:.0f})",
               tick_labels=tick_labels)
    ax.set_ylabel("Macro-F1")
    ax.legend(frameon=False, fontsize=9, labelcolor=INK_MUTED)

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "finetune_strategy.png", bbox_inches="tight")
    plt.close(fig)


def plot_strategy_curves(runs):
    subset, batch_size = strategy_subset(runs)
    if subset is None or len(subset) < 2:
        return

    palette = ["#2a78d6", "#eb6834", "#2f9e6f", "#8b5cd6"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), dpi=150)

    for i, (_, run) in enumerate(subset.iterrows()):
        history = _load_json(METRICS_DIR / f"{run['run']}_history.json")
        colour = palette[i % len(palette)]
        epochs = range(1, len(history["val_loss"]) + 1)
        label = run["family"].replace("xception_", "")
        axes[0].plot(epochs, history["val_loss"], color=colour, linewidth=2, label=label)
        axes[1].plot(epochs, history["val_accuracy"], color=colour, linewidth=2)

        marker = int(run["best_epoch"]) - 1
        axes[0].plot(marker + 1, history["val_loss"][marker], "o", color=colour,
                     markersize=7, markeredgecolor="white", markeredgewidth=1.5)
        axes[1].plot(marker + 1, history["val_accuracy"][marker], "o", color=colour,
                     markersize=7, markeredgecolor="white", markeredgewidth=1.5)

        warmup = history.get("warmup_epochs")
        if warmup:
            for ax in axes:
                ax.axvline(warmup + 0.5, color=colour, linewidth=1, linestyle=":", alpha=0.8)

    axes[0].set_title(f"Validation loss (batch {batch_size:.0f})", color=INK, fontsize=12)
    axes[1].set_title(f"Validation accuracy (batch {batch_size:.0f})", color=INK, fontsize=12)
    for ax in axes:
        ax.set_xlabel("Epoch")
        _style_axis(ax)
    axes[0].legend(frameon=False, fontsize=9, labelcolor=INK_MUTED)

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "xception_strategy_curves.png", bbox_inches="tight")
    plt.close(fig)


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    runs = collect_runs()
    if runs.empty:
        raise SystemExit(
            f"no evaluated run found in {METRICS_DIR}; run src/train.py and "
            f"src/evaluate.py first")

    runs.to_csv(METRICS_DIR / "all_runs.csv", index=False)

    summary = runs[["run", "batch_size", "trainable_pct", "epochs_run", "best_epoch",
                    "best_val_loss", "accuracy_fer2013", "macro_f1_fer2013",
                    "accuracy_fane", "macro_f1_fane", "macro_f1_drop"]]
    print(summary.to_string(index=False, float_format="%.4f"))

    batch_sizes = build_batch_size_table(runs)
    if not batch_sizes.empty:
        batch_sizes.to_csv(METRICS_DIR / "batch_size_effect.csv", index=False)
        print()
        print(batch_sizes.to_string(index=False, float_format="%.4f"))

    plot_batch_size_comparison(runs)
    plot_finetune_strategy(runs)
    plot_strategy_curves(runs)


if __name__ == "__main__":
    main()
