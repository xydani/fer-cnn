import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RESULTS_DIR

SERIES = {"custom_cnn": "#2a78d6", "xception": "#eb6834"}
LABELS = {"custom_cnn": "Custom CNN", "xception": "Xception"}
PALETTE = ["#2a78d6", "#eb6834", "#2f9e6f", "#8b5cd6"]
INK = "#0b0b0b"
INK_MUTED = "#52514e"

METRICS_DIR = RESULTS_DIR / "metrics"
FIGURES_DIR = RESULTS_DIR / "figures"

DEFAULT_RUNS = ["custom_cnn", "xception"]


def _style_axis(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="both", colors=INK_MUTED, length=0)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color="#e5e4e0", linewidth=1)


def _load_json(path):
    if not path.exists():
        print(f"missing {path.name}, skipped")
        return None
    with open(path) as f:
        return json.load(f)


def _architecture(run, history):
    return history.get("model", run)


def _colours(runs, histories):
    models = [_architecture(run, histories[run]) for run in runs]
    if len(set(models)) == len(models) and all(model in SERIES for model in models):
        return {run: SERIES[model] for run, model in zip(runs, models)}
    return {run: PALETTE[i % len(PALETTE)] for i, run in enumerate(runs)}


def _label(run, history):
    model = _architecture(run, history)
    if run == model:
        return LABELS.get(model, run)
    return run.replace("_", " ")


def plot_training_curves(runs=DEFAULT_RUNS):
    histories = {}
    for run in runs:
        history = _load_json(METRICS_DIR / f"{run}_history.json")
        if history is not None:
            histories[run] = history
    if not histories:
        return

    runs = list(histories)
    colours = _colours(runs, histories)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), dpi=150)

    for run in runs:
        h = histories[run]
        colour = colours[run]
        label = _label(run, h)
        epochs = range(1, len(h["loss"]) + 1)
        axes[0].plot(epochs, h["loss"], color=colour, linestyle="--",
                     linewidth=1.2, alpha=0.65, label=f"{label} · train")
        axes[0].plot(epochs, h["val_loss"], color=colour, linewidth=2,
                     label=f"{label} · val")
        axes[1].plot(epochs, h["accuracy"], color=colour, linestyle="--",
                     linewidth=1.2, alpha=0.65)
        axes[1].plot(epochs, h["val_accuracy"], color=colour, linewidth=2)

        best = min(range(len(h["val_loss"])), key=lambda i: h["val_loss"][i])
        axes[0].plot(best + 1, h["val_loss"][best], "o", color=colour,
                     markersize=7, markeredgecolor="white", markeredgewidth=1.5)
        axes[1].plot(best + 1, h["val_accuracy"][best], "o", color=colour,
                     markersize=7, markeredgecolor="white", markeredgewidth=1.5)

        warmup = h.get("warmup_epochs")
        if warmup:
            for ax in axes:
                ax.axvline(warmup + 0.5, color=colour, linewidth=1,
                           linestyle=":", alpha=0.8)
            axes[0].annotate("fine-tuning", (warmup + 0.5, 1), xycoords=("data", "axes fraction"),
                             textcoords="offset points", xytext=(4, -10),
                             fontsize=8, color=colour)

    axes[0].set_title("Loss", color=INK, fontsize=12)
    axes[1].set_title("Accuracy", color=INK, fontsize=12)
    for ax in axes:
        ax.set_xlabel("Epoch")
        _style_axis(ax)
    axes[0].legend(frameon=False, fontsize=9, labelcolor=INK_MUTED)

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "training_curves.png", bbox_inches="tight")
    plt.close(fig)


def plot_generalization_gap(runs=DEFAULT_RUNS):
    scores, histories = {}, {}
    for run in runs:
        history = _load_json(METRICS_DIR / f"{run}_history.json")
        values = {}
        for dataset in ("fer2013", "fane"):
            report = _load_json(METRICS_DIR / f"{run}_{dataset}_report.json")
            if report is not None:
                values[dataset] = report["macro_f1"]
        if len(values) == 2 and history is not None:
            scores[run] = values
            histories[run] = history
    if not scores:
        return

    colours = _colours(list(scores), histories)

    fig, ax = plt.subplots(figsize=(7, 5.5), dpi=150)

    for run, values in scores.items():
        colour = colours[run]
        start, end = values["fer2013"], values["fane"]
        ax.plot([0, 1], [start, end], color=colour, linewidth=2.5,
                marker="o", markersize=9, markeredgecolor="white",
                markeredgewidth=2, label=_label(run, histories[run]))
        ax.annotate(f"{start:.3f}", (0, start), textcoords="offset points",
                    xytext=(-12, 0), ha="right", va="center", fontsize=10, color=INK)
        ax.annotate(f"{end:.3f}", (1, end), textcoords="offset points",
                    xytext=(12, 4), ha="left", va="center", fontsize=10, color=INK)
        ax.annotate(f"−{100*(start-end):.2f} pp", (1, end), textcoords="offset points",
                    xytext=(12, -10), ha="left", va="center", fontsize=9, color=colour)

    ax.set_xlim(-0.35, 1.42)
    lows = [v["fane"] for v in scores.values()]
    highs = [v["fer2013"] for v in scores.values()]
    ax.set_ylim(min(lows) - 0.045, max(highs) + 0.03)
    ax.set_xticks([0, 1], ["FER-2013 test", "FANE"])
    ax.set_ylabel("Macro-F1")
    ax.set_title("Macro-F1 drop under domain shift", color=INK, fontsize=12)
    ax.legend(frameon=False, labelcolor=INK_MUTED)
    _style_axis(ax)

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "generalization_gap_slope.png", bbox_inches="tight")
    plt.close(fig)


def plot_epoch_budget_comparison(long_run="custom_cnn"):
    short = _load_json(METRICS_DIR / "custom_cnn_history_50.json")
    long = _load_json(METRICS_DIR / f"{long_run}_history.json")
    if short is None or long is None:
        return
    runs = {"50": short, "100": long}

    colours = {"50": "#8b929c", "100": SERIES["custom_cnn"]}

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), dpi=150)
    for tag, h in runs.items():
        epochs = range(1, len(h["loss"]) + 1)
        axes[0].plot(epochs, h["val_loss"], color=colours[tag], linewidth=2,
                     label=f"cap {tag} epoche ({len(h['loss'])} eseguite)")
        axes[1].plot(epochs, h["val_accuracy"], color=colours[tag], linewidth=2)

        best = min(range(len(h["val_loss"])), key=lambda i: h["val_loss"][i])
        axes[0].plot(best + 1, h["val_loss"][best], "o", color=colours[tag],
                     markersize=7, markeredgecolor="white", markeredgewidth=1.5)
        axes[1].plot(best + 1, h["val_accuracy"][best], "o", color=colours[tag],
                     markersize=7, markeredgecolor="white", markeredgewidth=1.5)

    for ax in axes:
        ax.axvline(50, color="#c0c4cb", linewidth=1, linestyle=":")
        ax.set_xlabel("Epoch")
        _style_axis(ax)
    axes[0].set_title("Validation loss", color=INK, fontsize=12)
    axes[1].set_title("Validation accuracy", color=INK, fontsize=12)
    axes[0].legend(frameon=False, fontsize=9, labelcolor=INK_MUTED)

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "epoch_budget_comparison.png", bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Draw the per-run figures.")
    parser.add_argument("--runs", nargs="+", default=DEFAULT_RUNS)
    parser.add_argument("--epoch_budget_run", default="custom_cnn")
    args = parser.parse_args()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plot_training_curves(args.runs)
    plot_generalization_gap(args.runs)
    plot_epoch_budget_comparison(args.epoch_budget_run)


if __name__ == "__main__":
    main()
