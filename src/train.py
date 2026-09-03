import argparse
import json
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    EPOCHS,
    BATCH_SIZE,
    LEARNING_RATE,
    LEARNING_RATE_FINETUNE,
    MODELS_DIR,
    RESULTS_DIR,
    SEED,
    WARMUP_EPOCHS,
)

import tensorflow as tf

from data_loader import get_fer_datasets
from models.custom_cnn import fer_resnet
from models.xception_finetune import FINE_TUNE_FROM, build_finetuned_model, unfreeze_from
from utils import count_parameters, set_seed

VERBOSE = 1 if sys.stdout.isatty() else 2


def build_model(model_type, fine_tune_from=FINE_TUNE_FROM):
    if model_type == "custom_cnn":
        return fer_resnet()
    elif model_type == "xception":
        return build_finetuned_model(fine_tune_from=fine_tune_from)
    raise ValueError(f"Unknown model type: {model_type}")


def get_learning_rate(model_type):
    return LEARNING_RATE_FINETUNE if model_type == "xception" else LEARNING_RATE


def compile_model(model, learning_rate):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )


def fit_one_phase(model, train_dataset, val_dataset, epochs, learning_rate, callbacks):
    compile_model(model, learning_rate)
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=epochs,
        callbacks=callbacks,
        verbose=VERBOSE,
    )
    return history.history


def fit_two_phase(model, train_dataset, val_dataset, epochs, warmup_epochs,
                  fine_tune_from, callbacks):
    warmup = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=warmup_epochs,
        verbose=VERBOSE,
    )

    unfreeze_from(model, fine_tune_from)
    compile_model(model, LEARNING_RATE_FINETUNE)

    finetune = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=epochs,
        initial_epoch=warmup_epochs,
        callbacks=callbacks,
        verbose=VERBOSE,
    )

    return {k: warmup.history[k] + finetune.history[k] for k in warmup.history}


def main():
    parser = argparse.ArgumentParser(description="Train a FER model.")
    parser.add_argument("--model", choices=["custom_cnn", "xception"], default="custom_cnn")
    parser.add_argument("--run_name", default=None)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--warmup_epochs", type=int, default=WARMUP_EPOCHS)
    parser.add_argument("--fine_tune_from", default=FINE_TUNE_FROM)
    args = parser.parse_args()

    run_name = args.run_name or args.model
    set_seed(SEED)

    train_dataset, val_dataset, _ = get_fer_datasets(
        model_type=args.model, batch_size=args.batch_size
    )

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    metrics_dir = RESULTS_DIR / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = MODELS_DIR / f"{run_name}.keras"

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=10, restore_best_weights=True
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_path), monitor="val_loss", save_best_only=True
        ),
    ]

    warmup_epochs = args.warmup_epochs if args.model == "xception" else 0
    if 0 < args.epochs <= warmup_epochs:
        raise SystemExit(
            f"--warmup_epochs ({warmup_epochs}) must be smaller than --epochs ({args.epochs})")

    if warmup_epochs > 0:
        model = build_finetuned_model(fine_tune_from=None)
        compile_model(model, LEARNING_RATE)
        history = fit_two_phase(
            model, train_dataset, val_dataset, args.epochs, warmup_epochs,
            args.fine_tune_from, callbacks,
        )
    else:
        model = build_model(args.model, fine_tune_from=args.fine_tune_from)
        history = fit_one_phase(
            model, train_dataset, val_dataset, args.epochs,
            get_learning_rate(args.model), callbacks,
        )

    total_params, trainable_params = count_parameters(model)
    history.update({
        "run_name": run_name,
        "model": args.model,
        "batch_size": args.batch_size,
        "epochs_cap": args.epochs,
        "epochs_run": len(history["loss"]),
        "total_params": total_params,
        "trainable_params": trainable_params,
    })
    if args.model == "xception":
        history["fine_tune_from"] = args.fine_tune_from
    if warmup_epochs > 0:
        history["warmup_epochs"] = warmup_epochs

    history_path = metrics_dir / f"{run_name}_history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)


if __name__ == "__main__":
    main()
