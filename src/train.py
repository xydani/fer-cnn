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
from models.xception_finetune import build_finetuned_model, unfreeze_from
from utils import set_seed


def build_model(model_type):
    if model_type == "custom_cnn":
        return fer_resnet()
    elif model_type == "xception":
        return build_finetuned_model()
    raise ValueError(f"Unknown model type: {model_type}")


def get_learning_rate(model_type):
    # xception is fine-tuned, so it needs the smaller learning rate
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
    )
    return history.history


def fit_two_phase(model, train_dataset, val_dataset, epochs, warmup_epochs, callbacks):
    """Trains the head first, then unfreezes the top of the base.

    The head starts from random weights, so in the first epochs the gradients are
    large. Letting them reach the pretrained weights straight away can damage
    them before the head has learned anything, which is what this warm-up avoids.
    The base is frozen during phase one, so a normal learning rate is fine there.
    """
    warmup = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=warmup_epochs,
    )

    unfreeze_from(model)
    # compiling again is required, otherwise Keras keeps the old weight list and
    # the layers we just unfroze would not receive any update
    compile_model(model, LEARNING_RATE_FINETUNE)

    finetune = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=epochs,
        initial_epoch=warmup_epochs,
        callbacks=callbacks,
    )

    # one single curve per metric, so the plots do not have to know about phases
    return {k: warmup.history[k] + finetune.history[k] for k in warmup.history}


def main():
    parser = argparse.ArgumentParser(description="Train a FER model.")
    parser.add_argument("--model", choices=["custom_cnn", "xception"], default="custom_cnn")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    # xception only, 0 disables the warm-up and trains in a single phase
    parser.add_argument("--warmup_epochs", type=int, default=WARMUP_EPOCHS)
    args = parser.parse_args()

    set_seed(SEED)

    # we don't use the test set here, only the validation split
    train_dataset, val_dataset, _ = get_fer_datasets(
        model_type=args.model, batch_size=args.batch_size
    )

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    metrics_dir = RESULTS_DIR / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = MODELS_DIR / f"{args.model}.keras"

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=10, restore_best_weights=True
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_path), monitor="val_loss", save_best_only=True
        ),
    ]

    warmup_epochs = args.warmup_epochs if args.model == "xception" else 0

    if warmup_epochs > 0:
        # build it with the base completely frozen, phase two unfreezes the tail
        model = build_finetuned_model(fine_tune_from=None)
        compile_model(model, LEARNING_RATE)
        history = fit_two_phase(
            model, train_dataset, val_dataset, args.epochs, warmup_epochs, callbacks
        )
    else:
        model = build_model(args.model)
        history = fit_one_phase(
            model, train_dataset, val_dataset, args.epochs,
            get_learning_rate(args.model), callbacks,
        )

    if warmup_epochs > 0:
        # lets the plots mark where the second phase starts
        history["warmup_epochs"] = warmup_epochs

    history_path = metrics_dir / f"{args.model}_history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)


if __name__ == "__main__":
    main()
