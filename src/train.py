import argparse
import json
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import EPOCHS, BATCH_SIZE, LEARNING_RATE, MODELS_DIR, RESULTS_DIR, SEED

import tensorflow as tf

from data_loader import get_fer_datasets
from models.custom_cnn import fer_resnet
from models.xception_finetune import build_finetuned_model
from utils import set_seed


def build_model(model_type):
    if model_type == "custom_cnn":
        return fer_resnet()
    elif model_type == "xception":
        return build_finetuned_model()
    raise ValueError(f"Unknown model type: {model_type}")


def main():
    parser = argparse.ArgumentParser(description="Train a FER model.")
    parser.add_argument("--model", choices=["custom_cnn", "xception"], default="custom_cnn")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    args = parser.parse_args()

    set_seed(SEED)

    train_dataset, test_dataset = get_fer_datasets(model_type=args.model, batch_size=args.batch_size)

    model = build_model(args.model)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
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

    history = model.fit(
        train_dataset,
        validation_data=test_dataset,
        epochs=args.epochs,
        callbacks=callbacks,
    )

    history_path = metrics_dir / f"{args.model}_history.json"
    with open(history_path, "w") as f:
        json.dump(history.history, f, indent=2)


if __name__ == "__main__":
    main()
