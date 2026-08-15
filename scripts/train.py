"""Train and fine-tune the SmartPlate EfficientNetV2B0 model."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from smartplate.config import ProjectConfig, read_json, write_json
from smartplate.model import build_multitask_model, unfreeze_for_fine_tuning
from smartplate.tfdata import compute_target_scaler, make_dataset, read_manifest


def _history_dict(history: object) -> dict[str, list[float]]:
    return {
        str(key): [float(item) for item in values]
        for key, values in history.history.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--head-epochs", type=int, default=12)
    parser.add_argument("--fine-tune-epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument(
        "--weights", choices=("imagenet", "none"), default="imagenet"
    )
    args = parser.parse_args()

    try:
        import tensorflow as tf
    except ImportError as exc:
        raise SystemExit("TensorFlow is missing. Install requirements.txt first.") from exc

    config_values = dict(read_json(args.prepared_dir / "project_config.json"))
    if args.batch_size:
        config_values["batch_size"] = args.batch_size
    config = ProjectConfig.from_dict(config_values)
    vocabulary = list(read_json(args.prepared_dir / "ingredient_vocabulary.json"))
    frame = read_manifest(args.prepared_dir / "manifest.csv")
    train_frame = frame.loc[frame["split"] == "train"].reset_index(drop=True)
    validation_frame = frame.loc[frame["split"] == "validation"].reset_index(drop=True)
    if train_frame.empty or validation_frame.empty:
        raise SystemExit("Both train and validation splits must contain images")

    scaler = compute_target_scaler(train_frame)
    train_data = make_dataset(
        train_frame,
        len(vocabulary),
        scaler["scales"],
        config.image_size,
        config.batch_size,
        training=True,
        seed=config.seed,
    )
    validation_data = make_dataset(
        validation_frame,
        len(vocabulary),
        scaler["scales"],
        config.image_size,
        config.batch_size,
        training=False,
        seed=config.seed,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "target_scaler.json", scaler)
    write_json(args.output_dir / "ingredient_vocabulary.json", vocabulary)
    write_json(args.output_dir / "project_config.json", config.to_dict())

    model, backbone = build_multitask_model(
        len(vocabulary), config, None if args.weights == "none" else "imagenet"
    )
    head_checkpoint = args.output_dir / "head_best.keras"
    fine_checkpoint = args.output_dir / "fine_tuned_best.keras"

    def callbacks(checkpoint: Path) -> list[object]:
        return [
            tf.keras.callbacks.ModelCheckpoint(
                checkpoint, monitor="val_loss", save_best_only=True, verbose=1
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=4, restore_best_weights=True, verbose=1
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.3, patience=2, min_lr=1e-7, verbose=1
            ),
            tf.keras.callbacks.CSVLogger(args.output_dir / "training_log.csv", append=True),
        ]

    head_history = model.fit(
        train_data,
        validation_data=validation_data,
        epochs=args.head_epochs,
        shuffle=False,
        callbacks=callbacks(head_checkpoint),
    )
    head_best = min(head_history.history["val_loss"])

    unfreeze_for_fine_tuning(model, backbone, len(vocabulary), config)
    fine_history = model.fit(
        train_data,
        validation_data=validation_data,
        epochs=args.fine_tune_epochs,
        shuffle=False,
        callbacks=callbacks(fine_checkpoint),
    )
    fine_best = min(fine_history.history["val_loss"])

    selected = fine_checkpoint if fine_best <= head_best else head_checkpoint
    shutil.copyfile(selected, args.output_dir / "smartplate.keras")
    write_json(
        args.output_dir / "training_history.json",
        {
            "head": _history_dict(head_history),
            "fine_tuning": _history_dict(fine_history),
            "selected_checkpoint": selected.name,
            "best_validation_loss": float(min(head_best, fine_best)),
        },
    )
    print(f"Selected model: {selected.name}")
    print(f"Final artifact: {(args.output_dir / 'smartplate.keras').resolve()}")


if __name__ == "__main__":
    main()
