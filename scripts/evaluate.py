"""Evaluate recognition and nutrition outputs and calibrate uncertainty ranges."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from smartplate.config import ProjectConfig, read_json, write_json
from smartplate.metrics import conformal_absolute_error, multilabel_metrics, regression_metrics
from smartplate.tfdata import (
    TARGET_COLUMNS,
    dense_ingredient_labels,
    make_dataset,
    model_predictions_as_dict,
    read_manifest,
)


def _predict(
    model: object,
    frame: pd.DataFrame,
    vocabulary_size: int,
    scales: list[float],
    config: ProjectConfig,
) -> dict[str, np.ndarray]:
    dataset = make_dataset(
        frame,
        vocabulary_size,
        scales,
        config.image_size,
        config.batch_size,
        training=False,
        seed=config.seed,
    )
    return model_predictions_as_dict(model, model.predict(dataset, verbose=1))


def _save_regression_plot(
    y_true: np.ndarray, y_pred: np.ndarray, output_dir: Path
) -> None:
    names = ("Calories (kcal)", "Protein (g)")
    figure, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    for index, (axis, name) in enumerate(zip(axes, names, strict=True)):
        axis.scatter(y_true[:, index], y_pred[:, index], s=18, alpha=0.55)
        low = min(y_true[:, index].min(), y_pred[:, index].min())
        high = max(y_true[:, index].max(), y_pred[:, index].max())
        axis.plot([low, high], [low, high], "--", color="black", linewidth=1)
        axis.set(title=f"Predicted vs actual: {name}", xlabel="Actual", ylabel="Predicted")
    figure.tight_layout()
    figure.savefig(output_dir / "predicted_vs_actual.png", dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--artifacts-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--threshold", type=float, default=0.50)
    parser.add_argument("--coverage", type=float, default=0.95)
    args = parser.parse_args()

    try:
        import tensorflow as tf
    except ImportError as exc:
        raise SystemExit("TensorFlow is missing. Install requirements.txt first.") from exc

    config = ProjectConfig.from_dict(dict(read_json(args.artifacts_dir / "project_config.json")))
    vocabulary = list(read_json(args.artifacts_dir / "ingredient_vocabulary.json"))
    scaler = dict(read_json(args.artifacts_dir / "target_scaler.json"))
    scales = [float(value) for value in scaler["scales"]]
    frame = read_manifest(args.prepared_dir / "manifest.csv")
    test_frame = frame.loc[frame["split"] == "test"].reset_index(drop=True)
    validation_frame = frame.loc[frame["split"] == "validation"].reset_index(drop=True)
    train_frame = frame.loc[frame["split"] == "train"].reset_index(drop=True)
    if test_frame.empty or validation_frame.empty:
        raise SystemExit("Test and validation splits are required for final evaluation")

    model = tf.keras.models.load_model(args.artifacts_dir / "smartplate.keras", compile=False)
    test_outputs = _predict(model, test_frame, len(vocabulary), scales, config)
    validation_outputs = _predict(model, validation_frame, len(vocabulary), scales, config)

    y_test_nutrition = test_frame[list(TARGET_COLUMNS)].to_numpy(dtype=float)
    y_pred_nutrition = np.maximum(
        test_outputs["nutrition"] * np.asarray(scales), 0.0
    )
    y_validation_nutrition = validation_frame[list(TARGET_COLUMNS)].to_numpy(dtype=float)
    y_validation_pred = np.maximum(
        validation_outputs["nutrition"] * np.asarray(scales), 0.0
    )
    y_test_ingredients = dense_ingredient_labels(
        test_frame["ingredient_indices"].astype(str).tolist(), len(vocabulary)
    )

    metrics: dict[str, object] = {
        "recognition": multilabel_metrics(
            y_test_ingredients, test_outputs["ingredients"], args.threshold
        ),
        "nutrition": {
            name: regression_metrics(y_test_nutrition[:, index], y_pred_nutrition[:, index])
            for index, name in enumerate(("calories", "protein"))
        },
    }
    train_means = train_frame[list(TARGET_COLUMNS)].mean().to_numpy(dtype=float)
    baseline = np.tile(train_means, (len(test_frame), 1))
    metrics["mean_baseline"] = {
        name: regression_metrics(y_test_nutrition[:, index], baseline[:, index])
        for index, name in enumerate(("calories", "protein"))
    }

    absolute_error = conformal_absolute_error(
        y_validation_nutrition, y_validation_pred, args.coverage
    )
    uncertainty = {
        "coverage": args.coverage,
        "method": "validation_absolute_residual_quantile",
        "columns": list(TARGET_COLUMNS),
        "absolute_error": [float(value) for value in absolute_error],
        "calibration_samples": int(len(validation_frame)),
    }
    args.artifacts_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.artifacts_dir / "evaluation_metrics.json", metrics)
    write_json(args.artifacts_dir / "uncertainty.json", uncertainty)

    predictions = test_frame[["dish_id", "image_path"]].copy()
    predictions["actual_calories"] = y_test_nutrition[:, 0]
    predictions["predicted_calories"] = y_pred_nutrition[:, 0]
    predictions["actual_protein"] = y_test_nutrition[:, 1]
    predictions["predicted_protein"] = y_pred_nutrition[:, 1]
    top_indices = np.argsort(test_outputs["ingredients"], axis=1)[:, ::-1][:, :3]
    predictions["top_3_ingredients"] = [
        json.dumps([vocabulary[index] for index in row], ensure_ascii=False)
        for row in top_indices
    ]
    predictions.to_csv(args.artifacts_dir / "test_predictions.csv", index=False)
    _save_regression_plot(y_test_nutrition, y_pred_nutrition, args.artifacts_dir)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

