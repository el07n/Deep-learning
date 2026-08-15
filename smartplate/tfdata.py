"""TensorFlow input-pipeline helpers for the prepared CSV manifest."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd


TARGET_COLUMNS = ("total_calories", "total_protein")


def _tensorflow() -> Any:
    try:
        import tensorflow as tf
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "TensorFlow is required for the image pipeline. Install requirements.txt first."
        ) from exc
    return tf


def read_manifest(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {
        "dish_id",
        "image_path",
        "split",
        "ingredient_indices",
        *TARGET_COLUMNS,
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Manifest is missing columns: {sorted(missing)}")
    return frame


def dense_ingredient_labels(
    ingredient_indices: Sequence[str], num_ingredients: int
) -> np.ndarray:
    labels = np.zeros((len(ingredient_indices), num_ingredients), dtype=np.float32)
    for row_index, encoded in enumerate(ingredient_indices):
        for label_index in json.loads(encoded):
            label_index = int(label_index)
            if not 0 <= label_index < num_ingredients:
                raise ValueError(f"Ingredient index out of range: {label_index}")
            labels[row_index, label_index] = 1.0
    return labels


def compute_target_scaler(training_frame: pd.DataFrame) -> dict[str, object]:
    """Use positive p95 divisors so the regression heads retain non-negative targets."""

    scales: list[float] = []
    for column in TARGET_COLUMNS:
        value = float(np.quantile(training_frame[column].to_numpy(dtype=float), 0.95))
        scales.append(max(value, 1.0))
    return {"columns": list(TARGET_COLUMNS), "scales": scales, "method": "train_p95"}


def make_dataset(
    frame: pd.DataFrame,
    num_ingredients: int,
    target_scales: Sequence[float],
    image_size: int,
    batch_size: int,
    training: bool,
    seed: int,
) -> Any:
    tf = _tensorflow()
    paths = frame["image_path"].astype(str).to_numpy()
    ingredients = dense_ingredient_labels(
        frame["ingredient_indices"].astype(str).tolist(), num_ingredients
    )
    nutrition = frame[list(TARGET_COLUMNS)].to_numpy(dtype=np.float32)
    nutrition = nutrition / np.asarray(target_scales, dtype=np.float32)

    dataset = tf.data.Dataset.from_tensor_slices(
        (
            paths,
            {"ingredients": ingredients, "nutrition": nutrition},
        )
    )
    if training:
        dataset = dataset.shuffle(
            buffer_size=max(len(frame), batch_size * 4),
            seed=seed,
            reshuffle_each_iteration=True,
        )

    def load_image(path: Any, targets: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
        encoded = tf.io.read_file(path)
        image = tf.io.decode_image(encoded, channels=3, expand_animations=False)
        image.set_shape([None, None, 3])
        image = tf.image.resize(tf.cast(image, tf.float32), [image_size, image_size])
        return image, targets

    dataset = dataset.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size, drop_remainder=False)
    return dataset.prefetch(tf.data.AUTOTUNE)


def model_predictions_as_dict(model: Any, predictions: Any) -> dict[str, np.ndarray]:
    if isinstance(predictions, dict):
        return {str(key): np.asarray(value) for key, value in predictions.items()}
    if not isinstance(predictions, (list, tuple)):
        raise TypeError("Expected dict/list model predictions for the multi-output model")
    return {
        str(name): np.asarray(value)
        for name, value in zip(model.output_names, predictions, strict=True)
    }

