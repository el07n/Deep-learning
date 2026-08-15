"""Model artifact loading and single-image inference for Streamlit."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .config import read_json
from .tfdata import model_predictions_as_dict


class SmartPlatePredictor:
    """Load exported artifacts once and return user-facing predictions."""

    def __init__(self, artifacts_dir: str | Path = "artifacts") -> None:
        directory = Path(artifacts_dir)
        required = {
            "model": directory / "smartplate.keras",
            "vocabulary": directory / "ingredient_vocabulary.json",
            "scaler": directory / "target_scaler.json",
            "uncertainty": directory / "uncertainty.json",
            "config": directory / "project_config.json",
        }
        missing = [name for name, path in required.items() if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                f"Missing model artifacts in {directory}: {', '.join(missing)}"
            )

        try:
            import tensorflow as tf
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("Install TensorFlow before running inference") from exc

        self.model = tf.keras.models.load_model(required["model"], compile=False)
        self.vocabulary = list(read_json(required["vocabulary"]))
        self.scaler = dict(read_json(required["scaler"]))
        self.uncertainty = dict(read_json(required["uncertainty"]))
        self.config = dict(read_json(required["config"]))
        self.image_size = int(self.config.get("image_size", 224))

    def predict(
        self,
        image: Image.Image | str | Path,
        top_k: int = 3,
        confidence_threshold: float = 0.50,
    ) -> dict[str, object]:
        if isinstance(image, Image.Image):
            prepared = image.convert("RGB").resize((self.image_size, self.image_size))
        else:
            with Image.open(image) as opened:
                prepared = opened.convert("RGB").resize(
                    (self.image_size, self.image_size)
                )
        batch = np.expand_dims(np.asarray(prepared, dtype=np.float32), axis=0)
        raw = self.model.predict(batch, verbose=0)
        outputs = model_predictions_as_dict(self.model, raw)

        probabilities = np.asarray(outputs["ingredients"])[0]
        nutrition_scaled = np.asarray(outputs["nutrition"])[0]
        scales = np.asarray(self.scaler["scales"], dtype=float)
        nutrition = np.maximum(nutrition_scaled * scales, 0.0)

        top_k = max(1, min(top_k, len(self.vocabulary)))
        order = np.argsort(probabilities)[::-1][:top_k]
        ingredients = [
            {
                "name": self.vocabulary[index],
                "confidence": float(probabilities[index]),
                "is_confident": bool(probabilities[index] >= confidence_threshold),
            }
            for index in order
        ]

        errors = np.asarray(self.uncertainty["absolute_error"], dtype=float)
        lower = np.maximum(nutrition - errors, 0.0)
        upper = nutrition + errors
        return {
            "ingredients": ingredients,
            "calories": float(nutrition[0]),
            "protein": float(nutrition[1]),
            "calories_interval": [float(lower[0]), float(upper[0])],
            "protein_interval": [float(lower[1]), float(upper[1])],
            "coverage": float(self.uncertainty.get("coverage", 0.95)),
        }
