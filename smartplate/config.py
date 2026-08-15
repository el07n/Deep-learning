"""Project configuration and JSON artifact helpers."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ProjectConfig:
    """Training defaults chosen for EfficientNetV2B0 and Nutrition5k."""

    image_size: int = 224
    batch_size: int = 16
    max_ingredients: int = 50
    validation_fraction: float = 0.15
    seed: int = 42
    dropout_rate: float = 0.30
    head_learning_rate: float = 1e-3
    fine_tune_learning_rate: float = 1e-5
    fine_tune_fraction: float = 0.25
    recognition_loss_weight: float = 1.0
    nutrition_loss_weight: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "ProjectConfig":
        allowed = cls.__dataclass_fields__.keys()
        return cls(**{key: value for key, value in values.items() if key in allowed})


def read_json(path: str | Path) -> dict[str, Any] | list[Any]:
    with Path(path).open("r", encoding="utf-8") as stream:
        return json.load(stream)


def write_json(path: str | Path, value: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False)
