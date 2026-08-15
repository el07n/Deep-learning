"""Nutrition5k metadata parsing, cleaning, splitting, and manifest creation."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from PIL import Image, UnidentifiedImageError


CORE_COLUMNS = (
    "dish_id",
    "total_calories",
    "total_mass",
    "total_fat",
    "total_carb",
    "total_protein",
)
INGREDIENT_WIDTH = 7
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


class MetadataFormatError(ValueError):
    """Raised when a Nutrition5k metadata row cannot be interpreted safely."""


def _as_float(value: str, field: str, dish_id: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise MetadataFormatError(
            f"Invalid {field!r} value for {dish_id}: {value!r}"
        ) from exc
    if not np.isfinite(parsed):
        raise MetadataFormatError(f"Non-finite {field!r} value for {dish_id}")
    return parsed


def _ingredient_offset(row: Sequence[str]) -> int:
    """Handle both documented rows (with num_ingrs) and released rows without it."""

    remainder_without_count = len(row) - len(CORE_COLUMNS)
    if remainder_without_count >= 0 and remainder_without_count % INGREDIENT_WIDTH == 0:
        return len(CORE_COLUMNS)

    remainder_with_count = len(row) - len(CORE_COLUMNS) - 1
    if remainder_with_count >= 0 and remainder_with_count % INGREDIENT_WIDTH == 0:
        return len(CORE_COLUMNS) + 1

    raise MetadataFormatError(
        f"Expected ingredient fields in groups of {INGREDIENT_WIDTH}; got {len(row)} columns"
    )


def parse_dish_metadata(path: str | Path) -> pd.DataFrame:
    """Parse one variable-width Nutrition5k dish metadata CSV.

    The official documentation includes a ``num_ingrs`` field, while some released
    files omit it. This parser accepts both variants and normalizes ingredients to a
    JSON list so the resulting frame can be saved safely as CSV.
    """

    records: list[dict[str, object]] = []
    source = Path(path)
    with source.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream)
        for line_number, raw_row in enumerate(reader, start=1):
            row = [item.strip() for item in raw_row]
            if not row or not any(row):
                continue
            if row[0].lower() in {"dish_id", "id"}:
                continue
            if len(row) < len(CORE_COLUMNS):
                raise MetadataFormatError(
                    f"{source}:{line_number} has only {len(row)} columns"
                )

            dish_id = row[0]
            offset = _ingredient_offset(row)
            ingredient_names: list[str] = []
            ingredient_ids: list[str] = []
            for index in range(offset, len(row), INGREDIENT_WIDTH):
                group = row[index : index + INGREDIENT_WIDTH]
                if len(group) != INGREDIENT_WIDTH:
                    raise MetadataFormatError(
                        f"{source}:{line_number} contains an incomplete ingredient group"
                    )
                ingredient_id, ingredient_name = group[0], group[1]
                if ingredient_id:
                    ingredient_ids.append(ingredient_id)
                if ingredient_name:
                    ingredient_names.append(ingredient_name.strip().lower())

            records.append(
                {
                    "dish_id": dish_id,
                    "total_calories": _as_float(row[1], "total_calories", dish_id),
                    "total_mass": _as_float(row[2], "total_mass", dish_id),
                    "total_fat": _as_float(row[3], "total_fat", dish_id),
                    "total_carb": _as_float(row[4], "total_carb", dish_id),
                    "total_protein": _as_float(row[5], "total_protein", dish_id),
                    "ingredients": json.dumps(ingredient_names, ensure_ascii=False),
                    "ingredient_ids": json.dumps(ingredient_ids),
                    "source_file": source.name,
                }
            )

    return pd.DataFrame.from_records(records)


def load_all_dish_metadata(dataset_root: str | Path) -> pd.DataFrame:
    root = Path(dataset_root)
    metadata_dir = root / "metadata"
    paths = sorted(metadata_dir.glob("dish_metadata*.csv"))
    if not paths:
        raise FileNotFoundError(
            f"No dish_metadata*.csv files found under {metadata_dir}"
        )
    frames = [parse_dish_metadata(path) for path in paths]
    combined = pd.concat(frames, ignore_index=True)
    return combined.drop_duplicates(subset="dish_id", keep="first").reset_index(drop=True)


def _read_id_file(path: Path) -> set[str]:
    identifiers: set[str] = set()
    with path.open("r", encoding="utf-8-sig") as stream:
        for line in stream:
            value = line.strip().split(",")[0]
            if value.startswith("dish_"):
                identifiers.add(value)
    return identifiers


def load_official_split_ids(
    dataset_root: str | Path, split_family: str = "depth"
) -> tuple[set[str], set[str]]:
    """Load one official split family.

    Nutrition5k publishes separate ``rgb_*`` splits for side-angle video frames
    and ``depth_*`` splits for overhead RGB-D captures. The project uses the
    overhead RGB image, so ``depth`` is the safe default.
    """

    split_dir = Path(dataset_root) / "dish_ids" / "splits"
    if not split_dir.exists():
        return set(), set()

    candidates = [path for path in sorted(split_dir.rglob("*")) if path.is_file()]
    preferred = [
        path for path in candidates if path.name.lower().startswith(f"{split_family.lower()}_")
    ]
    selected = preferred or candidates

    train_ids: set[str] = set()
    test_ids: set[str] = set()
    for path in selected:
        name = path.name.lower()
        if "train" in name:
            train_ids.update(_read_id_file(path))
        elif "test" in name:
            test_ids.update(_read_id_file(path))

    overlap = train_ids & test_ids
    if overlap:
        sample = sorted(overlap)[:3]
        raise ValueError(f"Official split files overlap for dish IDs such as {sample}")
    return train_ids, test_ids


def _stable_fraction(identifier: str, seed: int) -> float:
    digest = hashlib.sha256(f"{seed}:{identifier}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / float(2**64)


def assign_splits(
    dish_ids: Iterable[str],
    official_train_ids: set[str],
    official_test_ids: set[str],
    validation_fraction: float,
    seed: int,
    generated_test_fraction: float = 0.15,
) -> list[str]:
    """Assign splits at dish level, preserving the official test split."""

    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1")
    if not 0 < generated_test_fraction < 1:
        raise ValueError("generated_test_fraction must be between 0 and 1")

    results: list[str] = []
    for dish_id in dish_ids:
        if dish_id in official_test_ids:
            results.append("test")
            continue
        fraction = _stable_fraction(dish_id, seed)
        if not official_test_ids and fraction < generated_test_fraction:
            results.append("test")
            continue
        # Validation is created only from train candidates and remains dish-level.
        lower_bound = generated_test_fraction if not official_test_ids else 0.0
        validation_cutoff = lower_bound + validation_fraction
        results.append("validation" if fraction < validation_cutoff else "train")
    return results


def find_overhead_rgb_image(dataset_root: str | Path, dish_id: str) -> Path | None:
    """Resolve one non-depth overhead RGB image for a dish."""

    dish_dir = Path(dataset_root) / "imagery" / "realsense_overhead" / dish_id
    if not dish_dir.exists():
        return None

    preferred_names = (
        "rgb.png",
        "rgb.jpg",
        "rgb.jpeg",
        f"{dish_id}.png",
        f"{dish_id}.jpg",
    )
    for name in preferred_names:
        candidate = dish_dir / name
        if candidate.is_file():
            return candidate.resolve()

    candidates = [
        path
        for path in dish_dir.rglob("*")
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
        and "depth" not in path.name.lower()
    ]
    rgb_named = [path for path in candidates if "rgb" in path.name.lower()]
    selected = sorted(rgb_named or candidates)
    return selected[0].resolve() if selected else None


def decode_ingredients(value: str | Sequence[str]) -> list[str]:
    if isinstance(value, str):
        decoded = json.loads(value)
    else:
        decoded = list(value)
    return [str(item).strip().lower() for item in decoded if str(item).strip()]


def build_ingredient_vocabulary(
    ingredient_values: Iterable[str | Sequence[str]], max_ingredients: int
) -> list[str]:
    counter: Counter[str] = Counter()
    for value in ingredient_values:
        counter.update(set(decode_ingredients(value)))
    return [name for name, _ in counter.most_common(max_ingredients)]


def encode_ingredients(value: str | Sequence[str], vocabulary: Sequence[str]) -> list[int]:
    present = set(decode_ingredients(value))
    return [index for index, name in enumerate(vocabulary) if name in present]


def _is_decodable_image(path: str | Path) -> bool:
    try:
        with Image.open(path) as image:
            image.verify()
        return True
    except (OSError, UnidentifiedImageError, ValueError):
        return False


def clean_manifest(
    frame: pd.DataFrame, verify_images: bool = True
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Apply auditable cleaning rules and return exclusion counts."""

    current = frame.copy()
    report: dict[str, int] = {"input_rows": int(len(current))}

    before = len(current)
    current = current.drop_duplicates(subset="dish_id", keep="first")
    report["duplicate_dish_ids_removed"] = before - len(current)

    numeric_columns = [
        "total_calories",
        "total_mass",
        "total_fat",
        "total_carb",
        "total_protein",
    ]
    before = len(current)
    current = current.dropna(subset=numeric_columns)
    report["missing_values_removed"] = before - len(current)

    before = len(current)
    current = current.dropna(subset=["image_path"])
    report["missing_image_paths_removed"] = before - len(current)

    before = len(current)
    nonnegative = (current[numeric_columns] >= 0).all(axis=1)
    positive_targets = (current["total_calories"] > 0) & (current["total_mass"] > 0)
    current = current.loc[nonnegative & positive_targets]
    report["invalid_nutrition_removed"] = before - len(current)

    before = len(current)
    exists = current["image_path"].map(lambda value: Path(str(value)).is_file())
    current = current.loc[exists]
    report["missing_images_removed"] = before - len(current)

    if verify_images:
        before = len(current)
        decodable = current["image_path"].map(_is_decodable_image)
        current = current.loc[decodable]
        report["corrupt_images_removed"] = before - len(current)
    report["output_rows"] = int(len(current))
    return current.reset_index(drop=True), report


def dataset_statistics(frame: pd.DataFrame, vocabulary: Sequence[str]) -> dict[str, object]:
    split_counts = frame["split"].value_counts().to_dict()
    ingredient_counter: Counter[str] = Counter()
    for value in frame["ingredients"]:
        ingredient_counter.update(set(decode_ingredients(value)))
    return {
        "rows": int(len(frame)),
        "unique_dishes": int(frame["dish_id"].nunique()),
        "split_counts": {str(key): int(value) for key, value in split_counts.items()},
        "nutrition": {
            column: {
                "mean": float(frame[column].mean()),
                "std": float(frame[column].std(ddof=0)),
                "median": float(frame[column].median()),
                "min": float(frame[column].min()),
                "max": float(frame[column].max()),
            }
            for column in ("total_calories", "total_protein")
        },
        "ingredient_vocabulary_size": len(vocabulary),
        "top_ingredients": [
            {"name": name, "dish_count": int(count)}
            for name, count in ingredient_counter.most_common(len(vocabulary))
        ],
    }


def build_manifest(
    dataset_root: str | Path,
    max_ingredients: int = 50,
    validation_fraction: float = 0.15,
    seed: int = 42,
    generated_test_fraction: float = 0.15,
    verify_images: bool = True,
    excluded_dish_ids: set[str] | None = None,
) -> tuple[pd.DataFrame, list[str], dict[str, object]]:
    """Build a cleaned, dish-level manifest using overhead RGB images."""

    root = Path(dataset_root).resolve()
    frame = load_all_dish_metadata(root)
    frame["image_path"] = frame["dish_id"].map(
        lambda dish_id: find_overhead_rgb_image(root, str(dish_id))
    )
    manual_exclusions = excluded_dish_ids or set()
    if manual_exclusions:
        frame = frame.loc[~frame["dish_id"].isin(manual_exclusions)].copy()
    train_ids, test_ids = load_official_split_ids(root, split_family="depth")
    frame["split"] = assign_splits(
        frame["dish_id"],
        train_ids,
        test_ids,
        validation_fraction,
        seed,
        generated_test_fraction,
    )
    frame, cleaning_report = clean_manifest(frame, verify_images=verify_images)
    cleaning_report["manually_excluded"] = int(len(manual_exclusions))

    training_values = frame.loc[frame["split"] == "train", "ingredients"]
    vocabulary = build_ingredient_vocabulary(training_values, max_ingredients)
    if not vocabulary:
        raise ValueError("No ingredient labels remained after cleaning")
    frame["ingredient_indices"] = frame["ingredients"].map(
        lambda value: json.dumps(encode_ingredients(value, vocabulary))
    )

    stats = dataset_statistics(frame, vocabulary)
    stats["cleaning_report"] = cleaning_report
    stats["used_official_test_split"] = bool(test_ids)
    stats["used_official_train_split"] = bool(train_ids)
    return frame, vocabulary, stats
