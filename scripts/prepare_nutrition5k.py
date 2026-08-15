"""Build a clean, leakage-safe Nutrition5k manifest and EDA artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from smartplate.config import ProjectConfig, write_json
from smartplate.data import build_manifest


def _read_exclusions(path: Path | None) -> set[str]:
    if path is None or not path.is_file():
        return set()
    return {
        line.strip()
        for line in path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }


def _save_eda(frame: pd.DataFrame, stats: dict[str, object], output_dir: Path) -> None:
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].hist(frame["total_calories"], bins=30, color="#e76f51", edgecolor="white")
    axes[0].set(title="Calories distribution", xlabel="kcal", ylabel="Dishes")
    axes[1].hist(frame["total_protein"], bins=30, color="#2a9d8f", edgecolor="white")
    axes[1].set(title="Protein distribution", xlabel="grams", ylabel="Dishes")
    figure.tight_layout()
    figure.savefig(plots_dir / "nutrition_distributions.png", dpi=180)
    plt.close(figure)

    ingredients = list(stats["top_ingredients"])
    names = [str(item["name"]) for item in ingredients][::-1]
    counts = [int(item["dish_count"]) for item in ingredients][::-1]
    figure, axis = plt.subplots(figsize=(9, max(5, len(names) * 0.25)))
    axis.barh(names, counts, color="#457b9d")
    axis.set(title="Most frequent ingredient labels", xlabel="Dish count")
    figure.tight_layout()
    figure.savefig(plots_dir / "ingredient_frequencies.png", dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--max-ingredients", type=int, default=50)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--generated-test-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--exclude-ids", type=Path)
    parser.add_argument(
        "--skip-image-verification",
        action="store_true",
        help="Skip PIL decoding checks. Faster, but not recommended for the final dataset.",
    )
    args = parser.parse_args()

    config = ProjectConfig(
        image_size=args.image_size,
        batch_size=args.batch_size,
        max_ingredients=args.max_ingredients,
        validation_fraction=args.validation_fraction,
        seed=args.seed,
    )
    frame, vocabulary, stats = build_manifest(
        args.dataset_root,
        max_ingredients=args.max_ingredients,
        validation_fraction=args.validation_fraction,
        generated_test_fraction=args.generated_test_fraction,
        seed=args.seed,
        verify_images=not args.skip_image_verification,
        excluded_dish_ids=_read_exclusions(args.exclude_ids),
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output_dir / "manifest.csv", index=False)
    write_json(args.output_dir / "ingredient_vocabulary.json", vocabulary)
    write_json(args.output_dir / "dataset_statistics.json", stats)
    write_json(args.output_dir / "project_config.json", config.to_dict())
    _save_eda(frame, stats, args.output_dir)

    review_reasons = pd.Series("", index=frame.index, dtype="object")
    review_reasons = review_reasons.mask(
        frame["total_calories"] > 2000, "calories_above_2000"
    )
    review_reasons = review_reasons.mask(
        (frame["total_mass"] > 0)
        & ((frame["total_calories"] / frame["total_mass"]) > 10),
        review_reasons.str.cat(pd.Series("calories_per_gram_above_10", index=frame.index), sep="; ").str.strip("; "),
    )
    review_frame = frame.loc[review_reasons.str.len() > 0].copy()
    review_frame["review_reason"] = review_reasons.loc[review_frame.index]
    review_frame.to_csv(args.output_dir / "manual_review_candidates.csv", index=False)

    print(f"Prepared {len(frame):,} dish images in {args.output_dir.resolve()}")
    print(f"Splits: {stats['split_counts']}")
    print(f"Official test split used: {stats['used_official_test_split']}")
    print(f"Manual review candidates: {len(review_frame):,}")


if __name__ == "__main__":
    main()
