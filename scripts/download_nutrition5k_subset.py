"""Download only Nutrition5k overhead RGB images listed by the depth split."""

from __future__ import annotations

import argparse
import random
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from PIL import Image


BASE_URL = (
    "https://storage.googleapis.com/nutrition5k_dataset/"
    "nutrition5k_dataset/imagery/realsense_overhead"
)
_print_lock = threading.Lock()


def _read_ids(path: Path) -> list[str]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing official split file: {path}")
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip().startswith("dish_")
    ]


def _valid_image(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        with Image.open(path) as image:
            image.verify()
        return True
    except OSError:
        return False


def _download(dataset_root: Path, dish_id: str) -> tuple[str, str]:
    destination = (
        dataset_root / "imagery" / "realsense_overhead" / dish_id / "rgb.png"
    )
    if _valid_image(destination):
        return dish_id, "skipped"

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".png.part")
    request = Request(f"{BASE_URL}/{dish_id}/rgb.png", headers={"User-Agent": "SmartPlate/0.1"})
    try:
        with urlopen(request, timeout=60) as response, temporary.open("wb") as stream:
            while chunk := response.read(1024 * 1024):
                stream.write(chunk)
        if not _valid_image(temporary):
            raise OSError("downloaded file is not a valid image")
        temporary.replace(destination)
        return dish_id, "downloaded"
    except (HTTPError, URLError, OSError) as exc:
        temporary.unlink(missing_ok=True)
        return dish_id, f"error: {exc}"


def _sample(values: list[str], maximum: int, seed: int) -> list[str]:
    if maximum <= 0 or maximum >= len(values):
        return values
    generator = random.Random(seed)
    return sorted(generator.sample(values, maximum))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument(
        "--max-train",
        type=int,
        default=0,
        help="0 downloads all official depth-train RGB images.",
    )
    parser.add_argument(
        "--max-test",
        type=int,
        default=0,
        help="0 downloads all official depth-test RGB images.",
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    split_dir = args.dataset_root / "dish_ids" / "splits"
    train_ids = _sample(
        _read_ids(split_dir / "depth_train_ids.txt"), args.max_train, args.seed
    )
    test_ids = _sample(
        _read_ids(split_dir / "depth_test_ids.txt"), args.max_test, args.seed + 1
    )
    dish_ids = list(dict.fromkeys([*train_ids, *test_ids]))
    counters = {"downloaded": 0, "skipped": 0, "errors": 0}
    failures: list[tuple[str, str]] = []

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = {
            executor.submit(_download, args.dataset_root, dish_id): dish_id
            for dish_id in dish_ids
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            dish_id, status = future.result()
            if status in counters:
                counters[status] += 1
            else:
                counters["errors"] += 1
                failures.append((dish_id, status))
            if completed % 100 == 0 or completed == len(futures):
                with _print_lock:
                    print(f"{completed:,}/{len(futures):,} images processed")

    print(counters)
    if failures:
        failure_path = args.dataset_root / "download_failures.txt"
        failure_path.write_text(
            "\n".join(f"{dish_id}\t{status}" for dish_id, status in failures),
            encoding="utf-8",
        )
        raise SystemExit(f"{len(failures)} downloads failed; see {failure_path}")


if __name__ == "__main__":
    main()

