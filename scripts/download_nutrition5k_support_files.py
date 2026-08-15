"""Download the small official Nutrition5k metadata and overhead split files."""

from __future__ import annotations

import argparse
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


BASE_URL = "https://storage.googleapis.com/nutrition5k_dataset/nutrition5k_dataset"
FILES = (
    "metadata/dish_metadata_cafe1.csv",
    "metadata/dish_metadata_cafe2.csv",
    "dish_ids/splits/depth_train_ids.txt",
    "dish_ids/splits/depth_test_ids.txt",
)


def _download(dataset_root: Path, relative_path: str, overwrite: bool) -> str:
    destination = dataset_root / Path(relative_path)
    if destination.is_file() and destination.stat().st_size > 0 and not overwrite:
        return "skipped"

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    request = Request(
        f"{BASE_URL}/{relative_path}", headers={"User-Agent": "SmartPlate/0.1"}
    )
    try:
        with urlopen(request, timeout=60) as response, temporary.open("wb") as stream:
            while chunk := response.read(1024 * 1024):
                stream.write(chunk)
        if not temporary.is_file() or temporary.stat().st_size == 0:
            raise OSError("downloaded file is empty")
        temporary.replace(destination)
    except (HTTPError, URLError, OSError):
        temporary.unlink(missing_ok=True)
        raise
    return "downloaded"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    try:
        for relative_path in FILES:
            status = _download(args.dataset_root, relative_path, args.overwrite)
            print(f"{status:10} {relative_path}")
    except (HTTPError, URLError, OSError) as exc:
        raise SystemExit(f"Official Nutrition5k download failed: {exc}") from exc


if __name__ == "__main__":
    main()
