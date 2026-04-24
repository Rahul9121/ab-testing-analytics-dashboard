from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Download a public Kaggle A/B testing dataset.")
    parser.add_argument(
        "--dataset",
        default="rohankulakarni/ab-test-marketing-campaign-dataset",
        help="Kaggle dataset slug in the form owner/dataset-name",
    )
    parser.add_argument(
        "--output-dir",
        default="data/raw",
        help="Output directory for downloaded and unzipped files",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    command = [
        "kaggle",
        "datasets",
        "download",
        "-d",
        args.dataset,
        "-p",
        str(output_dir),
        "--unzip",
    ]

    print(f"Downloading {args.dataset} into {output_dir} ...")
    result = subprocess.run(command, check=False)
    if result.returncode != 0:
        print(
            "Download failed. Make sure Kaggle API is installed and configured with kaggle.json credentials.",
            file=sys.stderr,
        )
        return result.returncode

    print("Download completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

