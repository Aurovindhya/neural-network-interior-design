"""
download_dataset.py

Downloads a curated set of open-licensed interior design images
from Unsplash Source (free, no API key required) and organizes
them into the expected folder structure:

    data/
    └── images/
        ├── mid_century_modern/
        ├── scandinavian/
        ├── industrial/
        ├── bohemian/
        └── minimalist/

Usage:
    python data/scripts/download_dataset.py
    python data/scripts/download_dataset.py --output data/images --per-class 30
"""

import argparse
import os
import time
import requests
from pathlib import Path
from tqdm import tqdm

# Unsplash Source API — free, no key needed, returns random photos by keyword
UNSPLASH_URL = "https://source.unsplash.com/800x600/?{query}"

STYLES = {
    "mid_century_modern": [
        "mid century modern living room",
        "retro interior design",
        "mid century modern furniture",
        "1960s interior design",
        "danish modern furniture",
    ],
    "scandinavian": [
        "scandinavian interior design",
        "nordic living room",
        "hygge interior",
        "white minimal scandinavian",
        "swedish interior design",
    ],
    "industrial": [
        "industrial interior design",
        "loft industrial apartment",
        "exposed brick interior",
        "industrial style kitchen",
        "concrete industrial living room",
    ],
    "bohemian": [
        "bohemian interior design",
        "boho chic living room",
        "eclectic bohemian bedroom",
        "moroccan interior style",
        "maximalist boho interior",
    ],
    "minimalist": [
        "minimalist interior design",
        "clean minimal living room",
        "japandi interior",
        "white minimal bedroom",
        "zen interior design",
    ],
}


def download_image(url: str, dest: Path, retries: int = 3) -> bool:
    """Download a single image with retry logic."""
    for attempt in range(retries):
        try:
            resp = requests.get(url, timeout=15, allow_redirects=True)
            if resp.status_code == 200 and resp.headers.get("content-type", "").startswith("image"):
                dest.write_bytes(resp.content)
                return True
        except requests.RequestException:
            pass
        time.sleep(1.5 * (attempt + 1))
    return False


def download_dataset(output_dir: str = "data/images", per_class: int = 30):
    output_path = Path(output_dir)

    total = len(STYLES) * per_class
    print(f"Downloading {total} images ({per_class} per class) to {output_path}/\n")

    for style, queries in STYLES.items():
        style_dir = output_path / style
        style_dir.mkdir(parents=True, exist_ok=True)

        existing = list(style_dir.glob("*.jpg"))
        if len(existing) >= per_class:
            print(f"  {style}: already has {len(existing)} images, skipping.")
            continue

        print(f"  Downloading {style}...")
        downloaded = len(existing)
        idx = downloaded

        with tqdm(total=per_class, initial=downloaded, desc=f"  {style}", ncols=70) as pbar:
            while downloaded < per_class:
                query = queries[idx % len(queries)]
                url = UNSPLASH_URL.format(query=query.replace(" ", "+"))
                dest = style_dir / f"{style}_{downloaded:03d}.jpg"

                if download_image(url, dest):
                    downloaded += 1
                    pbar.update(1)
                else:
                    print(f"\n    Warning: failed to download image {idx}, skipping.")

                idx += 1
                time.sleep(0.3)  # be polite to Unsplash

    print(f"\nDataset ready at {output_path}/")
    print("\nClass counts:")
    for style in STYLES:
        count = len(list((output_path / style).glob("*.jpg")))
        print(f"  {style}: {count} images")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download interior design dataset")
    parser.add_argument("--output", default="data/images", help="Output directory")
    parser.add_argument("--per-class", type=int, default=30, help="Images per class")
    args = parser.parse_args()

    download_dataset(args.output, args.per_class)
