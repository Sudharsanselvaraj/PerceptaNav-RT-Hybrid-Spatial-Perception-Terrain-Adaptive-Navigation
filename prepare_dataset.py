"""
Dataset Preparation Utility
Splits raw images into train/val for YOLO classification.
Usage: python prepare_dataset.py --source /path/to/raw --split 0.8
"""

import os
import shutil
import random
import argparse
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("PrepareDataset")

CLASSES    = ["rock", "soil", "sand", "gravel"]
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def prepare(source: str, dest: str = "mars-data", split: float = 0.8, seed: int = 42):
    random.seed(seed)
    source_path = Path(source)

    for cls in CLASSES:
        cls_dir = source_path / cls
        if not cls_dir.exists():
            log.warning(f"Class folder missing: {cls_dir} — skipping")
            continue

        images = [p for p in cls_dir.iterdir()
                  if p.suffix.lower() in EXTENSIONS]
        if not images:
            log.warning(f"No images in {cls_dir}")
            continue

        random.shuffle(images)
        n_train = int(len(images) * split)
        splits  = {"train": images[:n_train], "val": images[n_train:]}

        for split_name, files in splits.items():
            out_dir = Path(dest) / split_name / cls
            out_dir.mkdir(parents=True, exist_ok=True)
            for f in files:
                shutil.copy2(f, out_dir / f.name)
            log.info(f"  {cls}/{split_name}: {len(files)} images → {out_dir}")

    log.info(f"Dataset ready at: {dest}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True,
                        help="Root folder with class sub-folders")
    parser.add_argument("--dest",   default="mars-data")
    parser.add_argument("--split",  type=float, default=0.8)
    parser.add_argument("--seed",   type=int,   default=42)
    args = parser.parse_args()
    prepare(args.source, args.dest, args.split, args.seed)
