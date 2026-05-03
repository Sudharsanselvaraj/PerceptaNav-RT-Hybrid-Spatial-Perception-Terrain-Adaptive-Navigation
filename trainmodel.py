"""
Train YOLOv8 classifier on Mars terrain dataset.
Classes: rock, soil, sand, gravel
"""

import os
import sys
import logging
from pathlib import Path
from ultralytics import YOLO

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("TrainModel")

DATA_DIR    = "mars-data"
BASE_MODEL  = "yolov8n-cls.pt"
EPOCHS      = 50
IMG_SIZE    = 224
BATCH       = 16
PATIENCE    = 10          # early stopping
WORKERS     = 4
DEVICE      = "cpu"       # change to "mps" on Apple Silicon, "0" for CUDA


def validate_dataset(data_dir: str):
    required = ["train", "val"]
    classes  = ["rock", "soil", "sand", "gravel"]
    ok = True
    for split in required:
        for cls in classes:
            p = Path(data_dir) / split / cls
            if not p.exists():
                log.warning(f"Missing: {p}  — creating empty dir")
                p.mkdir(parents=True, exist_ok=True)
            count = len(list(p.glob("*.jpg")) + list(p.glob("*.png")))
            log.info(f"  {split}/{cls}: {count} images")
            if count == 0:
                log.warning(f"  ⚠ No images in {p}")
                ok = False
    return ok


def train():
    log.info("═══ Mars Terrain Classifier Training ═══")

    if not validate_dataset(DATA_DIR):
        log.error("Dataset incomplete. Add images before training.")
        sys.exit(1)

    model = YOLO(BASE_MODEL)
    log.info(f"Base model loaded: {BASE_MODEL}")

    results = model.train(
        data      = DATA_DIR,
        epochs    = EPOCHS,
        imgsz     = IMG_SIZE,
        batch     = BATCH,
        patience  = PATIENCE,
        workers   = WORKERS,
        device    = DEVICE,
        project   = "runs/classify",
        name      = "train",
        exist_ok  = True,
        plots     = True,
        save      = True,
        verbose   = True,
    )

    best_weights = Path("runs/classify/train/weights/best.pt")
    if best_weights.exists():
        log.info(f"✓ Best weights saved at: {best_weights}")
    else:
        log.error("Training completed but best.pt not found.")

    # ── Quick validation ──
    log.info("Running validation on val set...")
    val_results = model.val()
    log.info(f"Top-1 accuracy: {val_results.top1:.4f}")
    log.info(f"Top-5 accuracy: {val_results.top5:.4f}")

    return results


if __name__ == "__main__":
    train()
