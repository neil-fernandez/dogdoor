"""Crop the dataset with the detector (detect -> crop) before training.

Usage:
    python -m src.vision.preprocess --model yolo11n.pt
"""
from __future__ import annotations

import argparse

from src.vision import detector
from src.vision.config import load_roster


def main():
    ap = argparse.ArgumentParser(description="Detect + crop the dataset for training.")
    ap.add_argument("--model", default="yolo11n.pt")
    ap.add_argument("--conf", type=float, default=0.25)
    args = ap.parse_args()

    roster = load_roster()
    stats = detector.crop_dataset(roster, model_name=args.model, conf=args.conf)
    for cid, n in stats.items():
        print(f"{cid}: {n} crops")
    print("Done. Train on crops with:")
    print("  python -m src.vision.train --data-root data/processed/crops")


if __name__ == "__main__":
    main()
