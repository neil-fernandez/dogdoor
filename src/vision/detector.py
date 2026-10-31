"""Animal detection via Ultralytics YOLO (COCO-pretrained).

Detection is stage 1: it localises the animal so the classifier only ever sees a
crop (robust to clutter, distance, and background). YOLO11 is the mature,
edge-optimised choice; its COCO classes include dog/cat/person.
"""
from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image

from src.vision.config import DATA_CROPS, DATA_RAW, Roster, list_images

# Ultralytics prints emoji in its progress logs; make sure stdout is UTF-8.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# COCO classes that are animal-like (for "is there an animal in frame").
ANIMAL_CLASSES = {
    "person", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe",
}

_detector = None


def load_detector(model_name: str = "yolo11n.pt"):
    global _detector
    if _detector is None:
        from ultralytics import YOLO  # lazy: heavy import

        _detector = YOLO(model_name)
    return _detector


def detect(image: Image.Image, model_name: str = "yolo11n.pt", conf: float = 0.25) -> list[dict]:
    """Return detections [{name, conf, box:[x1,y1,x2,y2]}] sorted by confidence."""
    det = load_detector(model_name)
    results = det.predict(image, conf=conf, verbose=False)
    out = []
    for r in results:
        names = r.names
        for box in r.boxes:
            cls = int(box.cls[0])
            out.append(
                {
                    "name": names.get(cls, str(cls)),
                    "conf": float(box.conf[0]),
                    "box": [int(round(v)) for v in box.xyxy[0].tolist()],
                }
            )
    out.sort(key=lambda d: -d["conf"])
    return out


def largest_animal(detections: list[dict]) -> dict | None:
    """Most confident animal-like detection, else None."""
    for d in detections:
        if d["name"] in ANIMAL_CLASSES:
            return d
    return None


def crop(image: Image.Image, box) -> Image.Image:
    x1, y1, x2, y2 = box
    return image.crop((x1, y1, x2, y2))


def crop_dataset(
    roster: Roster,
    model_name: str = "yolo11n.pt",
    conf: float = 0.25,
    out_root: Path | None = None,
) -> dict:
    """Detect + crop every image in data/raw/<class>/ into data/processed/crops/<class>/."""
    out_root = Path(out_root) if out_root else DATA_CROPS
    stats = {}
    for cid in roster.class_ids:
        dst = out_root / cid
        dst.mkdir(parents=True, exist_ok=True)
        n = 0
        for p in list_images(DATA_RAW / cid):
            img = Image.open(p).convert("RGB")
            animal = largest_animal(detect(img, model_name=model_name, conf=conf))
            if animal is not None:
                crop(img, animal["box"]).save(dst / p.name)
                n += 1
        stats[cid] = n
    return stats
