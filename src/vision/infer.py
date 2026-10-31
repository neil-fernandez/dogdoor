"""End-to-end inference: detect -> crop -> classify."""
from __future__ import annotations

import json

import torch
from PIL import Image

from src.vision import detector
from src.vision.config import RUNS_DIR, load_roster
from src.vision.dataset import build_transforms
from src.vision.model import build_model

_checkpoint = None  # (model, classes, device, img_size)


def _latest_run():
    if not RUNS_DIR.exists():
        return None
    dirs = sorted(RUNS_DIR.glob("*/best.pt"))
    return dirs[-1].parent if dirs else None


def load_latest_model():
    """Load the most recent trained classifier, or None if no training has run yet."""
    global _checkpoint
    if _checkpoint is not None:
        return _checkpoint
    run = _latest_run()
    if run is None:
        return None
    cfg = json.loads((run / "config.json").read_text())
    classes = cfg["classes"]
    backbone = cfg["backbone"]
    img_size = cfg.get("img_size", 224)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(backbone, len(classes), freeze=False, pretrained=False)
    model.load_state_dict(torch.load(run / "best.pt", map_location=device))
    model.to(device).eval()
    _checkpoint = (model, classes, device, img_size)
    return _checkpoint


def classify(image: Image.Image, top_k: int = 3) -> list[tuple[str, float]]:
    """Classify a crop with the latest trained model; [] if no model exists."""
    cp = load_latest_model()
    if cp is None:
        return []
    model, classes, device, img_size = cp
    x = build_transforms(img_size, augment=False)(image).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x)[0]
    probs = torch.softmax(logits, dim=0)
    top = probs.topk(min(top_k, len(classes)))
    return [(classes[i], float(p)) for i, p in zip(top.indices.tolist(), top.values.tolist())]


def predict(image: Image.Image, top_k: int = 3) -> dict:
    """Full pipeline for the playground: detect -> crop -> classify."""
    detections = detector.detect(image)
    animal = detector.largest_animal(detections)
    result = {"detections": detections, "animal": animal, "predictions": []}
    if animal is not None:
        result["predictions"] = classify(detector.crop(image, animal["box"]), top_k=top_k)
    return result


def invalidate_cache():
    """Clear the cached model so a newly trained model is picked up."""
    global _checkpoint
    _checkpoint = None
