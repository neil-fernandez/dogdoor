"""End-to-end inference: detect -> crop -> identify (embedding preferred).

Prefer the ArcFace embedding + gallery (open-set: returns "unknown" below a
threshold). Fall back to the softmax classifier if no embedding model exists.
"""
from __future__ import annotations

import json

import torch
import torch.nn.functional as F
from PIL import Image

from src.vision import detector
from src.vision.config import RUNS_DIR, load_roster
from src.vision.dataset import build_transforms
from src.vision.embedding import build_embedding_model
from src.vision.model import build_model

_classifier_cache = None
_embedding_cache = None


def _latest(pattern, exclude=None):
    if not RUNS_DIR.exists():
        return None
    runs = [p.parent for p in RUNS_DIR.glob(pattern)]
    if exclude:
        runs = [r for r in runs if exclude not in r.name]
    return sorted(runs)[-1] if runs else None


def load_classifier():
    global _classifier_cache
    if _classifier_cache is not None:
        return _classifier_cache
    run = _latest("*/best.pt", exclude="_arcface")
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
    _classifier_cache = (model, classes, device, img_size)
    return _classifier_cache


def load_embedding():
    global _embedding_cache
    if _embedding_cache is not None:
        return _embedding_cache
    run = _latest("*_arcface/best.pt")
    if run is None:
        return None
    cfg = json.loads((run / "config.json").read_text())
    g = torch.load(run / "gallery.pt", map_location="cpu")
    classes = cfg["classes"]
    backbone = cfg["backbone"]
    embed_dim = cfg.get("embed_dim", 128)
    img_size = cfg.get("img_size", 224)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_embedding_model(backbone, embed_dim, freeze=False, pretrained=False)
    model.load_state_dict(torch.load(run / "best.pt", map_location=device))
    model.to(device).eval()
    _embedding_cache = (model, g["gallery"], g.get("threshold"), classes, device, img_size)
    return _embedding_cache


def classify(image: Image.Image, top_k: int = 3):
    """Softmax classifier top-k; [] if no classifier exists."""
    cp = load_classifier()
    if cp is None:
        return []
    model, classes, device, img_size = cp
    x = build_transforms(img_size, augment=False)(image).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x)[0]
    probs = torch.softmax(logits, dim=0)
    top = probs.topk(min(top_k, len(classes)))
    return [(classes[i], float(p)) for i, p in zip(top.indices.tolist(), top.values.tolist())]


def identify(image: Image.Image, top_k: int = 3):
    """Embedding + gallery nearest-neighbour; None if no embedding model."""
    cp = load_embedding()
    if cp is None:
        return None
    model, gallery, threshold, classes, device, img_size = cp
    x = build_transforms(img_size, augment=False)(image).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = F.normalize(model(x), dim=1)
    sims = emb @ gallery.t()  # [1, C]
    top = sims[0].topk(min(top_k, len(classes)))
    preds = [(classes[i], float(s)) for i, s in zip(top.indices.tolist(), top.values.tolist())]
    max_sim = float(sims.max())
    return {
        "predictions": preds,
        "max_sim": max_sim,
        "threshold": threshold,
        "is_unknown": threshold is not None and max_sim < threshold,
    }


def predict(image: Image.Image, top_k: int = 3) -> dict:
    detections = detector.detect(image)
    animal = detector.largest_animal(detections)
    result = {
        "detections": detections,
        "animal": animal,
        "predictions": [],
        "method": None,
        "is_unknown": None,
        "threshold": None,
    }
    if animal is not None:
        crop_img = detector.crop(image, animal["box"])
        emb = identify(crop_img, top_k)
        if emb is not None:
            result["method"] = "embedding"
            result["predictions"] = emb["predictions"]
            result["is_unknown"] = emb["is_unknown"]
            result["threshold"] = emb["threshold"]
        else:
            result["method"] = "classifier"
            result["predictions"] = classify(crop_img, top_k)
    return result


def invalidate_cache():
    global _classifier_cache, _embedding_cache
    _classifier_cache = None
    _embedding_cache = None
