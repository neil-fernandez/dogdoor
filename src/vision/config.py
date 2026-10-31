"""Project paths, roster loading, and class derivation.

This module is intentionally dependency-free (only PyYAML) so the studio can
import it without torch/torchvision installed.
"""
from __future__ import annotations

from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROSTER = PROJECT_ROOT / "config" / "roster.yaml"
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
DATA_CROPS = DATA_PROCESSED / "crops"
RUNS_DIR = PROJECT_ROOT / "runs"
MODELS_DIR = PROJECT_ROOT / "models"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".heic", ".heif"}


def list_images(folder: Path) -> list[Path]:
    """Return image files under `folder`, sorted and deterministic."""
    if not folder.exists():
        return []
    return sorted(p for p in folder.rglob("*") if p.suffix.lower() in IMAGE_EXTS and p.is_file())


class Roster:
    """The enrolled-animals config, from which the class list is derived."""

    def __init__(self, data: dict):
        self.version = data.get("version", 1)
        neg = data.get("negative_class") or {"id": "other", "name": "Other"}
        self.negative_id = neg["id"]
        self.negative_name = neg.get("name", self.negative_id)
        self.animals = [
            {
                "id": a["id"],
                "name": a.get("name", a["id"]),
                "description": a.get("description", ""),
            }
            for a in data.get("animals", [])
        ]
        # Class order is: animals (roster order), then the negative class.
        self.class_ids = [a["id"] for a in self.animals] + [self.negative_id]
        self.class_names = [a["name"] for a in self.animals] + [self.negative_name]
        self.id_to_index = {cid: i for i, cid in enumerate(self.class_ids)}
        self.index_to_name = {i: name for i, name in enumerate(self.class_names)}

    @property
    def num_classes(self) -> int:
        return len(self.class_ids)


def load_roster(path: Path | str | None = None) -> Roster:
    p = Path(path) if path else DEFAULT_ROSTER
    with open(p, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return Roster(data)
