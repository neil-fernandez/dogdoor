"""Roster-driven image dataset and transforms."""
from __future__ import annotations

from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T

from src.vision._pil import open_image
from src.vision.config import Roster, list_images

_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]


class RosterDataset(Dataset):
    """Dataset over data/raw/<class_id>/ folders, labels derived from the roster."""

    def __init__(self, root: Path, roster: Roster, transform=None):
        self.root = Path(root)
        self.roster = roster
        self.transform = transform
        self.samples: list[tuple[Path, int]] = []
        for cid, idx in roster.id_to_index.items():
            for p in list_images(self.root / cid):
                self.samples.append((p, idx))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i):
        path, label = self.samples[i]
        img = open_image(path)
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def class_counts(self) -> dict:
        counts = {cid: 0 for cid in self.roster.class_ids}
        for _, label in self.samples:
            counts[self.roster.class_ids[label]] += 1
        return counts


def build_transforms(img_size: int = 224, augment: bool = True):
    if augment:
        return T.Compose(
            [
                T.Resize((img_size + 32, img_size + 32)),
                T.RandomResizedCrop(img_size, scale=(0.7, 1.0)),
                T.RandomHorizontalFlip(),
                T.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.05),
                T.ToTensor(),
                T.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
            ]
        )
    return T.Compose(
        [
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
        ]
    )
