"""Metric learning: ArcFace embeddings for open-set animal identification.

Unlike the fixed-class classifier, this learns a 128-dim embedding where same-animal
images are close and different animals are far. Identification = nearest neighbour
against an enrolled gallery (open-set: "not in gallery" falls out via a distance
threshold). This is the approach used for human face recognition and the PetFace
animal re-ID benchmark.
"""
from __future__ import annotations

import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.utils.data import DataLoader, Subset

from src.vision.config import DATA_RAW, MODELS_DIR, RUNS_DIR, load_roster
from src.vision.dataset import RosterDataset, build_transforms
from src.vision.metrics import compute_report
from src.vision.train import stratified_split

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
def _build_base(backbone: str, pretrained: bool = True):
    weights = "DEFAULT" if pretrained else None
    if backbone == "mobilenet_v3_small":
        return models.mobilenet_v3_small(weights=weights)
    if backbone == "mobilenet_v3_large":
        return models.mobilenet_v3_large(weights=weights)
    if backbone == "efficientnet_b0":
        return models.efficientnet_b0(weights=weights)
    if backbone in {"resnet18", "resnet34", "resnet50"}:
        return getattr(models, backbone)(weights=weights)
    raise ValueError(f"Unknown backbone {backbone!r}")


def _feature_dim(model, backbone: str) -> int:
    if backbone.startswith("mobilenet_v3"):
        return model.classifier[0].in_features
    if backbone.startswith("efficientnet"):
        return model.classifier[1].in_features
    return model.fc.in_features  # resnet


def build_embedding_model(backbone: str, embed_dim: int = 128, freeze: bool = True, pretrained: bool = True):
    model = _build_base(backbone, pretrained)
    in_features = _feature_dim(model, backbone)
    head = nn.Sequential(nn.Linear(in_features, embed_dim), nn.BatchNorm1d(embed_dim))
    if hasattr(model, "classifier"):
        model.classifier = head
    else:
        model.fc = head
    if freeze:
        for p in model.parameters():
            p.requires_grad = False
        for p in head.parameters():
            p.requires_grad = True
    return model


class ArcMarginProduct(nn.Module):
    """ArcFace additive angular-margin head (InsightFace)."""

    def __init__(self, in_features: int, out_features: int, s: float = 30.0, m: float = 0.50):
        super().__init__()
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        self.s = s
        self.m = m
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, x: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        cosine = F.linear(F.normalize(x), F.normalize(self.weight))
        sine = torch.sqrt((1.0 - cosine.pow(2)).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1), 1)
        return (one_hot * phi + (1 - one_hot) * cosine) * self.s


# ---------------------------------------------------------------------------
# Embeddings + gallery
# ---------------------------------------------------------------------------
@torch.no_grad()
def _embeddings(model, loader, device):
    model.eval()
    embs, labels = [], []
    for xb, yb in loader:
        embs.append(F.normalize(model(xb.to(device)), dim=1).cpu())
        labels.append(yb)
    return torch.cat(embs), torch.cat(labels)


def _gallery(embs, labels, num_classes):
    protos = []
    for c in range(num_classes):
        m = labels == c
        if m.any():
            protos.append(F.normalize(embs[m].mean(0, keepdim=True), dim=1))
        else:
            protos.append(torch.zeros(1, embs.size(1)))
    return torch.cat(protos)  # [num_classes, embed_dim]


def _run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")


def train_embedding(cfg: dict) -> dict:
    roster = load_roster(cfg.get("roster"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    img_size = cfg.get("img_size", 224)
    batch_size = cfg.get("batch_size", 16)
    epochs = cfg.get("epochs", 30)
    lr = cfg.get("lr", 3e-4)
    backbone = cfg.get("backbone", "mobilenet_v3_large")
    embed_dim = cfg.get("embed_dim", 128)
    freeze = cfg.get("freeze", True)
    data_root = Path(cfg.get("data_root") or DATA_RAW)

    base = RosterDataset(data_root, roster, transform=None)
    if len(base) == 0:
        raise RuntimeError(f"No images under {data_root}/<class>/.")
    train_idx, val_idx = stratified_split(base.samples, cfg.get("val_frac", 0.2), cfg.get("seed", 42))
    train_ds = Subset(RosterDataset(data_root, roster, build_transforms(img_size, True)), train_idx)
    val_ds = Subset(RosterDataset(data_root, roster, build_transforms(img_size, False)), val_idx)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    model = build_embedding_model(backbone, embed_dim, freeze=freeze, pretrained=cfg.get("pretrained", True)).to(device)
    head = ArcMarginProduct(embed_dim, roster.num_classes).to(device)
    optimizer = torch.optim.AdamW(list(model.parameters()) + list(head.parameters()), lr=lr)
    criterion = nn.CrossEntropyLoss()

    rid = _run_id()
    out_dir = RUNS_DIR / f"{rid}_{backbone}_arcface"
    out_dir.mkdir(parents=True, exist_ok=True)

    best_acc = -1.0
    best_state = None
    best_report = None
    best_gallery = None

    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            emb = model(xb)
            logits = head(emb, yb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            running += loss.item() * xb.size(0)

        tr_embs, tr_labels = _embeddings(model, train_loader, device)
        va_embs, va_labels = _embeddings(model, val_loader, device)
        gallery = _gallery(tr_embs, tr_labels, roster.num_classes)
        pred = (va_embs @ gallery.t()).argmax(1)
        report = compute_report(va_labels.tolist(), pred.tolist(), roster.class_names)

        print(
            f"[{rid}] epoch {epoch}/{epochs} loss={running / len(train_idx):.4f} "
            f"acc={report['accuracy']:.3f} f1={report['macro_f1']:.3f}"
        )
        if report["accuracy"] > best_acc:
            best_acc = report["accuracy"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_report = report
            best_gallery = gallery.cpu()

    model.load_state_dict(best_state)
    torch.save(best_state, out_dir / "best.pt")
    torch.save({"gallery": best_gallery, "classes": roster.class_names}, out_dir / "gallery.pt")
    (out_dir / "classes.json").write_text(json.dumps(roster.class_names))
    cfg_out = {
        **cfg, "backbone": backbone, "embed_dim": embed_dim, "num_classes": roster.num_classes,
        "classes": roster.class_names, "counts": base.class_counts(), "method": "arcface",
    }
    (out_dir / "config.json").write_text(json.dumps(cfg_out, indent=2))
    best_report["run_id"] = rid
    (out_dir / "metrics.json").write_text(json.dumps(best_report, indent=2))

    print(f"Done. best accuracy={best_acc:.3f}. Run dir: {out_dir.name}")
    return best_report


def main():
    import argparse

    ap = argparse.ArgumentParser(description="Train an ArcFace embedding model.")
    ap.add_argument("--roster", default=None)
    ap.add_argument("--data-root", default=None, help="e.g. data/processed/crops")
    ap.add_argument("--backbone", default="mobilenet_v3_large")
    ap.add_argument("--embed-dim", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--freeze", dest="freeze", action="store_true", default=True)
    ap.add_argument("--no-freeze", dest="freeze", action="store_false")
    ap.add_argument("--pretrained", dest="pretrained", action="store_true", default=True)
    ap.add_argument("--no-pretrained", dest="pretrained", action="store_false")
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    train_embedding(vars(args))


if __name__ == "__main__":
    main()
