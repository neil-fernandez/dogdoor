"""Fine-tune a roster classifier and export it to ONNX.

CLI (from the project root):
    python -m src.vision.train --backbone mobilenet_v3_large --epochs 25

The `train()` function is also what the studio calls in a background thread.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

# On Windows the console may use a non-UTF-8 code page, which makes torch.onnx's
# progress logging (it prints emoji) crash. Reconfigure to UTF-8 defensively.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from src.vision.config import DATA_RAW, MODELS_DIR, RUNS_DIR, Roster, load_roster
from src.vision.dataset import RosterDataset, build_transforms
from src.vision.metrics import compute_report
from src.vision.model import build_model


def stratified_split(samples, val_frac: float = 0.2, seed: int = 42):
    """Split sample indices per-class so every class is present in both splits."""
    by_label = defaultdict(list)
    for i, (_, label) in enumerate(samples):
        by_label[label].append(i)

    train_idx, val_idx = [], []
    rng = random.Random(seed)
    for label in sorted(by_label):
        idxs = by_label[label]
        rng.shuffle(idxs)
        n_val = max(1, int(round(len(idxs) * val_frac)))
        val_idx.extend(idxs[:n_val])
        train_idx.extend(idxs[n_val:])
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    return train_idx, val_idx


def class_weights(roster: Roster, samples, indices) -> torch.Tensor:
    """Inverse-frequency weights to counter class imbalance."""
    counts = defaultdict(int)
    for i in indices:
        counts[samples[i][1]] += 1
    total = sum(counts.values())
    n = len(roster.class_ids)
    return torch.tensor(
        [total / (n * max(1, counts.get(c, 0))) for c in range(n)], dtype=torch.float32
    )


def _run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")


def train(cfg: dict) -> dict:
    roster = load_roster(cfg.get("roster"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    img_size = cfg.get("img_size", 224)
    batch_size = cfg.get("batch_size", 16)
    epochs = cfg.get("epochs", 25)
    lr = cfg.get("lr", 3e-4)
    backbone = cfg.get("backbone", "mobilenet_v3_large")
    freeze = cfg.get("freeze", True)
    pretrained = cfg.get("pretrained", True)

    data_root = Path(cfg.get("data_root") or DATA_RAW)

    base = RosterDataset(data_root, roster, transform=None)
    if len(base) == 0:
        raise RuntimeError(
            f"No images found under {data_root}/<class>/. Add images (or run preprocess) first."
        )

    train_idx, val_idx = stratified_split(base.samples, cfg.get("val_frac", 0.2), cfg.get("seed", 42))
    train_ds = Subset(RosterDataset(data_root, roster, build_transforms(img_size, True)), train_idx)
    val_ds = Subset(RosterDataset(data_root, roster, build_transforms(img_size, False)), val_idx)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    model = build_model(backbone, roster.num_classes, freeze=freeze, pretrained=pretrained).to(device)
    weights = class_weights(roster, base.samples, train_idx).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=3, factor=0.5)

    rid = _run_id()
    out_dir = RUNS_DIR / f"{rid}_{backbone}"
    out_dir.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    def write_status(**kw):
        (out_dir / "status.json").write_text(json.dumps({"run_id": rid, **kw}, indent=2))

    best_f1 = -1.0
    best_state = None
    best_report = None

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * xb.size(0)
        train_loss = running_loss / len(train_idx)

        model.eval()
        y_true, y_pred, val_loss = [], [], 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                val_loss += criterion(logits, yb).item() * xb.size(0)
                y_true.extend(yb.cpu().tolist())
                y_pred.extend(logits.argmax(1).cpu().tolist())
        val_loss = val_loss / len(val_idx)
        report = compute_report(y_true, y_pred, roster.class_names)
        scheduler.step(val_loss)

        write_status(
            state="training",
            epoch=epoch,
            total_epochs=epochs,
            train_loss=round(train_loss, 4),
            val_loss=round(val_loss, 4),
            val_accuracy=round(report["accuracy"], 4),
            val_f1=round(report["macro_f1"], 4),
        )
        print(
            f"[{rid}] epoch {epoch}/{epochs} "
            f"loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"acc={report['accuracy']:.3f} f1={report['macro_f1']:.3f}"
        )

        if report["macro_f1"] > best_f1:
            best_f1 = report["macro_f1"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_report = report

    model.load_state_dict(best_state)
    torch.save(best_state, out_dir / "best.pt")
    (out_dir / "classes.json").write_text(json.dumps(roster.class_names))

    cfg_out = {
        **cfg,
        "backbone": backbone,
        "num_classes": roster.num_classes,
        "classes": roster.class_names,
        "counts": base.class_counts(),
    }
    (out_dir / "config.json").write_text(json.dumps(cfg_out, indent=2))
    best_report["run_id"] = rid
    (out_dir / "metrics.json").write_text(json.dumps(best_report, indent=2))

    onnx_path = None
    try:
        model.eval()
        dummy = torch.randn(1, 3, img_size, img_size, device=device)
        onnx_path = MODELS_DIR / f"{rid}_{backbone}.onnx"
        torch.onnx.export(
            model, dummy, str(onnx_path),
            input_names=["input"], output_names=["logits"], opset_version=13,
        )
    except Exception as e:  # onnx optional
        print(f"ONNX export skipped: {e}")

    write_status(state="done", best_f1=round(best_f1, 4), onnx=str(onnx_path) if onnx_path else None)
    print(f"Done. best macro-F1={best_f1:.3f}. Run dir: {out_dir.name}")
    return best_report


def main():
    ap = argparse.ArgumentParser(description="Fine-tune the DogDoor roster classifier.")
    ap.add_argument("--roster", default=None)
    ap.add_argument("--data-root", default=None, help="e.g. data/processed/crops (default: data/raw)")
    ap.add_argument("--backbone", default="mobilenet_v3_large")
    ap.add_argument("--epochs", type=int, default=25)
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
    train(vars(args))


if __name__ == "__main__":
    main()
