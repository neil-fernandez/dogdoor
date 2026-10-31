import json

import pytest
from PIL import Image

from src.vision import train as train_module
from src.vision.config import load_roster

pytestmark = pytest.mark.slow


def _make_data(tmp_path):
    roster = load_roster()
    for cid in roster.class_ids:
        d = tmp_path / cid
        d.mkdir(parents=True)
        for i in range(6):
            Image.new("RGB", (64, 64), (i * 30, i * 30, i * 30)).save(d / f"{i}.png")
    return roster


def test_train_end_to_end(tmp_path, monkeypatch):
    _make_data(tmp_path)
    runs = tmp_path / "runs"
    models = tmp_path / "models"
    monkeypatch.setattr(train_module, "RUNS_DIR", runs)
    monkeypatch.setattr(train_module, "MODELS_DIR", models)

    report = train_module.train(
        {
            "data_root": str(tmp_path),
            "backbone": "mobilenet_v3_small",
            "epochs": 1,
            "img_size": 32,
            "batch_size": 4,
            "freeze": True,
            "pretrained": False,
        }
    )

    assert 0.0 <= report["accuracy"] <= 1.0
    metrics_files = list(runs.glob("*/metrics.json"))
    assert metrics_files, "expected a metrics.json in the run dir"
    metrics = json.loads(metrics_files[0].read_text())
    assert set(metrics["class_names"]) == {"Beau", "Red", "Misty", "Other"}
