# DogDoor — vision + RFID smart pet door

A smart dog door for three toy poodles (Beau, Red, Misty) that recognises the animal, lets it in
and out, and **soft-opens / soft-closes** with safety-critical pinch and obstacle protection.

It is built as a **virtual-then-physical prototype**, and — importantly — designed to be
**generalizable**: any owner can reuse the same pipeline to enroll and fine-tune a model for their
own animals, then deploy it to the same hardware.

> Safety principle (non-negotiable): **vision and RFID may *suggest* an open; a deterministic
> microcontroller *enforces* safety.** The camera/ML pipeline is probabilistic and must never
> directly drive the motor.

## Status

Live state + exact "done → next" order: see **`STATUS.md`**. Summary:

- Phase 0 — requirements, safety spec, FMEA: **done**.
- Phase 1 — vision pipeline + training studio: **implemented**, pending real data.
- Phases 2–6 — virtual sim, bench, physical, safety, field: pending (see `SPEC.md`).

## Dogs

| id | name | description |
|----|------|-------------|
| `beau` | Beau | black toy poodle |
| `red` | Red | light-brown toy poodle |
| `misty` | Misty | white / dark-brown toy poodle |
| `other` | (negative) | anyone/anything else — never admitted |

## Getting started

1. Read `SPEC.md` and `docs/safety.md`.
2. Install dependencies: `pip install -r requirements.txt`.
3. Collect and label images into `data/raw/<class>/` — see `docs/data_collection.md`.

### Train from the CLI

```powershell
python -m src.vision.train --backbone mobilenet_v3_large --epochs 25
```

Outputs go to `runs/<run>/` (metrics, confusion matrix, weights) and `models/*.onnx`.

### Detect → crop (optional but recommended)

Run a pretrained YOLO11 detector over `data/raw/` and write animal crops:

```powershell
python -m src.vision.preprocess --model yolo11n.pt
```

Then train on the crops (more robust to background/clutter):

```powershell
python -m src.vision.train --data-root data/processed/crops
```

### Training studio (web UI)

```powershell
uvicorn src.studio.app:app --reload --port 8000
```

Then open <http://localhost:8000> — label images, launch training, **test predictions on new images
(the Predict playground)**, compare experiments, and view the confusion matrix + per-class metrics.
The studio runs without torch installed (torch is imported only when a training job starts).

## Repo layout

```
STATUS.md                 live state: what's done, what's next, recovery prompt
AGENTS.md                 operating brief for AI coding assistants
SPEC.md                   system specification
config/roster.yaml        enrolled animals (single source of truth for training + runtime)
docs/safety.md            safety requirements + FMEA
docs/data_collection.md   how to collect/label a generalizable dataset
src/vision/               roster-driven training pipeline (config, dataset, model, train, export)
src/studio/               FastAPI training studio (label / train / evaluate / compare)
src/control/              ESP32 firmware (Phase 3)
src/sim/                  virtual control prototype (Phase 2)
requirements.txt          Python dependencies
data/raw/<animal>/        images per animal (gitignored — personal photos)
runs/                     training-run metrics + weights (gitignored)
models/                   exported ONNX models (gitignored)
```
