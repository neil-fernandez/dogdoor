# DogDoor — Project Status

Last updated: 2026-10-31

## Current status

The software foundation is built and verified end-to-end: a **generalizable, roster-driven training
pipeline** (YOLO11 detect → fine-tune a classifier → export ONNX) and a **FastAPI training studio**
(label / train / predict / evaluate / compare). Hardware is not started. **The blocker is data** —
no real dog photos have been added yet.

## New chat recovery prompt

Paste this into a fresh AI coding session to restore context:

```text
We are building "DogDoor", a smart pet door for three toy poodles (Beau, Red, Misty) in
C:\Users\nfern\Desktop\dogdoor. It recognises animals (hybrid RFID + vision), soft-opens/closes,
and is designed to be generalizable (any owner can fine-tune for their own dogs).

Project root is C:\Users\nfern\Desktop\dogdoor (a git repo). Windows environment: use
`python -m ...` and `cmd /c` for npm; `rg` may not be installed.

First run:
  Set-Location C:\Users\nfern\Desktop\dogdoor
  git log --oneline -5
  Get-Content STATUS.md
  Get-Content AGENTS.md
  Get-Content SPEC.md

Then read docs/safety.md, docs/data_collection.md, and docs/agent_orchestration.md as needed.
STATUS.md is the live state. The harness lives in .opencode/ (agent profiles + opencode.jsonc).

Commands:
  python -m compileall -q src                                   # syntax check
  python -m pytest -q                                           # test suite
  uvicorn src.studio.app:app --port 8000                        # studio -> http://localhost:8000
  python -m src.vision.preprocess --model yolo11n.pt            # detect -> crop
  python -m src.vision.train --backbone mobilenet_v3_large --epochs 25
  python -m src.vision.train --data-root data/processed/crops   # train on crops

Python 3.14; deps in requirements.txt (torch 2.15+cpu, torchvision, ultralytics, fastapi, ...).
Use the project venv: `.venv\Scripts\python.exe -m ...` (or activate `.venv\Scripts\Activate.ps1`).

Safety principle: vision and RFID may SUGGEST an open; a deterministic microcontroller ENFORCES
safety. The ML pipeline never drives the motor directly.
```

## Done (exact order)

| # | What | Commit |
|---|------|--------|
| 1 | Scaffold: spec, safety/FMEA, data plan, roster | `58b746a` |
| 2 | Vision pipeline + training studio (label/train/experiments/models) | `4475e53` |
| 3 | YOLO11 detection stage + prediction playground | `6502f85` |
| 4 | Project memory: STATUS.md + AGENTS.md (recovery + operating brief) | `836e4da` |
| 5 | Harness: `.opencode/` agent profiles + `docs/agent_orchestration.md` | `a9cbf64` |
| 6 | Test suite (pytest) + `--no-pretrained` training option | `cb7f72a` |
| 7 | Document venv setup | `0329fa0` |
| 8 | Slick evals: live progress, confusion heatmap, data-quality nudge | `HEAD` |

## Next (exact order)

1. **Data (blocker)** — add photos to `data/raw/{beau,red,misty,other}` + any video; note counts.
2. **First real training pass** — preprocess → train → evaluate; report per-dog accuracy + confusion matrix.
3. **Error thumbnails** — show misclassified validation images (the remaining slick-eval).
4. **(later)** ArcFace/metric-learning mode (PetFace reference) for best "fine-tune for own dogs" generalisation.
5. Add a **Dockerfile** for reproducible training/studio runs.
6. **Hardware phases** (SPEC.md milestones 2–6): control sim → bench → physical → safety → field.

## Blockers

- No real data yet (photos / video not added).
- No hardware yet (Pi / ESP32 / sensors / 3D printer) — not ordered.

## Decisions (locked)

- **Hybrid RFID + vision**: RFID is authoritative identity; vision is advisory + safety context.
- **Safety**: a deterministic microcontroller enforces safety; the ML pipeline only suggests.
- **Detector**: Ultralytics YOLO11n (COCO). **Classifier**: ImageNet backbone fine-tune (default `mobilenet_v3_large`).
- **Generalizable**: roster-driven classes + fixed `other` class; enrollment → train → eval → export loop.
- **Studio**: FastAPI + Jinja2, local single-user. Defer MLflow / Plotly / Optuna / Docker / auth until justified.

## Known inputs

- Dogs: **Beau** (black), **Red** (light brown), **Misty** (white/dark brown) — toy poodles.
- Door: embedded in a wooden/sliding door; must keep bugs + water out.
- Budget: flexible. Timeline: 3 months max. Goal: eventually automation/robotics full-time.

## Latest recovery checkpoint

- **Objective**: build the detection stage + prediction playground; survey latest tech.
- **Files changed**: `src/vision/{detector,infer,preprocess}.py`, `src/vision/{config,train}.py`,
  `src/studio/app.py`, `src/studio/templates/{base,predict}.html`, `README.md`, `requirements.txt`.
- **Verified**: pipeline trains end-to-end (synthetic data), YOLO11 loads, `/predict` returns 200.
- **Next atomic task**: wait for real photos → run the first real training pass.
