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

- Phase 0 — requirements, safety spec, FMEA: **done** (this repo).
- Phase 1 — vision (detect → fine-tune classifier): **in progress**.
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
2. Collect and label images: `docs/data_collection.md`.
3. Train the classifier (Phase 1) — `src/vision/train.py` (added next).

## Repo layout

```
config/roster.yaml        enrolled animals (single source of truth for training + runtime)
docs/safety.md            safety requirements + FMEA
docs/data_collection.md   how to collect/label a generalizable dataset
docs/architecture.md      system architecture + components
src/vision/               detection + fine-tune + export (Phase 1)
src/control/              ESP32 firmware (Phase 3)
src/sim/                  virtual prototype (Phase 2)
data/raw/<animal>/        images per animal (gitignored — personal photos)
data/processed/           crops, splits, manifests (gitignored)
models/                   exported model weights (gitignored)
```
