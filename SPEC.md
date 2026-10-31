# DogDoor — System Specification

## 1. Goal

Build a smart pet door that:

1. **Identifies** which of our three toy poodles (Beau, Red, Misty) is at the door, and only admits
   enrolled animals.
2. **Soft-opens** to let the animal through, then **soft-closes** behind it.
3. Keeps **bugs and water out** when closed.
4. Is **safe**: it must never pinch or trap an animal (or a human hand), and it must fail in a
   defined, safe way.

## 2. Non-negotiable requirements (Generalizable-first)

This is the requirement that shapes the whole design: **the system must be reusable by any owner to
fine-tune for their own animals.** Concretely, that means:

- **No hard-coded animals.** The enrolled set lives in `config/roster.yaml`. Adding or removing an
  animal changes only that file (+ retraining), never the code.
- **Standard enrollment → train → evaluate → deploy loop.** A new household records images of their
  animals into `data/raw/<animal_id>/`, runs the training script, and gets an exported model +
  evaluation report for the door.
- **A fixed "other/unknown" class** that represents *everything else* (people, cats, raccoons, an
  empty doorway) and is always rejected. This is what prevents false admits.
- **Deterministic safety layer independent of ML.** The safety controller's behaviour does not
  depend on model quality.

### Safety principle (applies everywhere)

> Vision and RFID may **suggest** an open. A deterministic microcontroller **enforces** safety.

The motor, obstacle detection, soft-close, and fail-safe logic run on a hard-coded controller that
ignores bad requests and stops/reverses instantly on any hazard. The ML pipeline can only say
"open for this animal" — it never moves the door.

## 3. Animals

| id | name | description |
|----|------|-------------|
| `beau` | Beau | black toy poodle |
| `red` | Red | light-brown toy poodle |
| `misty` | Misty | white / dark-brown toy poodle |
| `other` | negative class | everything else — reject |

Toy poodles are small (~4–6 kg, ~25 cm tall) and the three are same-breed, so **vision ID is a
fine-grained classification problem**. RFID (collar tag) is the authoritative identity; vision is
confirmation + fallback + safety context.

## 4. Architecture (high level)

```
                   ┌──────────────────────────────────────────────┐
   camera ───────► │ Raspberry Pi (vision)                        │
   RFID reader ──► │   detect animal → classify/verify identity   │
                   │   → sends "open for <animal>" (advisory)     │
                   └───────────────┬──────────────────────────────┘
                                   │  UART (single command)
                   ┌───────────────▼──────────────────────────────┐
   IR beam ──────► │ ESP32 (deterministic safety controller)      │
   limit sw ─────► │   state machine + motor + ALL safety logic   │
   current ──────► │   soft open/close, obstacle reverse, fail-safe│
                   └───────────────┬──────────────────────────────┘
                                   │
                                   ▼
                       motorized flap (stepper + TMC2209)
```

**Subsystems:**

1. **Sensing** — camera, RFID reader, IR break-beam (threshold), limit switches, motor
   current/stall, optional PIR.
2. **Recognition** — two-stage: (1) pretrained detector finds "dog/animal" and crops it; (2) a
   small CNN fine-tuned on the roster classifies the crop. RFID provides authoritative ID.
3. **Control** — a deterministic state machine (idle → approach → identify → open → animal cleared →
   soft-close → latch) with all safety interrupts.
4. **Actuation** — a stepper + TMC2209 driver (silent, smooth ramps, built-in stall detection = free
   obstacle sensor).
5. **Mechanics** — a sealed motorized flap (drip-lip, perimeter seal, magnetic catch) so it closes
   flush against bugs and water.

## 5. Recognition pipeline (Phase 1, generalizable)

1. **Detect** — a pretrained YOLO model localises "dog / animal / person" and crops the detection.
   Zero per-owner training.
2. **Classify** — a CNN backbone (e.g. MobileNetV3/EfficientNet) fine-tuned for
   `N enrolled animals + 1 "other"`. The class list is derived from `config/roster.yaml`.
3. **Evaluate** — per-class precision/recall + a confusion matrix; the model must reject "other"
   reliably before it is allowed near a door.
4. **Export** — ONNX for inference on the Raspberry Pi.

The enrollment → train → evaluate → export flow is the same for us and for any future owner.

## 6. Milestones (3-month target)

| # | Phase | Exit criterion |
|---|-------|----------------|
| 0 | Spec + safety + FMEA | This repo |
| 1 | Vision | Classifier ≥ target accuracy, "other" rejected reliably |
| 2 | Virtual control sim | State machine + safety interrupts validated in sim |
| 3 | Bench prototype | Motor + sensors behave correctly on a bench |
| 4 | Physical door | Sealed flap works in a test box |
| 5 | Safety + field | Pinch/obstacle/power-loss tests pass; supervised dog trials |
| 6 | Harden + portfolio | Write-up, demo, public repo + project page |

## 7. Non-goals (v1)

- No cloud dependency (runs fully on-device).
- No facial *verification* at human-grade security; this is a convenience + safety gate, not a
  security boundary.
- No remote monitoring/phone app (can be added later).
- Not certified for commercial sale (v1 is a prototype).
