# Data Collection & Labeling Plan

Goal: fine-tune a small classifier that distinguishes **Beau, Red, Misty** from **everything else**,
then export it for the Raspberry Pi. This is the single most important de-risking step — and it is
done before any hardware.

## 1. The labeling scheme (folder = label)

We use **folder-per-class** labeling. You do **not** need to draw boxes — a pretrained detector
(YOLO) finds the dog and crops it automatically at training time.

```
data/raw/
  beau/        ← photos where the dog is Beau
  red/         ← photos where the dog is Red
  misty/       ← photos where the dog is Misty
  other/       ← photos of anything that is NOT an enrolled dog
```

- `other/` = people, other animals (cats, birds), the empty doorway, hands, furniture, the door
  itself. **This class is just as important as the dogs** — it is what makes the system *reject*
  strangers instead of mis-labelling them.

## 2. What to capture

Per dog, aim for **at least 200–300 images to start, 500+ ideally**. More variety beats more
quantity of the same photo.

| Dimension | Coverage |
|-----------|----------|
| Angles | face-on, profile, rear, top-down, low angle |
| Distance | close (head fills frame), mid, far (whole body) |
| Lighting | daylight, indoor, shadow, evening — **include the doorway area** |
| Posture | standing, sitting, walking, lying, at the door |
| Background | doorway (target), plus 2–3 other backgrounds |
| Appearance | **before AND after grooming** (poodles change appearance!) |

### Video is better than photos

If you can take a few **short videos** (10–30 s) of each dog moving around the doorway, do it — we
can extract many varied frames from video automatically. Phone video is perfect.

### The "other" class

Collect the same volume (200+) of: people walking past, a cat if you have one, the empty doorway at
different times of day, and random household objects. This is what keeps a stranger from being let
in.

## 3. What to hand me

1. Copy/move your existing photos into the folders above (I've created them on your laptop).
2. Rough **count per folder** (e.g. "Beau 150, Red 90, Misty 200, other 60") so I know if we need a
   capture session.
3. Whether you also have **video**, and roughly how much.
4. (Optional but helpful) the **distinguishing features** you use to tell them apart at a glance.

## 4. Privacy & repo hygiene

- `data/raw/` is **gitignored** — personal photos never enter version control.
- Only `config/roster.yaml` (the animal metadata) is committed.
- If you later want a public portfolio repo, we keep photos private and share only code + the model +
  a redacted write-up.

## 5. What I'll do next (once images are in place)

1. Write a **frame-extraction script** if you have video.
2. Write the **training pipeline** (`src/vision/train.py`): build dataset from the roster → auto-crop
   with YOLO → fine-tune → output a confusion matrix + per-class precision/recall.
3. Tell you exactly what accuracy we achieve and whether we need more data.
