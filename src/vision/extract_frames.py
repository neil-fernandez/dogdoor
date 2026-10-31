"""Extract training frames from video into data/raw/<class>/.

Usage:
    python -m src.vision.extract_frames --video clip.mp4 --class beau --every 0.5
    python -m src.vision.extract_frames --dir path/to/videos --class red --every 0.5

Skips blurry frames via Laplacian variance (a cheap quality gate).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from src.vision.config import DATA_RAW, load_roster

VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".3gp"}


def extract(video: Path, class_id: str, every: float, min_quality: float, out_root: Path) -> int:
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        print(f"  (could not open {video.name} — skipping)")
        return 0
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    stride = max(1, int(round(fps * every)))
    out_dir = out_root / class_id
    out_dir.mkdir(parents=True, exist_ok=True)

    n, idx = 0, 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % stride == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            variance = cv2.Laplacian(gray, cv2.CV_64F).var()
            if variance >= min_quality:
                out = out_dir / f"{video.stem}_{idx:06d}.jpg"
                cv2.imwrite(str(out), frame)
                n += 1
        idx += 1
    cap.release()
    return n


def main():
    ap = argparse.ArgumentParser(description="Extract frames from video for training.")
    ap.add_argument("--video", help="single video file")
    ap.add_argument("--dir", help="folder of videos")
    ap.add_argument("--class", dest="class_id", required=True, help="target class id (e.g. beau)")
    ap.add_argument("--every", type=float, default=0.5, help="seconds between saved frames")
    ap.add_argument("--min-quality", type=float, default=50.0, help="Laplacian variance threshold (skip blur)")
    ap.add_argument("--out", default=None, help="output root (default: data/raw)")
    args = ap.parse_args()

    roster = load_roster()
    if args.class_id not in roster.class_ids:
        raise SystemExit(f"Unknown class {args.class_id!r}. Choose from {roster.class_ids}")

    if not args.video and not args.dir:
        raise SystemExit("Provide --video or --dir")

    videos = [Path(args.video)] if args.video else sorted(Path(args.dir).glob("*"))
    videos = [v for v in videos if v.suffix.lower() in VIDEO_EXTS]
    if not videos:
        raise SystemExit("No video files found.")

    out_root = Path(args.out) if args.out else DATA_RAW
    total = 0
    for v in videos:
        n = extract(v, args.class_id, args.every, args.min_quality, out_root)
        print(f"{v.name}: {n} frames")
        total += n
    print(f"Total: {total} frames -> {out_root / args.class_id}")


if __name__ == "__main__":
    main()
