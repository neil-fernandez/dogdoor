"""DogDoor training studio — local FastAPI app.

Run from the project root:
    uvicorn src.studio.app:app --reload --port 8000

The studio is dependency-light at startup: torch/torchvision are imported lazily
only when a training job starts, so labeling/experiments work without torch.
"""
from __future__ import annotations

import base64
import io
import json
import shutil
import threading
import time
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from PIL import Image, ImageDraw

from src.vision.config import DATA_RAW, MODELS_DIR, PROJECT_ROOT, RUNS_DIR, list_images, load_roster

app = FastAPI(title="DogDoor Studio")
app.mount(
    "/static",
    StaticFiles(directory=str(PROJECT_ROOT / "src" / "studio" / "static")),
    name="static",
)
templates = Jinja2Templates(directory=str(PROJECT_ROOT / "src" / "studio" / "templates"))

_TRAIN_THREAD: threading.Thread | None = None

LOW_THRESHOLD = 50  # warn when a class has fewer images than this


def _roster_and_counts():
    roster = load_roster()
    counts = {cid: len(list_images(DATA_RAW / cid)) for cid in roster.class_ids}
    return roster, counts


def _list_runs():
    if not RUNS_DIR.exists():
        return []
    runs = []
    for m in sorted(RUNS_DIR.glob("*/metrics.json"), reverse=True):
        try:
            data = json.loads(m.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        data["dir"] = m.parent.name
        runs.append(data)
    return runs


# ---- Dashboard -----------------------------------------------------------
@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    roster, counts = _roster_and_counts()
    low = [cid for cid, n in counts.items() if n < LOW_THRESHOLD]
    return templates.TemplateResponse(
        request, "index.html",
        {"roster": roster, "counts": counts, "low": low, "runs": _list_runs()[:5]},
    )


# ---- Datasets + labeling -------------------------------------------------
@app.get("/datasets", response_class=HTMLResponse)
def datasets(request: Request):
    roster, counts = _roster_and_counts()
    low = [cid for cid, n in counts.items() if n < LOW_THRESHOLD]
    return templates.TemplateResponse(
        request, "datasets.html", {"roster": roster, "counts": counts, "low": low}
    )


@app.get("/datasets/{class_id}", response_class=HTMLResponse)
def label_class(request: Request, class_id: str):
    roster, _ = _roster_and_counts()
    if class_id not in roster.class_ids:
        raise HTTPException(404, "Unknown class")
    images = [p.name for p in list_images(DATA_RAW / class_id)]
    return templates.TemplateResponse(
        request, "label.html",
        {"roster": roster, "class_id": class_id, "images": images},
    )


@app.post("/datasets/{class_id}/move")
def move_image(class_id: str, image: str = Form(...), target: str = Form(...)):
    roster, _ = _roster_and_counts()
    if class_id not in roster.class_ids or target not in roster.class_ids:
        raise HTTPException(400, "Unknown class")
    src = (DATA_RAW / class_id / image).resolve()
    if DATA_RAW.resolve() not in src.parents:
        raise HTTPException(400, "Invalid path")
    dst = DATA_RAW / target / image
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src), str(dst))
    return RedirectResponse(f"/datasets/{class_id}", status_code=303)


@app.get("/media/{class_id}/{name}")
def media(class_id: str, name: str):
    roster, _ = _roster_and_counts()
    if class_id not in roster.class_ids:
        raise HTTPException(404, "Unknown class")
    p = (DATA_RAW / class_id / name).resolve()
    if DATA_RAW.resolve() not in p.parents:
        raise HTTPException(400, "Invalid path")
    return FileResponse(str(p))


# ---- Upload (from phone / bulk) -----------------------------------------
@app.get("/upload", response_class=HTMLResponse)
def upload_page(request: Request):
    roster, counts = _roster_and_counts()
    return templates.TemplateResponse(
        request, "upload.html", {"roster": roster, "counts": counts}
    )


@app.post("/upload")
async def upload(files: list[UploadFile] = File(...), class_id: str = Form(...)):
    roster, _ = _roster_and_counts()
    if class_id not in roster.class_ids:
        raise HTTPException(400, "Unknown class")
    out = DATA_RAW / class_id
    out.mkdir(parents=True, exist_ok=True)
    saved = 0
    for f in files:
        if not f.filename:
            continue
        name = Path(f.filename).name
        (out / f"{int(time.time() * 1000)}_{name}").write_bytes(await f.read())
        saved += 1
    return RedirectResponse(f"/datasets/{class_id}", status_code=303)


# ---- Training ------------------------------------------------------------
@app.get("/train", response_class=HTMLResponse)
def train_page(request: Request):
    roster, counts = _roster_and_counts()
    return templates.TemplateResponse(
        request, "train.html", {"roster": roster, "counts": counts}
    )


def _run_training(cfg: dict):
    try:
        from src.vision import train as train_module  # lazy: imports torch

        train_module.train(cfg)
    except Exception as exc:  # pragma: no cover
        print(f"training failed: {exc}")


@app.post("/train/start")
def start_train(
    backbone: str = Form("mobilenet_v3_large"),
    epochs: int = Form(25),
    lr: float = Form(3e-4),
    batch_size: int = Form(16),
    img_size: int = Form(224),
    freeze: str = Form("on"),
):
    global _TRAIN_THREAD
    if _TRAIN_THREAD is not None and _TRAIN_THREAD.is_alive():
        raise HTTPException(409, "A training job is already running")
    cfg = {
        "backbone": backbone,
        "epochs": epochs,
        "lr": lr,
        "batch_size": batch_size,
        "img_size": img_size,
        "freeze": freeze == "on",
    }
    _TRAIN_THREAD = threading.Thread(target=_run_training, args=(cfg,), daemon=True)
    _TRAIN_THREAD.start()
    return RedirectResponse("/train", status_code=303)


@app.get("/train/status")
def train_status():
    if not RUNS_DIR.exists():
        return {"state": "idle"}
    statuses = list(RUNS_DIR.glob("*/status.json"))
    if not statuses:
        return {"state": "idle"}
    latest = max(statuses, key=lambda p: p.stat().st_mtime)
    return json.loads(latest.read_text())


# ---- Experiments ---------------------------------------------------------
@app.get("/experiments", response_class=HTMLResponse)
def experiments(request: Request):
    return templates.TemplateResponse(request, "experiments.html", {"runs": _list_runs()})


@app.get("/experiments/{run_dir}", response_class=HTMLResponse)
def experiment_detail(request: Request, run_dir: str):
    p = RUNS_DIR / run_dir
    if not p.exists() or not (p / "metrics.json").exists():
        raise HTTPException(404, "Run not found")
    metrics = json.loads((p / "metrics.json").read_text())

    matrix = metrics.get("confusion_matrix", [])
    max_count = max((max(row) for row in matrix), default=0)
    heat = []
    for row in matrix:
        heat_row = []
        for c in row:
            if max_count == 0 or c == 0:
                heat_row.append({"bg": "transparent", "fg": ""})
            else:
                ratio = c / max_count
                heat_row.append(
                    {
                        "bg": f"rgba(26,125,224,{min(ratio, 0.85):.2f})",
                        "fg": "color:#fff;" if ratio > 0.55 else "",
                    }
                )
        heat.append(heat_row)

    return templates.TemplateResponse(
        request, "experiment.html", {"run": run_dir, "metrics": metrics, "heat": heat}
    )


# ---- Predict (playground) ------------------------------------------------
@app.get("/predict", response_class=HTMLResponse)
def predict_page(request: Request):
    return templates.TemplateResponse(request, "predict.html", {})


@app.post("/predict", response_class=HTMLResponse)
async def predict_upload(request: Request, file: UploadFile = File(...)):
    from src.vision.infer import predict  # lazy: imports torch + ultralytics

    data = await file.read()
    image = Image.open(io.BytesIO(data)).convert("RGB")
    result = predict(image)

    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)
    for d in result["detections"][:6]:
        x1, y1, x2, y2 = d["box"]
        draw.rectangle([x1, y1, x2, y2], outline=(225, 29, 72), width=4)
        draw.text((x1, max(0, y1 - 14)), f"{d['name']} {d['conf']:.2f}", fill=(225, 29, 72))

    def b64(img: Image.Image) -> str:
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode()

    return templates.TemplateResponse(
        request,
        "predict.html",
        {
            "original": b64(image),
            "annotated": b64(annotated),
            "detections": result["detections"],
            "animal": result["animal"],
            "predictions": result["predictions"],
            "method": result.get("method"),
            "is_unknown": result.get("is_unknown"),
        },
    )


# ---- Models --------------------------------------------------------------
@app.get("/models", response_class=HTMLResponse)
def models(request: Request):
    files = sorted(p.name for p in MODELS_DIR.glob("*.onnx")) if MODELS_DIR.exists() else []
    return templates.TemplateResponse(request, "models.html", {"models": files})
