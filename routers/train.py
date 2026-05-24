import shutil
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, HTTPException

import trainer
from config import BASE_MODEL, TRAIN_EPOCHS, TRAIN_IMGSZ, MODEL_WEIGHTS, MIN_MAP, DATASET_DIR, CANDIDATE_WEIGHTS, ARCHIVE_DIR

router = APIRouter(prefix="/train")

DATASET_YAML = DATASET_DIR / "dataset.yaml"
CANDIDATE = CANDIDATE_WEIGHTS


@router.post("/start")
async def train_start(base_model: str = BASE_MODEL, epochs: int = TRAIN_EPOCHS, imgsz: int = TRAIN_IMGSZ):
    if not DATASET_YAML.exists():
        raise HTTPException(status_code=400, detail="Dataset not exported yet — run POST /dataset/export first")
    started = trainer.start_training(base_model, epochs, imgsz, str(DATASET_YAML))
    if not started:
        raise HTTPException(status_code=409, detail="Training already running")
    return {"status": "started", "base_model": base_model, "epochs": epochs, "imgsz": imgsz}


@router.get("/status")
async def train_status():
    return trainer.get_status()


@router.post("/promote")
async def train_promote():
    if not CANDIDATE.exists():
        raise HTTPException(status_code=404, detail="No candidate model found — run POST /train/start first")

    status = trainer.get_status()
    if status["state"] == "running":
        raise HTTPException(status_code=409, detail="Training still in progress")

    map50 = status.get("metrics", {}).get("metrics/mAP50(B)", 0.0)
    if map50 < MIN_MAP:
        raise HTTPException(
            status_code=400,
            detail=f"Candidate mAP50 {map50:.3f} is below minimum threshold {MIN_MAP}"
        )

    existing = Path(MODEL_WEIGHTS)
    if existing.exists():
        ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%dT%H%M%S")
        shutil.copy2(existing, ARCHIVE_DIR / f"{ts}_best.pt")

    shutil.copy2(CANDIDATE, MODEL_WEIGHTS)
    return {"status": "promoted", "map50": map50, "model": MODEL_WEIGHTS}
