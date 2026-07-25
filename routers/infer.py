import asyncio
import logging
import random

from fastapi import APIRouter, HTTPException
from ultralytics import YOLO

from config import UPLOAD_DIR, PROCESSED_DIR, MODEL_WEIGHTS, MODEL_CONF, MODEL_IOU, MODEL_IMGSZ
from inference import process_image, IMAGE_EXTS

router = APIRouter(prefix="/infer")


@router.post("/test")
async def infer_test(n: int = 5):
    images = [p for p in UPLOAD_DIR.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    if not images:
        raise HTTPException(status_code=404, detail="No images found in uploads/")

    sample = random.sample(images, min(n, len(images)))

    def _run():
        model = YOLO(MODEL_WEIGHTS)
        results = []
        for path in sample:
            try:
                process_image(str(path), weights=MODEL_WEIGHTS, project=str(PROCESSED_DIR),
                              conf=MODEL_CONF, iou=MODEL_IOU, imgsz=MODEL_IMGSZ, model=model)
                results.append({"image": path.name, "status": "ok"})
            except Exception as e:
                logging.exception(f"Failed inference on {path.name}")
                results.append({"image": path.name, "status": "error", "detail": str(e)})
        return results

    results = await asyncio.to_thread(_run)
    return {"processed": len(results), "results": results}
