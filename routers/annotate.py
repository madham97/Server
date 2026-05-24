import json
import logging
import random
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel

from config import UPLOAD_DIR, ANNOTATED_DIR, ANNOTATION_LOG, CLASSES_FILE
from inference import IMAGE_EXTS

router = APIRouter(prefix="/annotate")

STATIC_DIR = Path(__file__).parent.parent / "static"


@router.get("", response_class=HTMLResponse)
async def annotate_ui():
    return FileResponse(str(STATIC_DIR / "annotate.html"))

with open(CLASSES_FILE) as f:
    CLASSES: list[str] = json.load(f)["classes"]



def _load_annotation_log() -> set[str]:
    if not ANNOTATION_LOG.exists():
        return set()
    names = set()
    with open(ANNOTATION_LOG) as f:
        for line in f:
            parts = line.strip().split('\t')
            if parts:
                names.add(parts[-1])
    return names


def _record_annotated(filename: str):
    annotated_at = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
    with open(ANNOTATION_LOG, 'a') as f:
        f.write(f"{annotated_at}\t{filename}\n")


def _unannotated() -> list[str]:
    annotated = _load_annotation_log()
    return [
        p.name for p in UPLOAD_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS and p.name not in annotated
    ]


@router.get("/stats")
async def annotate_stats():
    pending = _unannotated()
    annotated_dirs = [p for p in ANNOTATED_DIR.iterdir() if p.is_dir()]
    class_counts = {name: 0 for name in CLASSES}
    for d in annotated_dirs:
        label_file = d / "labels" / f"{d.name}.txt"
        if label_file.exists():
            for line in label_file.read_text().splitlines():
                parts = line.strip().split()
                if parts and int(parts[0]) < len(CLASSES):
                    class_counts[CLASSES[int(parts[0])]] += 1
    return {
        "pending": len(pending),
        "annotated": len(annotated_dirs),
        "class_counts": class_counts,
    }


@router.get("/next")
async def annotate_next(shuffle: bool = True, after: str = ""):
    pool = _unannotated()
    if not pool:
        raise HTTPException(status_code=404, detail="No images pending annotation")
    if shuffle:
        image_name = random.choice(pool)
    else:
        pool_sorted = sorted(pool)
        # find first name that sorts after the current image, wrap to front if none
        following = [n for n in pool_sorted if n > after]
        image_name = following[0] if following else pool_sorted[0]
    return {
        "image_name": image_name,
        "classes": CLASSES,
    }


@router.get("/specific/{image_name}")
async def annotate_specific(image_name: str):
    path = UPLOAD_DIR / image_name
    if not path.exists():
        raise HTTPException(status_code=404, detail="Image not found in uploads")
    return {
        "image_name": image_name,
        "classes": CLASSES,
    }


@router.get("/image/{image_name}")
async def annotate_image(image_name: str):
    path = UPLOAD_DIR / image_name
    if not path.exists():
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(str(path), media_type="image/jpeg")


class LabelIn(BaseModel):
    class_id: int
    x: float
    y: float
    w: float
    h: float


class AnnotationIn(BaseModel):
    labels: list[LabelIn]


@router.post("/{image_name}")
async def annotate_save(image_name: str, body: AnnotationIn):
    stem = Path(image_name).stem
    out_dir = ANNOTATED_DIR / stem / "labels"
    out_dir.mkdir(parents=True, exist_ok=True)
    label_file = out_dir / f"{stem}.txt"
    lines = [
        f"{lbl.class_id} {lbl.x} {lbl.y} {lbl.w} {lbl.h}"
        for lbl in body.labels
    ]
    label_file.write_bytes("\n".join(lines).encode())
    _record_annotated(image_name)
    logging.info(f"Saved {len(body.labels)} labels for {image_name}")
    return {"status": "ok", "labels_saved": len(body.labels)}
