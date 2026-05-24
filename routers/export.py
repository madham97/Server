import json
import shutil
import random

from fastapi import APIRouter, HTTPException

from config import UPLOAD_DIR, ANNOTATED_DIR, DATASET_DIR, CLASSES_FILE
from inference import IMAGE_EXTS

router = APIRouter(prefix="/dataset")

with open(CLASSES_FILE) as f:
    CLASSES: list[str] = json.load(f)["classes"]


@router.post("/export")
async def export_dataset(val_split: float = 0.2):
    if not (0 < val_split < 1):
        raise HTTPException(status_code=422, detail="val_split must be between 0 and 1")

    annotated_dirs = [p for p in ANNOTATED_DIR.iterdir() if p.is_dir()]
    if not annotated_dirs:
        raise HTTPException(status_code=400, detail="No annotated images found")

    # Verify source images exist
    entries = []
    for d in annotated_dirs:
        label_path = d / "labels" / f"{d.name}.txt"
        image_path = next(
            (UPLOAD_DIR / f"{d.name}{ext}" for ext in IMAGE_EXTS
             if (UPLOAD_DIR / f"{d.name}{ext}").exists()),
            None,
        )
        if image_path is not None and label_path.exists():
            entries.append((image_path, label_path))

    if not entries:
        raise HTTPException(status_code=400, detail="No valid image/label pairs found")

    # Split into train/val
    random.shuffle(entries)
    n_val = max(1, round(len(entries) * val_split))
    val_entries = entries[:n_val]
    train_entries = entries[n_val:]

    # Build folder structure
    for split in ("train", "val"):
        (DATASET_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (DATASET_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)

    def _copy(entry_list, split):
        for image_path, label_path in entry_list:
            shutil.copy2(image_path, DATASET_DIR / "images" / split / image_path.name)
            shutil.copy2(label_path, DATASET_DIR / "labels" / split / label_path.name)

    _copy(train_entries, "train")
    _copy(val_entries, "val")

    # Write dataset.yaml
    yaml_path = DATASET_DIR / "dataset.yaml"
    yaml_path.write_text(
        f"path: {DATASET_DIR.resolve()}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"nc: {len(CLASSES)}\n"
        f"names: {CLASSES}\n"
    )

    return {
        "train": len(train_entries),
        "val": len(val_entries),
        "total": len(entries),
        "dataset_yaml": str(yaml_path.resolve()),
    }


@router.get("/stats")
async def dataset_stats():
    if not DATASET_DIR.exists():
        return {"exported": False}
    train_images = list((DATASET_DIR / "images" / "train").glob("*.jpg")) if (DATASET_DIR / "images" / "train").exists() else []
    val_images = list((DATASET_DIR / "images" / "val").glob("*.jpg")) if (DATASET_DIR / "images" / "val").exists() else []
    return {
        "exported": True,
        "train": len(train_images),
        "val": len(val_images),
        "dataset_yaml": str((DATASET_DIR / "dataset.yaml").resolve()),
    }
