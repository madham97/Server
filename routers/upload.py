import io
import logging
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from PIL import Image as PILImage

from config import UPLOAD_DIR, UPLOAD_LOG

router = APIRouter()


@router.post("/upload")
async def upload(
    image: UploadFile = File(None),
    device_id: str = Form(""),
    mode: str = Form(""),
    motion_score: str = Form(""),
    timestamp: str = Form(""),
):
    if image is None:
        raise HTTPException(status_code=422, detail="No file provided")

    data = await image.read()

    if image.filename.lower().endswith('.webp'):
        img = PILImage.open(io.BytesIO(data))
        buf = io.BytesIO()
        img.save(buf, format='JPEG', quality=95)
        data = buf.getvalue()
        filename = Path(image.filename).stem + '.jpg'
    else:
        filename = image.filename

    (UPLOAD_DIR / filename).write_bytes(data)

    meta_parts = [p for p in [
        device_id,
        mode,
        f"motion={motion_score}" if motion_score else None,
        timestamp,
    ] if p]
    logging.info(f"Received upload: {filename}" +
                 (f" [{', '.join(meta_parts)}]" if meta_parts else ""))

    received_at = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
    with open(UPLOAD_LOG, 'a') as f:
        f.write(f"{received_at}\t{filename}\t{device_id}\t{mode}\t{motion_score}\t{timestamp}\n")

    return {"status": "ok"}
