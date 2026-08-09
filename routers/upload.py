import io
import json
import logging
import secrets
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
from fastapi import APIRouter, BackgroundTasks, UploadFile, File, Form, Header, HTTPException
from PIL import Image as PILImage

from config import THERMAL_DIR, UPLOAD_DIR, UPLOAD_LOG, UPLOAD_TOKEN
from thermal_align import align_thermal, find_profile_for_timestamp

router = APIRouter()


def _align_new_capture(thermal_png: Path, stem: str, timestamp: str) -> None:
    """Applies whichever calibration profile covers this capture's own timestamp to a
    freshly-arrived thermal frame, so captures get an aligned counterpart as they come in
    rather than only after the next manual calibration run (which is the only other place
    align_all/align_thermal get called). Uses the capture's own timestamp — not "now" —
    so this stays correct even for a delayed/batched upload of an older capture: it gets
    the profile that was actually in effect when the photo was taken, not whatever profile
    happens to be current at upload time. Silently does nothing if no profile covers it."""
    try:
        profile = find_profile_for_timestamp(timestamp) if timestamp else None
        if profile is None:
            return
        homography = np.array(profile['homography'], dtype=np.float64)
        aligned = align_thermal(thermal_png, homography, ref_size=profile.get('ref_size'))
        cv2.imwrite(str(THERMAL_DIR / f'{stem}_thermal_aligned.png'), aligned)
    except Exception:
        logging.exception(f'Failed to align new capture {stem}')

# Thermal-fused frames from the Pi arrive as RGBA WebP: the visible image in RGB, the
# normalized thermal map in the alpha channel. JPEG can't hold alpha, so we split them —
# the RGB is saved to UPLOAD_DIR as a normal JPEG (the detection pipeline handles it
# unchanged), and the thermal channel is kept in THERMAL_DIR alongside a JSON sidecar with
# the actual temperature range (needed to reconstruct °C from the 0-255 alpha). THERMAL_DIR
# is a sibling of UPLOAD_DIR, not a subdirectory, so thermal files are never in scope for
# the uploads watcher regardless of how it filters.


def _check_upload_token(header_token: str, form_token: str) -> None:
    """Reject the request unless it carries the shared secret. Accepts it either as the
    X-Upload-Token header or as a `token` multipart field: the Pi's SIM800 sets custom
    headers through AT+HTTPPARA="USERDATA", whose behaviour varies by firmware, so the
    uploader sends the field instead — it needs no extra AT commands and rides along in the
    multipart body it already builds. compare_digest keeps the comparison constant-time."""
    if not UPLOAD_TOKEN:
        return
    supplied = header_token or form_token or ""
    if not secrets.compare_digest(supplied, UPLOAD_TOKEN):
        raise HTTPException(status_code=401, detail="Invalid or missing upload token")


@router.post("/upload")
async def upload(
    background_tasks: BackgroundTasks,
    image: UploadFile = File(None),
    token: str = Form(""),
    x_upload_token: str = Header(None),
    device_id: str = Form(""),
    mode: str = Form(""),
    motion_score: str = Form(""),
    timestamp: str = Form(""),
    format: str = Form(""),
    thermal_min_c: str = Form(""),
    thermal_max_c: str = Form(""),
    thermal_avg_c: str = Form(""),
):
    _check_upload_token(x_upload_token, token)

    if image is None:
        raise HTTPException(status_code=422, detail="No file provided")

    data = await image.read()
    thermal_file = None

    if image.filename.lower().endswith('.webp'):
        img = PILImage.open(io.BytesIO(data))
        stem = Path(image.filename).stem

        if 'A' in img.getbands():
            # Thermal-fused frame: separate the visible RGB from the thermal alpha channel.
            rgba = img.convert('RGBA')
            buf = io.BytesIO()
            rgba.convert('RGB').save(buf, format='JPEG', quality=95)
            data = buf.getvalue()
            filename = stem + '.jpg'

            THERMAL_DIR.mkdir(exist_ok=True)
            thermal_file = stem + '_thermal.png'
            rgba.getchannel('A').save(THERMAL_DIR / thermal_file, format='PNG')
            (THERMAL_DIR / (stem + '_thermal.json')).write_text(json.dumps({
                'source_image':  filename,
                'device_id':     device_id,
                'timestamp':     timestamp,
                'thermal_min_c': thermal_min_c,
                'thermal_max_c': thermal_max_c,
                'thermal_avg_c': thermal_avg_c,
            }))
            background_tasks.add_task(_align_new_capture, THERMAL_DIR / thermal_file, stem, timestamp)
        else:
            buf = io.BytesIO()
            img.save(buf, format='JPEG', quality=95)
            data = buf.getvalue()
            filename = stem + '.jpg'
    else:
        # Path().name strips any directory components the client sent. Without it a filename
        # like "../../.ssh/authorized_keys" escapes UPLOAD_DIR and turns this endpoint into an
        # arbitrary file write. The WebP branches above are already safe — they rebuild the
        # name from Path(...).stem — so this is the only path that sees the raw client string.
        filename = Path(image.filename or '').name
        if not filename:
            raise HTTPException(status_code=422, detail="Invalid filename")

    (UPLOAD_DIR / filename).write_bytes(data)

    meta_parts = [p for p in [
        device_id,
        mode,
        f"motion={motion_score}" if motion_score else None,
        timestamp,
        f"thermal[{thermal_min_c}..{thermal_max_c}C]" if thermal_min_c else None,
    ] if p]
    logging.info(f"Received upload: {filename}" +
                 (f" (+thermal {thermal_file})" if thermal_file else "") +
                 (f" [{', '.join(meta_parts)}]" if meta_parts else ""))

    received_at = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
    with open(UPLOAD_LOG, 'a') as f:
        f.write(f"{received_at}\t{filename}\t{device_id}\t{mode}\t{motion_score}\t{timestamp}\n")

    return {"status": "ok"}
