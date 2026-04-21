import asyncio
import io
import logging
import os
import shutil
import time
from contextlib import asynccontextmanager
from pathlib import Path
from datetime import datetime, timezone

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from PIL import Image as PILImage

from inference import process_image, IMAGE_EXTS
from tracker import TrackerDB, parse_timestamp_from_filename

UPLOAD_DIR = Path("uploads")
PROCESSED_DIR = Path("processed")
FAILED_DIR = Path("failed")
UPLOAD_DIR.mkdir(exist_ok=True)
PROCESSED_DIR.mkdir(exist_ok=True)

POLL_INTERVAL = 5
DB_PATH = "objects.db"
MAX_GAP_SECONDS = 300
MAX_PROCESS_ATTEMPTS = 3
UPLOAD_LOG = Path("upload_log.txt")

# Set ENABLE_PROCESSING=false env var to receive images without running inference.
# Toggle at runtime via POST /processing?enabled=true|false
ENABLE_PROCESSING: bool = os.getenv("ENABLE_PROCESSING", "true").lower() in ("1", "true", "yes")

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

db: TrackerDB | None = None
_failed_counts: dict[str, int] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    global db
    db = TrackerDB(DB_PATH)
    logging.info(f'Initialized database: {DB_PATH}')
    asyncio.create_task(watch_uploads())
    logging.info('Background upload watcher started')
    yield
    if db is not None:
        db.close()
        logging.info('Database closed')


app = FastAPI(lifespan=lifespan)


@app.post("/upload")
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


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/processing")
async def get_processing():
    return {"enabled": ENABLE_PROCESSING}


@app.post("/processing")
async def set_processing(enabled: bool):
    global ENABLE_PROCESSING
    ENABLE_PROCESSING = enabled
    logging.info(f"Processing {'enabled' if enabled else 'disabled'}")
    return {"enabled": ENABLE_PROCESSING}


async def _is_file_stable(path: Path, wait: float = 1.0) -> bool:
    try:
        size1 = path.stat().st_size
    except FileNotFoundError:
        return False
    await asyncio.sleep(wait)
    try:
        size2 = path.stat().st_size
    except FileNotFoundError:
        return False
    return size1 == size2


async def watch_uploads():
    logging.info('Starting uploads watcher')
    while True:
        try:
            if ENABLE_PROCESSING:
                files = sorted([
                    p for p in UPLOAD_DIR.iterdir()
                    if p.is_file() and p.suffix.lower() in IMAGE_EXTS
                ])
                for file_path in files:
                    fail_count = _failed_counts.get(file_path.name, 0)
                    if fail_count >= MAX_PROCESS_ATTEMPTS:
                        FAILED_DIR.mkdir(exist_ok=True)
                        dest = FAILED_DIR / file_path.name
                        shutil.move(str(file_path), str(dest))
                        logging.warning(
                            f"Moved permanently failed file to failed/: {file_path.name} "
                            f"({fail_count} attempts)"
                        )
                        del _failed_counts[file_path.name]
                        continue

                    stable = await _is_file_stable(file_path)
                    if not stable:
                        logging.info(f'Skipping unstable file: {file_path.name}')
                        continue

                    logging.info(f'Processing {file_path.name}...')
                    try:
                        await asyncio.to_thread(process_image, str(file_path), project=str(PROCESSED_DIR))

                        if db is not None:
                            image_time = parse_timestamp_from_filename(file_path.name)
                            db.process_yolo_labels_for_video(
                                str(PROCESSED_DIR),
                                file_path.stem,
                                file_path.name,
                                image_time,
                                max_gap_seconds=MAX_GAP_SECONDS
                            )
                            db.close_inactive_tracks(older_than_seconds=MAX_GAP_SECONDS * 2)
                            logging.info(f'Updated database for {file_path.name}')

                        _failed_counts.pop(file_path.name, None)

                        file_subdir = PROCESSED_DIR / file_path.stem
                        file_subdir.mkdir(exist_ok=True)
                        dest = file_subdir / file_path.name
                        if dest.exists():
                            dest = file_subdir / f"{file_path.stem}-{int(time.time())}{file_path.suffix}"
                        shutil.move(str(file_path), str(dest))
                        logging.info(f'Moved original to {dest}')
                    except Exception:
                        logging.exception(f'Failed to process {file_path.name}')
                        _failed_counts[file_path.name] = _failed_counts.get(file_path.name, 0) + 1
        except Exception:
            logging.exception('Error while watching uploads')

        await asyncio.sleep(POLL_INTERVAL)
