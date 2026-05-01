import asyncio
import io
import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from PIL import Image as PILImage

from inference import process_image, IMAGE_EXTS
from config import (
    ENABLE_PROCESSING,
    UPLOAD_DIR,
    PROCESSED_DIR,
    FAILED_DIR,
    POLL_INTERVAL,
    MAX_PROCESS_ATTEMPTS,
    UPLOAD_LOG,
    PROCESSED_LOG,
)

UPLOAD_DIR.mkdir(exist_ok=True)
PROCESSED_DIR.mkdir(exist_ok=True)

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

_failed_counts: dict[str, int] = {}
_processed: set[str] = set()


def _load_processed_log() -> set[str]:
    if not PROCESSED_LOG.exists():
        return set()
    names = set()
    with open(PROCESSED_LOG) as f:
        for line in f:
            parts = line.strip().split('\t')
            if parts:
                names.add(parts[-1])
    return names


def _record_processed(filename: str):
    processed_at = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
    with open(PROCESSED_LOG, 'a') as f:
        f.write(f"{processed_at}\t{filename}\n")
    _processed.add(filename)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _processed
    _processed = _load_processed_log()
    logging.info(f'Loaded {len(_processed)} entries from processed log')
    asyncio.create_task(watch_uploads())
    logging.info('Background upload watcher started')
    yield


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
                    if file_path.name in _processed:
                        continue

                    fail_count = _failed_counts.get(file_path.name, 0)
                    if fail_count >= MAX_PROCESS_ATTEMPTS:
                        FAILED_DIR.mkdir(exist_ok=True)
                        dest = FAILED_DIR / file_path.name
                        file_path.rename(dest)
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
                        _failed_counts.pop(file_path.name, None)
                        _record_processed(file_path.name)
                        logging.info(f'Recorded {file_path.name} in processed log')
                    except Exception:
                        logging.exception(f'Failed to process {file_path.name}')
                        _failed_counts[file_path.name] = _failed_counts.get(file_path.name, 0) + 1
        except Exception:
            logging.exception('Error while watching uploads')

        await asyncio.sleep(POLL_INTERVAL)
