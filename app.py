import asyncio
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI

from inference import process_image, IMAGE_EXTS
from config import (
    UPLOAD_DIR,
    PROCESSED_DIR,
    ANNOTATED_DIR,
    FAILED_DIR,
    POLL_INTERVAL,
    MAX_PROCESS_ATTEMPTS,
    MODEL_WEIGHTS,
    MODEL_CONF,
    MODEL_IOU,
)
import state
from routers import upload, annotate, export, train, infer, config_help

UPLOAD_DIR.mkdir(exist_ok=True)
PROCESSED_DIR.mkdir(exist_ok=True)
ANNOTATED_DIR.mkdir(exist_ok=True)

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')


@asynccontextmanager
async def lifespan(app: FastAPI):
    state._processed = state.load_processed_log()
    logging.info(f'Loaded {len(state._processed)} entries from processed log')
    asyncio.create_task(watch_uploads())
    logging.info('Background upload watcher started')
    yield


app = FastAPI(lifespan=lifespan)
app.include_router(upload.router)
app.include_router(annotate.router)
app.include_router(export.router)
app.include_router(train.router)
app.include_router(infer.router)
app.include_router(config_help.router)


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/processing")
async def get_processing():
    return {"enabled": state.processing_enabled}


@app.post("/processing")
async def set_processing(enabled: bool):
    state.processing_enabled = enabled
    return {"enabled": state.processing_enabled}


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
            if state.processing_enabled:
                files = sorted([
                    p for p in UPLOAD_DIR.iterdir()
                    if p.is_file() and p.suffix.lower() in IMAGE_EXTS
                ])
                for file_path in files:
                    if file_path.name in state._processed:
                        continue

                    fail_count = state._failed_counts.get(file_path.name, 0)
                    if fail_count >= MAX_PROCESS_ATTEMPTS:
                        FAILED_DIR.mkdir(exist_ok=True)
                        dest = FAILED_DIR / file_path.name
                        file_path.rename(dest)
                        logging.warning(
                            f"Moved permanently failed file to failed/: {file_path.name} "
                            f"({fail_count} attempts)"
                        )
                        del state._failed_counts[file_path.name]
                        continue

                    stable = await _is_file_stable(file_path)
                    if not stable:
                        logging.info(f'Skipping unstable file: {file_path.name}')
                        continue

                    logging.info(f'Processing {file_path.name}...')
                    try:
                        await asyncio.to_thread(process_image, str(file_path), weights=MODEL_WEIGHTS, project=str(PROCESSED_DIR), conf=MODEL_CONF, iou=MODEL_IOU)
                        state._failed_counts.pop(file_path.name, None)
                        state.record_processed(file_path.name)
                        logging.info(f'Recorded {file_path.name} in processed log')
                    except Exception:
                        logging.exception(f'Failed to process {file_path.name}')
                        state._failed_counts[file_path.name] = state._failed_counts.get(file_path.name, 0) + 1
        except Exception:
            logging.exception('Error while watching uploads')

        await asyncio.sleep(POLL_INTERVAL)
