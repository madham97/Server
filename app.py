import asyncio
import logging
import shutil
import time
from pathlib import Path
from datetime import datetime
from fastapi import FastAPI, UploadFile, File

from inference import process_video, VIDEO_EXTS
from tracker import TrackerDB, parse_timestamp_from_filename

app = FastAPI()

UPLOAD_DIR = Path("uploads")
PROCESSED_DIR = Path("processed")
UPLOAD_DIR.mkdir(exist_ok=True)
PROCESSED_DIR.mkdir(exist_ok=True)

POLL_INTERVAL = 5  # seconds to wait between polling for new files
DB_PATH = "objects.db"
MAX_GAP_SECONDS = 300  # max time between video chunks for track linking

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

# Global database connection
db: TrackerDB | None = None


@app.post("/upload")
async def upload(
    video: UploadFile = File(...),
):
    video_path = UPLOAD_DIR / video.filename
    with open(video_path, "wb") as f:
        shutil.copyfileobj(video.file, f)
    logging.info(f"Received upload: {video.filename}")
    return {"status": "ok"}


@app.get("/health")
async def health():
    return {"status": "ok"}


async def _is_file_stable(path: Path, wait: float = 1.0) -> bool:
    """Ensure file size isn't changing (simple protection against incomplete uploads)."""
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
    """Background task: poll the uploads folder, process new videos alphabetically, and move originals to processed."""
    logging.info('Starting uploads watcher')
    while True:
        try:
            # gather files matching known extensions, sorted alphabetically
            files = sorted([p for p in UPLOAD_DIR.iterdir() if p.is_file() and p.suffix.lower() in VIDEO_EXTS])
            for video_path in files:
                # check file is stable (likely finished uploading)
                stable = await _is_file_stable(video_path)
                if not stable:
                    logging.info(f'Skipping unstable file: {video_path.name}')
                    continue

                logging.info(f'Processing {video_path.name}...')
                try:
                    # run blocking processing in a thread so the event loop isn't blocked
                    await asyncio.to_thread(process_video, str(video_path), project=str(PROCESSED_DIR))

                    # update database with detections from this video
                    if db is not None:
                        video_time = parse_timestamp_from_filename(video_path.name)
                        db.process_yolo_labels_for_video(
                            str(PROCESSED_DIR),
                            video_path.stem,
                            video_path.name,
                            video_time,
                            max_gap_seconds=MAX_GAP_SECONDS
                        )
                        db.close_inactive_tracks(older_than_seconds=MAX_GAP_SECONDS * 2)
                        logging.info(f'Updated database for {video_path.name}')

                    # move original into its video subdirectory (ensure no overwrite)
                    video_subdir = PROCESSED_DIR / video_path.stem
                    video_subdir.mkdir(exist_ok=True)
                    dest = video_subdir / video_path.name
                    if dest.exists():
                        dest = video_subdir / f"{video_path.stem}-{int(time.time())}{video_path.suffix}"
                    shutil.move(str(video_path), str(dest))
                    logging.info(f'Moved original to {dest}')
                except Exception:
                    logging.exception(f'Failed to process {video_path.name}')
        except Exception:
            logging.exception('Error while watching uploads')

        await asyncio.sleep(POLL_INTERVAL)


@app.on_event("startup")
async def startup_event():
    global db
    # Initialize database
    db = TrackerDB(DB_PATH)
    logging.info(f'Initialized database: {DB_PATH}')
    # Launch background watcher
    asyncio.create_task(watch_uploads())
    logging.info('Background upload watcher started')


@app.on_event("shutdown")
async def shutdown_event():
    global db
    if db is not None:
        db.close()
        logging.info('Database closed')
