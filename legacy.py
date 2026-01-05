from fastapi import FastAPI, UploadFile, File
import shutil
from pathlib import Path

app = FastAPI()

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

@app.post("/upload")
async def upload(
    video: UploadFile = File(...),
    # metadata: UploadFile = File(...)
):
    video_path = UPLOAD_DIR / video.filename
    # metadata_path = UPLOAD_DIR / metadata.filename

    with open(video_path, "wb") as f:
        shutil.copyfileobj(video.file, f)

    # with open(metadata_path, "wb") as f:
    #     shutil.copyfileobj(metadata.file, f)

    return {"status": "ok"}
