# Monitoring Pipeline — Server

Receives images from Raspberry Pi field devices, runs YOLOv8 object detection on each image, and persists detection results in a SQLite database with cross-image track linking.

## Setup

**Requirements:** Python 3.10+, a YOLO model weights file at `models/best.pt`.

```bash
cd Server
python3 -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Running

```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

Expose publicly (e.g. for a Pi on cellular):
```bash
ngrok http --scheme http 8000
```

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/upload` | Receive an image from a client device |
| `GET` | `/health` | Liveness check |
| `GET` | `/processing` | Check whether inference is enabled |
| `POST` | `/processing?enabled=true\|false` | Toggle inference on/off at runtime |

### Upload fields

The `/upload` endpoint accepts `multipart/form-data` with:

| Field | Type | Description |
|-------|------|-------------|
| `image` | file | JPEG or PNG (WebP is accepted and converted to JPEG) |
| `device_id` | text | Identifier of the sending device |
| `mode` | text | Recording mode (`image_motion`, `image_interval`) |
| `motion_score` | text | Motion ratio that triggered capture (motion mode only) |
| `timestamp` | text | ISO 8601 capture time |

Example:
```bash
curl -X POST http://localhost:8000/upload \
  -F "image=@image_20260103T112511Z.jpg" \
  -F "device_id=pi-barn" \
  -F "mode=image_motion" \
  -F "motion_score=0.042" \
  -F "timestamp=2026-01-03T11:25:11Z"
```

## Processing pipeline

1. Client POSTs an image → saved to `uploads/`
2. Background watcher detects the new file
3. When `ENABLE_PROCESSING=true`, runs YOLOv8 inference via `process_image()`
4. Detection labels saved to `processed/{image_name}/labels/`
5. Annotated image saved to `processed/{image_name}/`
6. `TrackerDB` links detections to existing or new objects across images
7. Original image moved to `processed/{image_name}/`

If processing is disabled, images accumulate in `uploads/` until you re-enable it.

Files that fail inference 3 times are moved to `failed/` to prevent an infinite retry loop.

## Controlling inference

Disable at startup:
```bash
ENABLE_PROCESSING=false uvicorn app:app --host 0.0.0.0 --port 8000
```

Toggle at runtime without restarting:
```bash
curl -X POST "http://localhost:8000/processing?enabled=false"
curl -X POST "http://localhost:8000/processing?enabled=true"
```

## Folder structure

```
uploads/          — incoming images waiting to be processed
processed/
  {image_name}/
    labels/       — YOLO detection labels (.txt)
    {image}.jpg   — annotated image
failed/           — images that could not be processed after 3 attempts
models/
  best.pt         — YOLOv8 model weights
objects.db        — SQLite tracking database
upload_log.txt    — TSV log of every received upload
```

## Image filename convention

Filenames should embed a UTC timestamp so the tracker can order them correctly:
```
image_YYYYMMDDThhmmssZ.jpg
```
Example: `image_20260103T112511Z.jpg`

The recorder on the Pi generates this format automatically.

## Utilities

**Inspect the tracking database:**
```bash
python inspect_db.py
```

**Batch-process a directory of images manually:**
```bash
python inference.py uploads/ --project processed --db objects.db
```

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| Images not processed | Check `GET /processing` — inference may be disabled |
| CUDA errors | Server auto-detects GPU; if none is available it falls back to CPU |
| Image stuck in `uploads/` after 3 errors | Check `failed/` folder and server logs |
| Port conflict | Change port: `uvicorn app:app --port 8001` |
| Database locked | Ensure only one server instance is running |
