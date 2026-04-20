# Rodent Server

Receives image and video uploads from Raspberry Pi edge devices over GSM, runs YOLOv8 object detection on video files, and tracks detected objects across sessions in a SQLite database.

## Requirements

- Python 3.8+
- GPU with CUDA recommended for inference (falls back to CPU automatically)
- ngrok or equivalent for public access from GSM devices

## Installation

```bash
cd Server
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Running

```bash
python -m uvicorn app:app --host 0.0.0.0 --port 8000
```

**Expose publicly for GSM devices (no static IP):**
```bash
ngrok http --scheme=http 8000
```

Copy the ngrok URL into `server_url` in the client's `client.json`.

## Endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/upload` | Receive image or video from a client device |
| `GET` | `/health` | Liveness check |
| `GET` | `/config-help` | Interactive SMS config builder for client devices |
| `GET` | `/docs` | Swagger UI (auto-generated) |

### Upload format

Multipart form POST with the following fields:

| Field | Required | Description |
|---|---|---|
| `image` or `video` | Yes | The file (JPEG, WebP, or MP4) |
| `device_id` | No | Hostname of the sending device |
| `mode` | No | Recording mode (`image_motion`, `segment`, etc.) |
| `motion_score` | No | Fraction of pixels that changed (0–1) |
| `timestamp` | No | UTC capture time (ISO 8601) |

WebP images are automatically converted to JPEG on receipt.

## Upload log

Every received upload is appended to `upload_log.txt` as a tab-separated line:

```
received_at    filename    device_id    mode    motion_score    capture_timestamp
```

Example:
```
2026-04-20T14:56:23Z	image_20260420T145057Z.jpg	rodent2	image_motion	0.065	2026-04-20T14:50:58Z
```

## Video processing pipeline

1. Uploaded videos land in `uploads/`
2. A background watcher polls every 5 seconds
3. Stable files are processed with YOLOv8 (`models/best.pt`)
4. Detection labels are written to `processed/{video_name}/labels/`
5. Object tracks are linked across video chunks using IoU matching
6. Results are stored in `objects.db`
7. Original video is moved to `processed/{video_name}/`

### File naming convention

Video filenames should embed a UTC timestamp for correct chronological ordering:

```
video_YYYYMMDDThhmmssZ_[number].mp4
```

Example: `video_20260420T145057Z_1.mp4`

## Database

SQLite database at `objects.db` with three tables:

| Table | Description |
|---|---|
| `objects` | Persistent tracked entities with first/last seen times and confidence |
| `sightings` | Individual per-frame detections with bounding box and confidence |
| `active_tracks` | Current track state used to link detections across video chunks |

Inspect the database:

```bash
python inspect_db.py --db objects.db
```

## SMS config helper

Open `http://<server>/config-help` in a browser to build config patch SMS messages for client devices interactively. Select a config key, enter a value, and copy the JSON to send as a text to the device's SIM number.

## Folder structure

```
uploads/          Incoming files (cleared after processing)
processed/
  {video_name}/
    labels/       YOLO detection label files per frame
    {video}.avi   Annotated output video
models/
  best.pt         YOLOv8 weights
objects.db        Tracked object database
upload_log.txt    Tab-separated upload history
```

## Troubleshooting

| Symptom | Check |
|---|---|
| Videos not processing | Filename includes timestamp? Check server logs |
| No detections | `models/best.pt` present? Confidence threshold in `inference.py` |
| Database locked | Only one server instance running? |
| Port in use | `--port 8001` to use a different port |
