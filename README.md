# Monitoring Pipeline — Server

Receives images from Raspberry Pi field devices, runs YOLOv8 object detection, and provides an annotation + training pipeline for building custom detection models.

## Setup

**Requirements:** Python 3.13, CUDA-capable GPU recommended.

```bash
cd Server
py -3.13 -m venv venv
venv\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu126
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

### Device upload

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/upload` | Receive an image from a client device |
| `GET` | `/health` | Liveness check |
| `GET` | `/processing` | Check whether inference is enabled |
| `POST` | `/processing?enabled=true\|false` | Enable or disable background inference at runtime |

The `/upload` endpoint accepts `multipart/form-data` with:

| Field | Type | Description |
|-------|------|-------------|
| `image` | file | JPEG or PNG (WebP accepted and converted) |
| `device_id` | text | Identifier of the sending device |
| `mode` | text | Recording mode (`image_motion`, `image_interval`) |
| `motion_score` | text | Motion ratio that triggered capture |
| `timestamp` | text | ISO 8601 capture time |

### Annotation

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/annotate` | Open annotation UI in browser |
| `GET` | `/annotate/next` | Next random unannotated image + class list |
| `GET` | `/annotate/specific/{image_name}` | Load a specific image by name |
| `GET` | `/annotate/image/{image_name}` | Serve raw image file |
| `POST` | `/annotate/{image_name}` | Save labels for an image |
| `GET` | `/annotate/stats` | Pending count, annotated count, per-class totals |

### Inference

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/infer/test?n=5` | Run inference on `n` random images from `uploads/` |

### Dataset & training

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/dataset/export` | Export annotated images to YOLO training format |
| `GET` | `/dataset/stats` | Current dataset export state |
| `POST` | `/train/start` | Start training in the background |
| `GET` | `/train/status` | Current epoch, metrics, state |
| `POST` | `/train/promote` | Promote candidate model to `models/best.pt` |

## Training workflow

1. Annotate images at `http://localhost:8000/annotate`
2. Export dataset: `POST /dataset/export?val_split=0.2`
3. Start training: `POST /train/start?epochs=100&imgsz=640`
4. Monitor: `GET /train/status`
5. Promote: `POST /train/promote` (requires mAP50 ≥ 0.3)

Training parameters (`POST /train/start`):

| Param | Default | Description |
|-------|---------|-------------|
| `base_model` | `yolov8n.pt` | Base YOLO checkpoint to train from |
| `epochs` | `100` | Number of training epochs |
| `imgsz` | `640` | Input image size (square) |

## Folder structure

```
app.py                  — FastAPI entry point, background watcher
config.py               — All configuration
state.py                — Shared watcher state
trainer.py              — YOLO training logic
inference.py            — YOLO inference wrapper
routers/
  upload.py             — POST /upload
  annotate.py           — /annotate/* endpoints
  export.py             — /dataset/* endpoints
  train.py              — /train/* endpoints
  infer.py              — /infer/* endpoints
static/
  annotate.html         — Annotation UI
classes.json            — Class definitions (rat, human)
models/
  best.pt               — Live model weights (gitignored)
  candidate.pt          — Staged model after training (gitignored)
  runs/                 — Training run outputs (gitignored)
  archive/              — Superseded model versions (gitignored)
uploads/                — All received images (gitignored)
processed/              — YOLO inference outputs (gitignored)
annotated/              — Human-confirmed label files (gitignored)
dataset/                — Exported training dataset (gitignored)
```

## Configuration

All settings are in `config.py`:

| Setting | Default | Description |
|---------|---------|-------------|
| `POLL_INTERVAL` | `5` | Seconds between watcher cycles |
| `MAX_PROCESS_ATTEMPTS` | `3` | Retries before moving to `failed/` |
| `BASE_MODEL` | `yolov8n.pt` | Default base model for training |
| `TRAIN_EPOCHS` | `100` | Default training epochs |
| `TRAIN_IMGSZ` | `640` | Default training image size |
| `MIN_MAP` | `0.3` | Minimum mAP50 required for promotion |
| `MODEL_CONF` | `0.25` | Inference confidence threshold |
| `MODEL_IOU` | `0.45` | Inference IOU threshold |

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| Images not processed | Call `POST /processing?enabled=true` to enable background inference |
| Training fails immediately | Windows DataLoader issue — `workers=0` is set by default |
| YOLO saves to `runs/detect/...` | Ensure trainer uses absolute project path (already fixed) |
| Port conflict | `uvicorn app:app --port 8001` |
| CUDA not detected | Reinstall PyTorch with CUDA: `pip install torch --index-url https://download.pytorch.org/whl/nightly/cu126` |
