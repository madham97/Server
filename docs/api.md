# API Reference

Base URL: `http://localhost:8000` (or your ngrok URL for remote devices).

---

## Core

### `GET /health`
Liveness check.

**Response**
```json
{ "status": "ok" }
```

---

### `GET /processing`
Returns whether background inference is currently enabled.

**Response**
```json
{ "enabled": false }
```

---

### `POST /processing?enabled=true|false`
Enable or disable background inference at runtime. Does not require a server restart.

**Query params**

| Param | Type | Description |
|---|---|---|
| `enabled` | bool | `true` to start processing uploads, `false` to pause |

**Response**
```json
{ "enabled": true }
```

**Notes:**
- The server starts with `enabled: false`. You must explicitly enable it.
- Disabling does not cancel any in-flight inference task; it prevents new ones from starting.

---

## Upload

### `POST /upload`
Receive an image from a field device.

**Content-Type:** `multipart/form-data`

| Field | Type | Required | Description |
|---|---|---|---|
| `image` | file | yes | JPEG, PNG, or WebP. WebP is converted to JPEG on receipt. Thermal-fused RGBA WebP is split — see Notes. |
| `device_id` | text | no | Identifier of the sending device |
| `mode` | text | no | `image_motion` or `image_interval` |
| `motion_score` | text | no | Motion ratio that triggered capture (float as string) |
| `timestamp` | text | no | ISO 8601 capture time |
| `format` | text | no | Format hint sent by the Pi (currently informational only, not read by the server) |
| `thermal_min_c` | text | no | Minimum temperature (°C) in the thermal frame, if `image` is a thermal-fused RGBA WebP |
| `thermal_max_c` | text | no | Maximum temperature (°C) in the thermal frame, if `image` is a thermal-fused RGBA WebP |
| `thermal_avg_c` | text | no | Average temperature (°C) in the thermal frame, if `image` is a thermal-fused RGBA WebP |

**Response**
```json
{ "status": "ok" }
```

**Side effects:**
- Saves image to `uploads/<filename>`.
- Appends a row to `upload_log.txt`.
- If the uploaded WebP has an alpha channel (thermal-fused frame): saves the alpha channel to
  `thermal/<stem>_thermal.png` and writes `thermal/<stem>_thermal.json` with `source_image`,
  `device_id`, `timestamp`, and the `thermal_min_c`/`max_c`/`avg_c` values.

**Notes:**
- Thermal-fused frames arrive as RGBA WebP: visible image in RGB, normalized thermal map in
  alpha. JPEG can't hold alpha, so the RGB and thermal channel are split and saved separately.
  `thermal/` is a sibling of `uploads/`, not a subdirectory, so the background watcher (which
  only reads `uploads/`) never sees thermal files.

---

## Annotation

### `GET /annotate`
Opens the annotation UI in the browser. Returns HTML.

---

### `GET /annotate/next?shuffle=true&after=`
Returns the next image to annotate.

**Query params**

| Param | Default | Description |
|---|---|---|
| `shuffle` | `true` | If true, picks randomly. If false, steps alphabetically, using `after` as the cursor. |
| `after` | `""` | When `shuffle=false`, returns the first name that sorts after this value. Wraps around. |

**Response**
```json
{
  "image_name": "frame_0042.jpg",
  "classes": ["rat", "human"]
}
```

**404** if no images are pending annotation.

---

### `GET /annotate/specific/{image_name}`
Load a specific image by name (must exist in `uploads/`).

**Response**
```json
{
  "image_name": "frame_0042.jpg",
  "classes": ["rat", "human"]
}
```

---

### `GET /annotate/image/{image_name}`
Serve the raw image file from `uploads/`. Used by the annotation UI to display the image.

**Response:** image file (JPEG)

---

### `POST /annotate/{image_name}`
Save bounding box labels for an image.

**Body (JSON)**
```json
{
  "labels": [
    { "class_id": 0, "x": 0.5, "y": 0.4, "w": 0.2, "h": 0.3 },
    { "class_id": 1, "x": 0.7, "y": 0.6, "w": 0.1, "h": 0.25 }
  ]
}
```

All coordinates are YOLO-normalized (0.0–1.0, center-based).

| Field | Description |
|---|---|
| `class_id` | Index into `classes.json`. 0 = rat, 1 = human. |
| `x`, `y` | Bounding box center, normalized. |
| `w`, `h` | Bounding box width/height, normalized. |

Submitting an empty `labels` array marks the image as annotated with no detections (valid negative example).

**Response**
```json
{ "status": "ok", "labels_saved": 2 }
```

**Side effects:**
- Writes `annotated/<stem>/labels/<stem>.txt`.
- Appends to `annotation_log.txt`.
- Overwrites any previous label file for the same image.

---

### `GET /annotate/stats`
Returns annotation progress.

**Response**
```json
{
  "pending": 47,
  "annotated": 120,
  "class_counts": {
    "rat": 84,
    "human": 63
  }
}
```

`class_counts` tallies individual bounding box labels, not images.

---

## Dataset

### `POST /dataset/export?val_split=0.2`
Builds a YOLO-format training dataset from all annotated images.

**Query params**

| Param | Default | Description |
|---|---|---|
| `val_split` | `0.2` | Fraction of images held out for validation. Must be between 0 and 1 exclusive. |

**Response**
```json
{
  "train": 96,
  "val": 24,
  "total": 120,
  "dataset_yaml": "C:\\Users\\hamma\\Python\\Server\\dataset\\dataset.yaml"
}
```

**Side effects:**
- Rebuilds `dataset/` from scratch (destructive).
- Writes `dataset/dataset.yaml`.

**400** if no annotated images exist.

---

### `GET /dataset/stats`
Current state of the exported dataset.

**Response (exported)**
```json
{
  "exported": true,
  "train": 96,
  "val": 24,
  "dataset_yaml": "C:\\Users\\hamma\\Python\\Server\\dataset\\dataset.yaml"
}
```

**Response (not yet exported)**
```json
{ "exported": false }
```

---

## Training

### `POST /train/start?base_model=yolov8n.pt&epochs=100&imgsz=640`
Start a training run in the background.

**Query params**

| Param | Default | Description |
|---|---|---|
| `base_model` | `yolov8n.pt` | YOLO checkpoint to fine-tune from. Can be a local path or an ultralytics model name. |
| `epochs` | `100` | Training epochs. |
| `imgsz` | `640` | Input image size (square). |

**Response**
```json
{ "status": "started", "base_model": "yolov8n.pt", "epochs": 100, "imgsz": 640 }
```

**400** if `dataset/dataset.yaml` doesn't exist (run export first).  
**409** if training is already running.

---

### `GET /train/status`
Current training state.

**Response**
```json
{
  "state": "running",
  "epoch": 42,
  "total_epochs": 100,
  "run_dir": "C:\\...\\models\\runs\\20250524T143022",
  "metrics": {
    "metrics/mAP50(B)": 0.612,
    "metrics/mAP50-95(B)": 0.389
  },
  "error": null
}
```

`state` values: `idle`, `running`, `complete`, `failed`.

---

### `POST /train/promote`
Promote `models/candidate.pt` to `models/best.pt`.

**Conditions:**
- Training must not be currently running.
- `models/candidate.pt` must exist.
- `metrics/mAP50(B)` from the last run must be ≥ `MIN_MAP` (0.3).

**Response**
```json
{ "status": "promoted", "map50": 0.612, "model": "models/best.pt" }
```

**404** if no candidate exists.  
**409** if training is still running.  
**400** if mAP50 is below threshold.

**Side effects:**
- Archives old `models/best.pt` to `models/archive/<timestamp>_best.pt`.
- Copies `models/candidate.pt` → `models/best.pt`.

---

## Inference

### `POST /infer/test?n=5`
Run inference on a random sample of images from `uploads/`. Useful for verifying the current model works before enabling the background watcher.

**Query params**

| Param | Default | Description |
|---|---|---|
| `n` | `5` | Number of images to sample. Capped at the number of images available. |

**Response**
```json
{
  "processed": 5,
  "results": [
    { "image": "frame_0001.jpg", "status": "ok" },
    { "image": "frame_0002.jpg", "status": "ok" },
    { "image": "frame_0003.jpg", "status": "error", "detail": "Weights file not found: ..." }
  ]
}
```

**404** if `uploads/` is empty.

**Notes:**
- This endpoint does NOT update `processed_log.txt` — it's a test endpoint. The same images can be processed again by the background watcher.
- Results are saved to `processed/` the same way as the background watcher.
