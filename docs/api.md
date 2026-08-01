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
- If a thermal frame was split out **and** a calibration is currently saved
  (`calibration/homography.json`), schedules a background task that warps the new thermal
  frame with the saved homography and writes `thermal/<stem>_thermal_aligned.png`. Runs after
  the response is returned (`BackgroundTasks`), so it doesn't add latency to the upload
  request. Silently does nothing if no calibration has been saved yet.

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

---

## Utilities

### `GET /thermal?limit=100`
Browse RGB/thermal capture pairs side by side, newest first. Read-only, no side effects.

**Query params**

| Param | Default | Description |
|---|---|---|
| `limit` | `100` | Maximum number of pairs to show |

**Response:** HTML page. Each thermal sidecar in `thermal/` is matched to its source JPEG (via the sidecar's `source_image` field) and rendered as a card with both images, `device_id`, `timestamp`, and the min/max/avg °C range. If the source JPEG no longer exists in `uploads/` (e.g. deleted for disk space), that side shows a "rgb missing" placeholder instead of a broken image.

---

### `GET /thermal/image/{name}`
Serve a raw thermal PNG from `thermal/`. Used by the `/thermal` page; `name` is validated to resolve inside `thermal/` before serving.

**Response:** image file (PNG). **404** if the name doesn't resolve inside `thermal/` or the file doesn't exist.

---

## Thermal Calibration

Computes and applies the thermal-to-RGB homography (`thermal_align.py`, `routers/thermal_calibrate.py`). Calibration is stored as a list of **profiles** in `calibration/profiles.json`, not a single transform — each profile carries a `[effective_from, effective_until)` window (an ISO timestamp and either another ISO timestamp or `null` for open-ended), and a capture is aligned using whichever profile's window contains *that capture's own* timestamp (`find_profile_for_timestamp`), not necessarily the newest profile. This is what lets the camera be recalibrated after being physically moved without corrupting the alignment of footage captured before the move — old captures keep using the profile that was actually in effect when they were taken.

On first read, if `calibration/profiles.json` doesn't exist yet but the old single-file `calibration/homography.json` does, it's migrated into one profile covering all time (`effective_from` the Unix epoch, open-ended) — an existing deployment's calibration keeps working unchanged until a real recalibration creates a second, time-scoped profile.

Consumed by `align_thermal`/`align_all` and by the background alignment task on `/upload` (see above), both of which resolve each capture's applicable profile independently.

### `GET /thermal/calibrate`
Serves the calibration UI (`static/thermal_calibrate.html`) — draw matching boxes around the animal on RGB/thermal pairs, or trigger automatic calibration, against a chosen effective window.

---

### `GET /thermal/calibrate/candidates?limit=200`
List capture pairs available to calibrate against.

**Response:** JSON array of `{ stem, source_image, timestamp, device_id }`, newest first, excluding any capture whose thermal frame is flagged as a corrupted SPI/CRC read (see `is_thermal_frame_corrupted`) or whose RGB/thermal file is missing.

---

### `GET /thermal/calibrate/profiles`
List all saved calibration profiles, newest `effective_from` first.

**Response:** JSON array of `{ id, label, effective_from, effective_until, calibrated_at, method, point_count, inlier_count }`.

---

### `DELETE /thermal/calibrate/profiles/{profile_id}`
Delete a calibration profile. Captures whose timestamp falls in its window are left with whatever aligned file they already have — they become unaligned only once something re-runs `align_all` and finds no profile covers them anymore. **404** if the id doesn't exist.

---

### `POST /thermal/calibrate`
Fit a homography from manually-clicked point pairs (at least 4), saved as a calibration profile.

**Body**
```json
{
  "pairs": [{ "stem": "...", "rgb": [x, y], "thermal": [x, y] }, ...],
  "effective_from": "2026-08-01T00:00:00Z",
  "effective_until": null,
  "profile_id": null,
  "label": "",
  "preview_stem": "..."
}
```
`effective_from` is required. `effective_until` defaults to `null` (open-ended). `profile_id`, if given, edits that existing profile in place (new homography, same id, window can also change) instead of creating a new one — validated against every *other* profile's window either way.

**Response:** `{ status, profile_id, point_count, image_count, aligned_count, preview, points }` — `points` includes per-pair reprojection error (`error_px`) and RANSAC inlier status, worst first.

**Errors:** `400` if the effective window overlaps another existing profile's window, or the usual point-count/homography-estimation failures.

**Side effects:** saves the profile and re-aligns every capture whose applicable profile is this one (`align_all(force=True, profile_id=...)`) — not every file in `thermal/`, only the ones this profile actually governs.

---

### `POST /thermal/calibrate/boxes`
Fit an affine transform from matched bounding boxes (at least 3) — the box's center **and** size both constrain the fit, which tolerates imprecise drawing far better than clicking a single point on a blurry thermal blob. This is what the calibration UI actually uses. Same profile semantics as above.

**Body**
```json
{
  "pairs": [{ "stem": "...", "rgb": [x1,y1,x2,y2], "thermal": [x1,y1,x2,y2] }, ...],
  "effective_from": "2026-08-01T00:00:00Z",
  "effective_until": null,
  "profile_id": null,
  "label": "",
  "preview_stem": "..."
}
```

**Response:** `{ status, profile_id, point_count, image_count, aligned_count, preview, boxes }` — `boxes` includes IoU between the transformed thermal box and the RGB box, and inlier status, worst first.

**Errors / side effects:** same as `POST /thermal/calibrate`.

---

### `POST /thermal/calibrate/auto`
Extracts correspondences automatically via background subtraction — no manual clicking. For every capture, finds the centroid of whatever stands out most from a per-pixel background reference in each spectrum independently, keeps the frame only if both sides found one plausible blob, and fits an affine transform (`cv2.estimateAffine2D` + RANSAC) through the pooled centroids. Same profile semantics as the point/box endpoints.

**Body**
```json
{ "effective_from": "2026-08-01T00:00:00Z", "effective_until": null, "profile_id": null, "label": "" }
```

**Response:** `{ status, profile_id, point_count, pairs_considered, pairs_matched, aligned_count, points }`.

**Errors:** `400` if fewer than `min_pairs` (default 8) capture pairs are available, share a common resolution, or produce a usable blob on both sides; if the RANSAC fit's inlier ratio is below 30% (the fit is rejected outright rather than saved, to avoid silently calibrating from noise); or if the effective window overlaps another profile's.

**Side effects:** same as the point/box endpoints.

---

### `GET /thermal/calibrate/stats?profile_id=&at=`
Fit summary for one calibration profile. With neither param, defaults to whichever profile covers right now; `at` (an ISO timestamp) checks a different moment instead; `profile_id` looks up a specific profile directly regardless of its window. Use `GET /thermal/calibrate/profiles` to see all of them.

**Response:** as before, plus `id`, `label`, `effective_from`, `effective_until`. **404** if no profile matches.

---

### `GET /config-help`
Interactive page for building `SET key=value key=value ...` SMS commands to text to the Pi's SIM number, per the SMS remote-configuration protocol in `docs/stakeholder-project-guide.md`. No request params; entirely client-side JS, no backend state.
