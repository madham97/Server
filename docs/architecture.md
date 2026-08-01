# Architecture

## Overview

A FastAPI server that receives images from Raspberry Pi field devices, runs YOLOv8 inference in the background, provides a browser-based annotation UI, and supports a full training loop to produce and promote custom detection models.

The two classes being detected are **rat** and **human**, defined in `classes.json`.

---

## Component Map

```
app.py              FastAPI app, lifespan startup, background watcher loop
config.py           All constants — paths, thresholds, model defaults
state.py            Mutable runtime state shared between the watcher and API
inference.py        YOLO inference wrapper (also runnable as a CLI script)
trainer.py          Background training thread, status reporting
thermal_align.py     Thermal-to-RGB homography: calibrate (checkerboard, manual points/boxes,
                     or automatic background-subtraction), align, corruption detection.
                     Also runnable as a CLI (`calibrate`, `align`, `auto-calibrate`).
routers/
  upload.py         POST /upload — receives images from Pi devices; also fires a background
                     task that aligns any new thermal frame with the saved calibration
  annotate.py       /annotate/* — annotation UI and label persistence
  export.py         /dataset/* — YOLO-format dataset export
  train.py          /train/* — training lifecycle management
  infer.py          /infer/* — ad-hoc inference for testing
  thermal_view.py   GET /thermal, /thermal/image/{name} — read-only RGB/thermal pair viewer
  thermal_calibrate.py  /thermal/calibrate/* — calibration UI and fit endpoints
static/
  annotate.html     Browser annotation UI (single-page)
  thermal_calibrate.html  Browser calibration UI — draw box pairs or run auto-calibration
classes.json        Class list ["rat", "human"]
calibration/
  homography.json   Saved 3x3 transform (thermal → RGB pixel space) + fit diagnostics
```

---

## Data Flow

### 1. Image ingestion

```
Pi device
  └─ POST /upload (multipart/form-data)
       ├─ WebP converted to JPEG on receipt
       ├─ If RGBA (thermal-fused frame): split into RGB + alpha
       │    ├─ RGB → JPEG, saved to uploads/<filename> (unchanged path below)
       │    ├─ Alpha → thermal/<stem>_thermal.png + thermal/<stem>_thermal.json
       │    │    (thermal/ is a sibling of uploads/, never scanned by the watcher)
       │    └─ Background task: if calibration/homography.json exists, warp the new
       │         thermal frame with it → thermal/<stem>_thermal_aligned.png
       │         (runs after the response is sent; no-op if no calibration saved yet)
       ├─ Saved to uploads/<filename>
       └─ Logged to upload_log.txt
```

### Thermal calibration and alignment

```
GET  /thermal/calibrate           Calibration UI: draw RGB/thermal box pairs, or
                                   POST /thermal/calibrate/auto for background-subtraction
POST /thermal/calibrate/boxes     Fit affine transform from box pairs (calibrate_from_boxes)
POST /thermal/calibrate           Fit homography from point pairs (calibrate_from_points)
POST /thermal/calibrate/auto      Fit affine transform from auto-detected blob centroids
                                   (auto_calibrate) — rejects the fit outright if under 30%
                                   of correspondences agree, rather than saving a bad one
       │
       ├─ Saves calibration/homography.json (transform + per-point/box fit diagnostics)
       └─ align_all(force=True): re-warps every thermal/*_thermal.png with the new transform
```

Every calibration method (checkerboard, manual points, manual boxes, automatic) converges on
the same `calibration/homography.json`, consumed by `align_thermal`/`align_all` and by the
`/upload` background alignment task. The manual box method is the one actually exposed in the
UI's main flow; automatic calibration is available but its correspondence quality depends
heavily on how well background subtraction can isolate the subject in both spectra — see
`docs/decisions.md` for what was tried.

### 2. Background inference

```
watch_uploads() [asyncio task, runs every POLL_INTERVAL=5s]
  ├─ Skips if state.processing_enabled is False
  ├─ Reads uploads/, filters by IMAGE_EXTS
  ├─ Skips filenames already in state._processed (loaded from processed_log.txt)
  ├─ Skips files still being written (_is_file_stable compares size 1s apart)
  ├─ Calls process_image() via asyncio.to_thread (non-blocking)
  │    └─ YOLO model.predict() → saves annotated image + labels to processed/<stem>/
  ├─ On success: appends to processed_log.txt, adds to state._processed
  └─ On failure: increments state._failed_counts[filename]
       └─ After MAX_PROCESS_ATTEMPTS=3: moves file to failed/
```

### 3. Annotation

```
Browser → GET /annotate
  └─ /annotate/next returns a random unannotated image name
       ├─ "Unannotated" = in uploads/ but NOT in annotation_log.txt
       └─ User draws bounding boxes, selects class, submits
            └─ POST /annotate/<name> saves YOLO-format .txt to annotated/<stem>/labels/
                 └─ Appends to annotation_log.txt
```

### 4. Training loop

```
POST /dataset/export
  └─ Copies annotated images + labels into dataset/{images,labels}/{train,val}/
       └─ Writes dataset/dataset.yaml

POST /train/start
  └─ trainer._run() in daemon thread
       ├─ YOLO.train() with workers=0 (Windows DataLoader fix)
       ├─ Saves best.pt to models/runs/<timestamp>/weights/best.pt
       └─ Copies best.pt → models/candidate.pt

GET /train/status
  └─ Returns epoch, metrics, state (idle|running|complete|failed)

POST /train/promote
  ├─ Checks mAP50 >= MIN_MAP (0.3)
  ├─ Archives existing models/best.pt → models/archive/<timestamp>_best.pt
  └─ Copies models/candidate.pt → models/best.pt
```

---

## State Management

`state.py` holds three shared mutable objects:

| Variable | Type | Purpose |
|---|---|---|
| `_processed` | `set[str]` | Filenames already run through inference; loaded from `processed_log.txt` at startup |
| `_failed_counts` | `dict[str, int]` | In-memory retry counter; reset on server restart |
| `processing_enabled` | `bool` | Runtime toggle; starts `False`, set via `POST /processing?enabled=true` |

Persistence is file-based (append-only log files), not a database. This was an intentional simplification — see [decisions.md](decisions.md).

---

## Directory Layout at Runtime

```
uploads/        All received images (never deleted automatically)
thermal/        Thermal PNG + JSON sidecar split from RGBA uploads, keyed by source stem;
                <stem>_thermal_aligned.png alongside once a calibration exists
calibration/
  homography.json  Saved thermal-to-RGB transform + fit diagnostics (see Thermal Calibration)
processed/      YOLO output per image — processed/<stem>/{<stem>.jpg, labels/<stem>.txt}
annotated/      Human labels — annotated/<stem>/labels/<stem>.txt
failed/         Images that exceeded MAX_PROCESS_ATTEMPTS
dataset/        Exported training dataset (overwritten on each export)
models/
  best.pt       Live inference weights
  candidate.pt  Weights from last completed training run
  runs/         Full training outputs (weights, plots, metrics CSVs)
  archive/      Superseded best.pt versions, timestamped
```

---

## Concurrency Model

- The FastAPI event loop handles all HTTP requests.
- `watch_uploads()` is an `asyncio.create_task` — runs on the same event loop.
- Inference (`process_image`) is dispatched via `asyncio.to_thread` so it doesn't block the event loop.
- Training runs in a `threading.Thread` (daemon) and uses a `threading.Lock` to protect `_status`. It does not interact with the asyncio loop.
- There is no locking between the watcher and annotation writes. The watcher only reads `uploads/` filenames; annotation writes to `annotated/`. Race conditions are not a practical concern at current scale.
