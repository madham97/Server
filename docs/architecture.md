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
routers/
  upload.py         POST /upload — receives images from Pi devices
  annotate.py       /annotate/* — annotation UI and label persistence
  export.py         /dataset/* — YOLO-format dataset export
  train.py          /train/* — training lifecycle management
  infer.py          /infer/* — ad-hoc inference for testing
static/
  annotate.html     Browser annotation UI (single-page)
classes.json        Class list ["rat", "human"]
```

---

## Data Flow

### 1. Image ingestion

```
Pi device
  └─ POST /upload (multipart/form-data)
       ├─ WebP converted to JPEG on receipt
       ├─ Saved to uploads/<filename>
       └─ Logged to upload_log.txt
```

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
