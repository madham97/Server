# Decisions, Lessons, and Known Issues

This file records what was tried, what was changed, and why. It is the most important reference for future development — read this before making changes to core pipeline logic.

---

## What Worked

### Flat log files instead of a database

**Decision:** Track processed and annotated images via append-only `.txt` log files (`processed_log.txt`, `upload_log.txt`, `annotation_log.txt`) rather than a database or ORM.

**Why it worked:** The server is single-process, single-instance, and image volume is low enough that reading a flat file at startup is instant. No schema migrations, no DB driver issues, no connection pooling. The logs are also human-readable and easy to inspect or repair manually.

**Trade-off:** If the server crashes mid-write, a partial line could appear. In practice this hasn't caused issues because the watcher skips already-processed files based on exact filename matches, and a partial last line simply won't parse correctly and will be ignored. If scale grows significantly (tens of thousands of images), this should be revisited in favour of SQLite.

---

### `asyncio.to_thread` for inference

**Decision:** Run `process_image()` via `asyncio.to_thread` rather than a subprocess or worker process.

**Why it worked:** Keeps inference non-blocking without the overhead of a process pool. YOLO inference is CPU/GPU-bound and releases the GIL, so threading is effective here. The event loop stays responsive for incoming uploads during inference.

---

### `threading.Thread` for training

**Decision:** Run YOLO training in a daemon thread rather than a subprocess.

**Why it worked:** Training can take minutes to hours. A daemon thread lets the server stay up and serve status polls (`GET /train/status`) while training runs. The `threading.Lock` around `_status` is sufficient since only one training run can be active at a time (enforced by the `running` state check).

---

### `workers=0` for YOLO training on Windows

**Decision:** Always pass `workers=0` to `YOLO.train()`.

**Why it worked:** The default PyTorch DataLoader uses `multiprocessing` with `spawn` on Windows, which causes a freeze or crash when called from a thread inside a running FastAPI app. Setting `workers=0` forces single-process data loading. This is slower but stable.

**Do not remove this without testing on Windows first.**

---

### Absolute paths anchored to `__file__`

**Decision:** All directory paths in `config.py` are computed relative to `Path(__file__).parent` rather than relative to the current working directory.

**Why it matters:** When the server is started from a directory other than the project root (e.g., `uvicorn Server.app:app` from the parent), relative paths like `Path("uploads")` resolve to the wrong location. Anchoring to `__file__` makes the server location-independent.

**This fixed a real bug where uploads were landing in the wrong directory.**

---

### WebP → JPEG conversion on upload

**Decision:** Convert `.webp` images to `.jpeg` on receipt in `upload.py` before saving.

**Why it worked:** Some Pi camera modes emit WebP. YOLO's inference pipeline handles JPEG/PNG natively but WebP support is inconsistent across platforms. Converting at the boundary keeps everything downstream simple.

---

### Model promotion gate (mAP50 ≥ 0.3)

**Decision:** `POST /train/promote` requires `mAP50 >= MIN_MAP` before overwriting `best.pt`.

**Why it works:** Prevents accidentally deploying a model that is worse than the current one (e.g., from a too-small dataset or misconfigured run). The threshold is conservative — 0.3 is a floor, not a target. As the dataset grows, raise `MIN_MAP` in `config.py`.

---

### Candidate/archive model pattern

**Decision:** Training outputs to `models/candidate.pt`, promotion copies it to `models/best.pt` and archives the old one.

**Why it works:** Decouples training completion from deployment. A training run finishing does not immediately affect live inference. Archive copies (`models/archive/<timestamp>_best.pt`) allow rollback without needing git LFS or external storage.

---

## What Didn't Work / Was Removed

### TrackerDB

**What it was:** An earlier version used a database (`TrackerDB`) to record upload and processing state.

**Why it was removed:** Added setup complexity (DB driver, connection management, schema) with no meaningful benefit at the current scale. Replaced by flat log files. Removed in commit `37be71e`.

---

### Static `ENABLE_PROCESSING` config flag

**What it was:** A boolean constant `ENABLE_PROCESSING = False` in `config.py` that controlled whether the background watcher ran inference.

**Why it was removed:** A static constant requires restarting the server to change. This was impractical during field testing when images were accumulating. Replaced with `state.processing_enabled` (mutable at runtime) and a `POST /processing?enabled=true|false` endpoint. Removed in commit `e3e8372`.

---

### YOLO saving to `runs/detect/` by default

**Problem:** Without explicit `project` and `name` arguments, YOLO saves inference results to `runs/detect/predict/` relative to the current working directory. This scattered outputs unpredictably.

**Fix:** Always pass `project=str(PROCESSED_DIR)` and `name=image_name` to `model.predict()`. Results now land in `processed/<stem>/` consistently. The same pattern applies in training: `project=str(project_dir)`, `name=timestamp`.

---

## Known Limitations and Future Work

### No deduplication on upload

If the same filename is uploaded twice, the second file silently overwrites the first in `uploads/`. The processed log tracks by filename, so the second copy will not be re-inferred (it looks already processed). Consider adding content hashing or timestamp-suffixed filenames if duplicate uploads become a problem.

### Failed counts reset on restart

`state._failed_counts` is in-memory only. If the server restarts while a file is in a failing state (but below the 3-attempt threshold), the counter resets and the file gets three more attempts. This is intentional conservatism — it's better to retry than to silently skip.

### Annotation log grows unbounded

`annotation_log.txt` is append-only. If an image is re-annotated (labels corrected), both entries exist in the log. `_load_annotation_log()` only checks filenames so the image will be treated as annotated regardless. The label file in `annotated/<stem>/` is overwritten correctly, so the actual labels are correct — only the timestamp of "last annotated" would be ambiguous.

### Dataset export is destructive

`POST /dataset/export` rebuilds the entire `dataset/` directory from scratch each time. Any manual edits to `dataset/dataset.yaml` or the split will be lost. Do not edit the exported dataset directly; always re-export.

### Single model slot

The server holds one live model (`models/best.pt`). There is no per-class or per-camera model routing. If the detection target diversifies, the model management layer will need to be extended.

### No authentication

All endpoints are unauthenticated. The server is intended to run behind ngrok or a private network. Do not expose it directly to the public internet without adding at least API key authentication to the upload and control endpoints.

### Training on CPU is very slow

YOLOv8n on CPU can take hours per epoch on a laptop. CUDA is strongly recommended. The PyTorch CUDA install requires the nightly index URL (see README setup section) — the stable channel does not always carry the right CUDA build for newer GPUs.
