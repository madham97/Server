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

### Thermal channel split into a sibling `thermal/` directory

**Decision:** Thermal-fused frames from the Pi arrive as RGBA WebP (visible image in RGB, normalized thermal map in alpha). JPEG can't hold alpha, so on upload the RGB is saved as a normal JPEG to `uploads/` (unchanged for the detection pipeline) and the alpha channel is saved to `thermal/<stem>_thermal.png` with a `thermal/<stem>_thermal.json` sidecar carrying `thermal_min_c`/`max_c`/`avg_c` needed to reconstruct real degrees from the 0-255 alpha values. `thermal/` is a top-level directory, a sibling of `uploads/` — not a subdirectory of it.

**Why it worked:** Before this, any RGBA upload raised `cannot write mode RGBA as JPEG` and returned HTTP 500, silently dropping the thermal data (plain RGB uploads were unaffected). Splitting preserves both channels without changing what the detection pipeline receives. Making `thermal/` a sibling of `uploads/`, following the same pattern as `processed/`, `annotated/`, and `failed/`, keeps its exclusion from `watch_uploads()` structural — the watcher only ever reads `uploads/` — rather than relying on the watcher's file-type filter to incidentally skip a subdirectory.

**Trade-off:** Nothing in the pipeline consumes `thermal/` yet; this change only stops the crash and data loss. A downstream consumer (annotation, detection fusion, or a UI) would need to be built separately if the thermal data is meant to be used, not just retained.

---

### `/thermal` viewer for browsing RGB/thermal pairs

**Decision:** Added `GET /thermal` (`routers/thermal_view.py`), a read-only HTML page listing every thermal sidecar newest-first, each paired with its source JPEG side by side (device id, timestamp, min/max/avg °C underneath). Serves the thermal PNG via `GET /thermal/image/{name}`; the RGB side reuses the existing `GET /annotate/image/{image_name}`.

**Why it worked:** The obvious way to make pairs "easy to see" would be to co-locate the RGB and thermal files in the same folder or a per-capture subdirectory. Both were rejected: same-folder risks the watcher picking up thermal PNGs again (see the sibling-directory decision above), and a per-capture subdirectory would require rewriting every place that currently builds a path as `UPLOAD_DIR / filename` (`annotate.py`, `export.py`, the watcher, `/infer/test`). A read-only viewer gets the same practical outcome — visual pairing — without touching storage layout or any pipeline code.

**Trade-off:** It's a manual pairing view (`stem` matched at request time via the sidecar's `source_image` field), not a stored/indexed relationship. If a JPEG has been deleted (e.g. during disk cleanup) the viewer shows a "rgb missing" placeholder rather than erroring.

---

### SMS config-helper rebuilt for the current `SET key=value` protocol

**Decision:** Added `GET /config-help` (`routers/config_help.py`), an interactive page for building SMS config commands to text to the Pi's SIM number.

**Why it worked:** A `/config-help` page already existed on `main` before the images-only refactor, but it built JSON-patch SMS bodies (`{"recording":{"mode":"VALUE"}}`) matching an older client protocol. The current Pi client (`docs/stakeholder-project-guide.md`) uses a plain-text `SET key=value key=value ...` protocol with a different, smaller set of keys (`mode`, `motion_threshold`, `motion_cooldown`, `detection_interval`, `image_interval`, `image_quality`, `webp_compress`, `webp_quality`). Porting the old page as-is would have generated SMS commands the current client can't parse, so it was rebuilt from scratch against the documented current protocol instead of merged in.

---

### `thermal/` and the log files bind-mounted in `docker-compose.yml`

**Decision:** `docker-compose.yml` now mounts `./thermal:/app/thermal`, `./upload_log.txt:/app/upload_log.txt`, `./processed_log.txt:/app/processed_log.txt`, and `./annotation_log.txt:/app/annotation_log.txt`, alongside the pre-existing `uploads/`, `processed/`, `annotated/`, `failed/`, `dataset/`, `models/` mounts.

**Why it worked:** `THERMAL_DIR` and the three log files live at `config.py`'s `_BASE` (i.e. `/app/` in the container) like everything else, but unlike `uploads/`/`processed/`/etc. they weren't in the compose file's volume list — so they existed only in the container's ephemeral filesystem. A container recreation (rebuild, `docker compose up -d`, host reboot) silently wiped them. This was discovered by inspecting a running container directly (`docker exec ... cat /app/config.py`) and finding it didn't match either the last built image or the current repo — someone had live-patched the container to stop 500s without a real rebuild, and the thermal captures that patch produced were about to be lost on the next recreation.

**Note:** bind-mounting a single file (not a directory) requires the host-side file to already exist before `docker compose up`, or Docker will create a directory in its place and the app's `open(path, 'a')` call will fail. `upload_log.txt`, `processed_log.txt`, `annotation_log.txt` must be `touch`ed on the host once before the first mount.

---

### Dual ngrok tunnels: `--scheme http` for the Pi, default (https) for browsers

**Decision:** Two separate `ngrok` agent processes run simultaneously against the same static reserved domain (`maryrose-rejoiceful-avah.ngrok-free.dev`): one started with `ngrok http --scheme http 8000` (http-only, no redirect), and a second with plain `ngrok http 8000` (https-capable, the default).

**Why it worked:** The Pi's GSM/cellular modem uploader needs plain, unredirected HTTP — `--scheme http` was chosen originally for exactly that reason. But `.dev` is a Google-owned gTLD that is unconditionally HSTS-preloaded in every modern browser: any `*.ngrok-free.dev` URL is force-upgraded to https client-side, permanently, with no way to opt out via site settings, HSTS deletion, or "always use secure connections" toggles (those only affect dynamic/per-site HSTS, not TLD-level preload entries baked into the browser). So an http-only tunnel is invisible to browsers — they get ngrok's `ERR_NGROK_3200` "offline" page over https, not even a real connection failure. Running a second agent in default https mode on the same domain answers browser requests correctly (ngrok can actually terminate TLS) while leaving the Pi's http-only tunnel untouched.

**Trade-off:** Both `ngrok` processes are unmanaged background processes (`nohup ... &`), not a service — they don't survive a host reboot and need to be restarted manually (or wired into `systemd`/`docker compose` if that becomes a recurring pain).

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

### Incomplete relative-path fix (fixed)

**What happened:** When `config.py` was updated to use `__file__`-anchored paths, the routers (`train.py`, `export.py`) and `trainer.py` were not updated. They still used `Path("dataset")`, `Path("models/candidate.pt")`, etc., relative to CWD. Additionally, both `annotate.py` and `export.py` opened `classes.json` as a bare relative path at module import time.

**Fix:** Added `DATASET_DIR`, `CLASSES_FILE`, `CANDIDATE_WEIGHTS`, `ARCHIVE_DIR`, and `TRAINING_RUNS_DIR` to `config.py`. All routers and `trainer.py` now import these constants. `open("classes.json")` replaced with `open(CLASSES_FILE)` in both routers.

### Export hardcoded `.jpg` extension (fixed)

**What happened:** `export.py` looked up `UPLOAD_DIR / f"{stem}.jpg"` to find the source image for each annotated entry. Images uploaded as `.png` or `.jpeg` would match annotations in `annotated/` but their source file would not be found, silently dropping them from the export.

**Fix:** Now searches across `IMAGE_EXTS` using `next(... for ext in IMAGE_EXTS if path.exists(), None)`.

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

### `uploads/` grows unbounded and can fill the disk

Images accumulate at roughly 440 MB/day (~510 images/day observed over ~70 days) and are never deleted automatically (see "No deduplication on upload" above — same root cause: nothing ever prunes `uploads/`). On a 29GB disk this reached 27GB used / 1.1GB free, which is not enough headroom for `docker compose up -d --build` to complete (`pip install torch` needs roughly 2-3GB of scratch space to download and unpack; it fails with `OSError: [Errno 28] No space left on device` partway through if the disk is much above ~90% full). When this happened, the fix was deleting the most recent 15 days of raw uploads (chosen because they were the ones under discussion at the time, not because "newest" is inherently safer to delete — at the time, `annotated/` and `processed/` were both completely empty, so no uploaded image anywhere in the archive had been annotated or run through YOLO yet; age wasn't a signal of anything).

**Before a rebuild, check `df -h /` first.** If free space is under ~3GB, either prune `uploads/` (there's no built-in retention/archival tooling yet — see below) or `docker builder prune -f` / `docker image prune -f` to reclaim build cache and unreferenced images (safe, doesn't touch running containers or data).

**Future work:** there's no automatic retention policy for `uploads/`. Given the annotation pipeline hasn't consumed any of it yet (as of this writing), consider either a retention window (e.g. delete raw uploads older than N days once they're annotated) or moving cold data off-box before it becomes a recurring disk-space emergency.
