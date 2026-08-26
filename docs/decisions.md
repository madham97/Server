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

### Field devices pull their own updates, on probation until an upload proves them

**Decision:** `GET /client-update/manifest` and `/client-update/bundle` publish a client bundle (newest `.tgz` in `client_updates/`, bind-mounted so publishing needs no rebuild). Devices fetch it over their existing GSM uplink, verify its digest, compile-check it, snapshot what it replaces, install it, and enter *probation*. Probation ends only when the device **completes a real upload**; if that doesn't happen within a window (default 3 h), a systemd timer restores the snapshot.

**Why it worked:** the devices have no inbound path at all. The SIM800 is driven by `AT+HTTPACTION`, so there is no IP stack on the uplink — SSH and Tailscale both need an interface that doesn't exist, and the measured throughput (~1.8 KB/s, ~14 kbps GPRS, from upload arrival gaps) makes an interactive session unusable even over the legacy pppd link, which in any case contends with the uploader for the same `/dev/serial0`. Pull was the only option. A ~52 KB bundle is ~38 s of transfer on that link — the link can carry an update, it just can't carry a shell.

The design question was never the download; it was that a self-updater on an unreachable device can destroy its own recovery path. Three choices follow from that:

- **Success is a completed upload**, not a started process or a parsed file. Losing the uplink is the only unrecoverable failure, so the uplink is the only thing worth testing. A version that boots happily but can't reach the server would pass any weaker check and still require a site visit.
- **The watchdog depends on nothing from the bundle.** Separate systemd timer, system Python, standard library only — because its entire job is to work when the code it watches does not.
- **Verify before installing, not after.** Digest mismatch and syntax errors are caught while the live tree is untouched; a truncated transfer over 2G is a routine event, not an exceptional one.

Config, the venv and the image queues are never bundled — a rollback that reverted the SIM PIN and server URL would be its own outage.

**On by default.** The devices have no other update path, so leaving it off would mean every future change is a site visit — the cost the feature exists to remove. The safety is the probation contract, not withholding the feature. Two guards make defaulting it on defensible, both of which only matter at fleet scale:

- **An empty outbox defers the rollback** (capped at 24 h). Nothing queued means nothing needed uploading, so a missing upload is not evidence against the new code. This deployment captures almost nothing between 19:00 and 06:00 UTC, so a fixed timer would revert good updates nightly.
- **A rolled-back version is not retried for 24 h.** Without it a device reinstalls what it just rejected on the next check and loops, spending a bundle of metered 2G each time. Matching is on the version string, so publishing a fix clears the block at once.

**Trade-off:** the failure mode inverts rather than disappears. The worst case becomes "the update didn't take, and you learn that from telemetry" instead of "the device went silent." That is a large improvement but not a guarantee: a bundle that compiles, uploads successfully, and *then* misbehaves will be confirmed and kept. The probation check proves the uplink survived, nothing more.

The watchdog is deliberately duplicated rather than shared: it imports nothing from `pi-client`, because importing the rescue path out of the tree an update just replaced puts it inside the blast radius. Staging compile-checks every bundle so a syntax error cannot get that far, but a module that imports cleanly and misbehaves at runtime can. Verified by destroying `updater.py` on a device in probation — the watchdog still restored it.

**Not addressed:** bundles are unsigned. The digest protects against corruption in transit, not against a compromised server — anyone who can write to `client_updates/` can run code on every device. That is acceptable while the server is the same trust domain as the devices, and should be revisited before this is used across an untrusted network.

---

### Thermal frames travel at sensor resolution instead of stretched into an alpha channel

**Decision:** The Pi sends the thermal frame as its own `thermal` part of the same upload POST, at the sensor's native 80×62, and the server stores it that way. `POST /upload` accepts either that or the previous RGBA-fused WebP; the recorder picks by `thermal_native_transport` (default on). `align_thermal` maps whatever resolution it finds onto the calibration reference via `homography @ scale_to_reference(...)`, so both formats align with the same saved profile.

**Why it worked:** the fusion existed to make the pair unsplittable — one file, no timestamp matching, no chance of mismatched RGB and thermal. But WebP alpha must match the image dimensions, so fusing *forced* an upsample from 80×62 (4,960 samples) to 1920×1080 (2,073,600). Measured cost: **40.8 KB of every ~118 KB upload, 34%, to carry 1.8 KB of actual measurement** — over a GSM link with a 293 KB hard cap. Sending it as a second part of the same multipart body keeps the atomicity (still one POST, still unsplittable) at 1/23rd the bytes.

Three findings shaped the design:

- **PNG beats raw.** A raw 80×62 uint8 array is 4,960 B; the same frame as PNG is 1,729 B, because thermal fields are spatially smooth and PNG's row filtering predicts them well. "Just send the numbers" is the *larger* option, and a bare zlib stream (2,180 B) still loses to PNG despite skipping the container.
- **No recalibration needed.** The native→stretched map is a pure scaling, so the existing homography carries over as `H @ S`. Verified against the old path: a warm blob lands within **0.21 px** of where the three-resample pipeline put it, and the composed transform's diagonal ratio drops from 1.342 to 1.028 — i.e. the old calibration was mostly absorbing an anisotropy the pipeline itself introduced (24.0× horizontal vs 17.4× vertical). A near-similarity is what two rigidly mounted cameras should produce, which also makes a future bad fit diagnosable.
- **8-bit is sufficient.** PNG-16 at centi-°C costs 3,053 B for 13.7× finer quantization, but only 0.1% of 9,926 archive frames clip the 10–45 °C window at all, and a rodent against background spans ~109 of the 256 levels. Not worth 1.3 KB per capture.

**Trade-off:** two capture formats now exist. The 9,931 existing `upsampled_rgb` frames keep working unchanged — `align_thermal` handles both by the same rule — but any consumer reasoning about resolution has to read `thermal_geometry` rather than assume. `is_thermal_frame_corrupted` also had to be re-based onto the sensor grid, since its threshold was tuned on upsampled frames and read 60% of native frames as corrupt; the new scorer agrees with the old on 99.7% of 1,200 archive frames. Output is now always at the calibration reference size, so frames the uploader had shrunk to fit the modem buffer (~24% of the archive, at 1440×810) produce 1920×1080 aligned output instead of 1440×810 — uniform, but a change for anything that assumed aligned output matched its input size.

**Not fixed by this:** the single-resample path is *not* sharper — measured +0.6% edge energy, within noise. The final warp is a ~15.9× upscale either way, so output detail is bounded by the sensor, not by how many times the data was resampled. The gain here is bandwidth and correctness, not image quality.

---

### Thermal frames record the normalization window they were encoded with

**Decision:** The capture sidecar now carries `thermal_norm_min_c`/`thermal_norm_max_c` — the fixed °C window the 0-255 pixel values were actually encoded against — plus `thermal_norm_source` (`reported` when the device sent it, `assumed_legacy_default` when the server filled in 10–45 °C for an older client). The Pi writes the window in `_write_sidecar`, the uploader passes it through its form-field whitelist, and `POST /upload` persists it.

**Why it worked:** `frame_to_gray8` documents the decode as `temp_c = norm_min + (pixel/255) * (norm_max - norm_min)` "using whatever norm_min_c/norm_max_c this frame was actually encoded with" — but that value was never recorded anywhere. Both bounds are `config.get()` values on the Pi, so retuning them (which is under active discussion, to improve apparent contrast) would silently invalidate the decode for every subsequent frame, with nothing in the data to distinguish old from new. The alternative — pinning the constants and forbidding changes — trades a ~30-byte-per-capture cost for a permanent constraint on a value that legitimately may need retuning per deployment.

The observed `thermal_min_c`/`max_c`/`avg_c` already in the sidecar are *not* a substitute: they are the frame's own range, and decoding against them rescales every frame differently — precisely the failure the fixed-window normalization was introduced to fix.

**Trade-off:** The 9,930 pre-existing captures have no window recorded and are not backfilled, so they rely on the `assumed_legacy_default` rule. That assumption is believed sound (nothing had overridden the defaults) but is not verifiable from the data — which is the whole reason for recording it explicitly from now on.

---

### Date filters, paging, and a click-to-open overlay on `/thermal`

**Decision:** `/thermal` now takes `start`/`end` (inclusive UTC capture dates), `hour_start`/`hour_end` (an inclusive UTC hour-of-day window that wraps through midnight when start > end, since a nocturnal window always straddles 00:00), `page`, and `per_page` (default 24, max 200), and clicking any thumbnail opens a lightbox showing the aligned thermal composited over the RGB with an opacity slider — the same overlay the calibration page renders for the single capture being calibrated, reused for browsing. `limit` is kept as an alias for `per_page` so pre-existing links don't break.

**Why it worked:** Two cheaper-looking options were rejected. (1) Making `/thermal` a JSON API plus a static SPA like `static/thermal_calibrate.html`: the page is a *browsing* surface, and server-rendered query-param pages get shareable URLs, working back/forward, and bookmarkable filtered views for free — an SPA would have to rebuild all of that in JS. (2) Reading each sidecar's JSON `timestamp` to date-filter: with ~10k sidecars in `thermal/` that's 10k file opens on every page load, for a value already encoded in the filename (`..._YYYYMMDDTHHMMSSZ_thermal.json`). Filtering on the filename means only the ≤200 sidecars actually rendered are ever opened, so page load stays flat as the capture count grows.

**Trade-off (hour filter):** The window is in UTC, matching the dates and the stored timestamps, so a Heidelberg user reading local time has to offset by 1–2h depending on DST. Storing local time instead would make the archive ambiguous across the DST boundary, which is worse for a filter whose whole purpose is selecting night hours.

**Trade-off:** Lightbox navigation (`←`/`→`) is bounded to the current page rather than walking the whole filtered set, because the client only knows about the cards the server rendered — crossing a page boundary means going back and clicking again. A sidecar whose filename carries no parseable timestamp can't be placed on the timeline, so it is dropped whenever either date bound is set (it remains visible unfiltered); that only affects captures not named by the current uploader.

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

**Trade-off:** Originally both `ngrok` processes were unmanaged background processes (`nohup ... &`), not a service — see the systemd entry below for why that turned out not to be durable enough and what replaced it.

---

### ngrok tunnels moved to `systemd --user` services with lingering enabled

**Decision:** Both ngrok tunnels run as `systemd --user` services (`deploy/systemd/ngrok-http.service`, `deploy/systemd/ngrok-https.service`), `Restart=always`, with `loginctl enable-linger` set for the account so the user's systemd instance — and these services — keep running with no active login session.

**Why it worked:** A `nohup ... & disown` ngrok process survived three weeks in one instance, which made it look durable enough. It isn't, in general: a later pair of tunnels, started via the CLI harness's own background-task tracking (not manual `nohup`), were both killed outright — silently, no crash, no error — when a session boundary was crossed overnight. Since ngrok is what makes the Pi's uploads reachable at all, that gap meant the device was silently failing to upload with nothing to signal it. The actual failure mode wasn't about *how* the process was backgrounded (`nohup` vs. the harness) so much as that neither approach detaches a process from every possible session/lineage boundary, and neither one restarts itself after any kind of interruption. `systemd --user` plus lingering does both: it's independent of any particular shell/session, and `Restart=always` means a crash (or the same kind of session-boundary kill, if that mechanism recurs elsewhere) self-heals in seconds instead of requiring someone to notice.

**Trade-off:** `loginctl enable-linger` is host-account state, not something version control can capture — it has to be run once per machine/account during setup (see `deploy/systemd/README.md`). The unit files themselves are portable (`%h` specifier for the home directory) and live in the repo.

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

### Auto-calibration's homography → affine, plus a minimum-inlier-ratio guard

**What happened:** `auto_calibrate` (background-subtraction correspondences, no manual clicking) was fitting a full projective homography (`cv2.findHomography`, RANSAC) through data that was ~92% outliers (9 of 115 correspondences were real inliers). A projective fit through that much noise produced a near-singular perspective row, and `cv2.warpPerspective` divided by near-zero for some pixels — every aligned thermal PNG came out as a radial "starburst," not a merely-misaligned image. The bad fit was saved without complaint; nothing checked inlier ratio before writing `calibration/homography.json`.

**Decision:** `auto_calibrate` now fits an affine transform (`cv2.estimateAffine2D` + RANSAC) instead of a homography, matching what `calibrate_from_boxes`/`_affine_transform` already assumed for this rig (fixed cameras, no real perspective distortion — see the entry below). Affine has no perspective row, so this failure mode is structurally impossible now regardless of input quality. Also added `_MIN_INLIER_RATIO = 0.3`: if fewer than 30% of correspondences agree with the fit, `auto_calibrate` raises instead of saving.

**Why it worked:** The starburst was pure arithmetic blowup, not "worse" misalignment — confirmed by rendering the actual aligned output and by checking the saved homography's bottom row (`[-0.0015, 0.0003, 1]`, i.e. a near-zero denominator lurking in the image). Removing the perspective term removes the mechanism, not just the symptom.

**Trade-off:** The inlier-ratio guard means `auto_calibrate` will now *fail loudly* on marginal data instead of producing something usable-if-imprecise. On this deployment's actual capture data it currently fails outright (~7% agreement even after the blob-detector improvements below) — see "Known Limitations" for why, and use manual box calibration (`/thermal/calibrate`) in the meantime.

---

### Auto-calibration's blob detector: z-score threshold + circularity filter, not Otsu

**What happened:** Investigating the low inlier ratio above, rendering the actual foreground masks `_blob_centroid` was thresholding showed the real problem: on the thermal side, Otsu-on-absolute-diff was scoring 300–850 separate "blobs" per frame — pure per-pixel sensor noise, not a coherent animal shape. On the RGB side, Otsu was picking up static structure (cage bars, lighting/shadow gradients) that differs from the median background but isn't the subject. Otsu splits whatever's in the frame into two halves regardless of whether real foreground is present, so on this noisy data the "largest blob in the plausible size range" was landing on noise most of the time.

**Decision:** Replaced the Otsu-on-raw-diff mask in `_blob_centroid` with a per-pixel z-score threshold (`|frame − median| / (background_std + 3)`, blurred first to suppress sensor noise), plus a circularity filter (`4π·area/perimeter² ≥ 0.25`) to reject elongated non-blob regions (bars, diagonal shadow edges) that a compact animal doesn't produce. `_median_background` became `_background_stats`, returning both the median and the per-pixel std from one pass over the sampled stack. Also corrected `min_area_frac`/`max_area_frac` (0.0015–0.2 → 0.0001–0.05): the old floor was tuned around the old noisy Otsu masks' much-larger spurious regions and was silently rejecting the real (much smaller) animal blob.

**Why it worked:** Verified by inspecting rendered masks before/after — the z-score approach collapsed thermal blob counts from hundreds down to single digits per frame. Auto-calibration's match rate roughly doubled on this dataset (from ~8% to ~19/30 frames finding a plausible blob on both sides).

**Trade-off / known limitation:** Even with this improvement, RGB-side detection still only geometrically agrees with the thermal side on the same real animal ~7% of the time on this deployment's actual footage — several spot-checked RGB detections clustered on the same coordinates across unrelated frames, suggesting the detector is latching onto some static feature (a shadow or edge) rather than tracking the moving subject. There's no trained RGB animal detector available to substitute (`models/` is empty, no `best.pt`) — plain grayscale background-subtraction on the RGB channel may simply not be reliable enough for a small, low-contrast rodent. **Use manual box calibration for now**; if auto-calibration needs to work reliably, the next step would be a trained RGB detector (once training data exists) feeding into `calibrate_from_boxes`, not further tuning of background-subtraction thresholds.

**Update — root cause found on the thermal side, fixed upstream:** the thermal-side "noise" (300–850 spurious blobs/frame, above) wasn't classical spatial sensor noise — `pi-client/thermal/thermal_common.py` in the `Rodent-client` repo normalized each raw 80×62 frame with `cv.normalize(..., NORM_MINMAX)`, stretching to *that frame's own* min/max. So the same real temperature maps to a different 0-255 pixel value depending on whatever else is in view that frame (a shadow, warm bedding), which defeats background subtraction's core assumption of a stable per-pixel baseline — no amount of spatial blur fixes a shifting baseline. Fixed in `Rodent-client` by normalizing against a fixed, deployment-wide °C range (`thermal_norm_min_c`/`thermal_norm_max_c`, default 10–45°C, chosen from ~200 real captures here: frame minimums 17.0–30.2°C, maximums 27.5–39.6°C) instead of each frame's own range. This doesn't by itself fix the RGB-side detection problem described above, but it removes a real confound from the thermal side, and should reduce blob counts on that side further than the z-score fix alone. Worth re-running `auto-calibrate` and re-checking the inlier ratio once devices are updated with the new client.

---

### Calibration preview: `mix-blend-mode: normal`, not `screen`

**What happened:** The calibration UI's aligned-thermal preview used `mix-blend-mode: screen` with an opacity slider. Screen blending always lightens using *both* layers' luminance (`result = 1 − (1−bg)(1−fg)`) — so even at 100% slider position, the RGB backdrop kept showing through, and there was no way to see the pure aligned-thermal image to check pixel-precise alignment.

**Decision:** Changed to `mix-blend-mode: normal`, making the slider a plain crossfade: 0% shows pure RGB, 100% shows pure aligned thermal, anywhere between lets you check edge alignment against the backdrop.

---

### Calibration preview: show "not aligned yet" instead of silently hiding

**What happened:** The preview's `<img>.onerror` handler set the entire preview block to `display: none` on a 404 (no aligned PNG for the current stem), with no message. The image picker defaults to the *newest* capture, and since captures arrive continuously via the background watcher, the newest one very often postdates the last calibration run and has no aligned file yet — so the preview looked broken on nearly every page load, not just occasionally.

**Decision:** Added an explicit `#alignedMissing` message ("Not aligned yet — this capture arrived after the last calibration ran") shown in place of the image on a 404, mirroring the "not aligned yet" placeholder `routers/thermal_view.py`'s gallery already used for the same underlying situation.

---

### New thermal captures aligned immediately on upload, not just at calibration time

**What happened:** Alignment only ever ran inside the three `/thermal/calibrate/*` submit endpoints (`align_all(force=True)`). A capture arriving between calibration runs had no aligned counterpart until the next manual recalibration — even though a perfectly good, already-saved homography existed and could have been applied to it immediately.

**Decision:** `routers/upload.py`'s `/upload` handler now schedules a `BackgroundTasks` job (`_align_new_capture`) right after writing a new thermal PNG + sidecar: it looks up whichever calibration covers that capture's own timestamp and warps the new frame with it, same as `align_thermal` does in `align_all`. Runs in the background so it doesn't add latency to the Pi's upload request; does nothing if no calibration covers that timestamp.

**Why it worked:** Verified end-to-end — synthesized an RGBA WebP from an existing thermal/RGB pair, POSTed it to `/upload`, and confirmed `<stem>_thermal_aligned.png` appeared within the same request cycle with no calibration re-run.

**Update — superseded by profile-scoped realignment below:** the original trade-off here was that any recalibration re-warped *every* existing thermal frame with the new transform, whether or not that was correct for old footage. That's no longer true — see "Calibration profiles, scoped by effective time window" below.

---

### Calibration profiles, scoped by effective time window

**What happened:** Calibration was a single `calibration/homography.json` — one transform, used for every capture regardless of when it was taken. That's fine as long as the camera rig never moves, but it doesn't hold once it does: recalibrating after a physical move would fit a new transform for the new geometry and then (via `align_all(force=True)`) blindly re-warp *every* existing thermal frame with it — including frames captured *before* the move, which were correctly aligned under the old geometry and would be silently corrupted by the new one.

**Decision:** Calibration is now a list of **profiles** in `calibration/profiles.json`, each carrying a `[effective_from, effective_until)` window (an ISO timestamp, and either another ISO timestamp or `null` for open-ended) alongside its transform and fit diagnostics. `find_profile_for_timestamp` picks whichever profile's window contains a given capture's own sidecar timestamp — not upload time, not "now," and not necessarily the newest profile. `align_all` and the `/upload` background task both resolve each capture's applicable profile independently through this lookup, so recalibrating after a camera move (create a new profile with `effective_from` = the move) only ever re-aligns the frames actually taken after it; earlier frames keep using the profile that was actually in effect when they were captured. A new/edited profile's window is validated against every other profile's window before saving — two profiles claiming the same moment would be ambiguous, so overlap is rejected outright (400) rather than silently picked between.

On first read, if `profiles.json` doesn't exist yet but the old `homography.json` does, it's migrated into one profile covering all time (epoch to open-ended) — an existing deployment's calibration keeps working unchanged until a real recalibration creates a second, properly time-scoped profile.

Per explicit request, profiles stay editable indefinitely (adding more box pairs to refine an old profile updates it in place, doesn't create a duplicate) and windows are explicit start/end fields rather than always-open-ended-from-now — both chosen over the simpler alternatives for more control, at the cost of needing the overlap validation above to keep the model unambiguous.

**Why it worked:** Verified end-to-end against the real deployment data: created a second profile with a real `effective_from` cutoff, confirmed `align_all(force=True, profile_id=...)` only touched captures at or after that cutoff (checked via file mtimes — a pre-cutoff capture's aligned PNG was untouched) while every post-cutoff capture got the new transform. Also verified the overlap guard rejects a conflicting window with a clear message naming the existing profile it conflicts with, and that editing a profile in place preserves its id while updating its fit and window.

**Trade-off:** A capture whose timestamp falls in a genuine gap between two profiles' windows (or before any profile exists) is left unaligned rather than falling back to some default — treated as "don't align this one" rather than an error, consistent with how a missing calibration was always handled, but it does mean a badly-chosen window boundary can silently leave a batch of captures unaligned. The calibration UI surfaces each profile's window in a list (with edit/delete) specifically so gaps are visible rather than discovered later.

---

## Known Limitations and Future Work

### Auto-calibration is unreliable on this deployment's footage

`POST /thermal/calibrate/auto` currently rejects its own fit outright (under 30% RANSAC inlier agreement) on this deployment's real captures, even after the z-score/circularity blob-detector improvements — see the entries above. The bottleneck is RGB-side detection: plain grayscale background-subtraction doesn't reliably localize a small, low-contrast rodent, and there's no trained RGB detector yet to substitute (`models/` is empty). Use manual box calibration (`/thermal/calibrate`, "draw boxes" flow) until either the footage/lighting changes enough for background-subtraction to work, or a trained RGB detector exists to feed `calibrate_from_boxes` directly.

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
