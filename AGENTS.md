# AGENTS.md

Guidance for AI coding agents working in this repository. Human-facing docs live in
[README.md](README.md) and [docs/](docs/); this file is the orientation layer on top of them.

## What this is

A FastAPI server that ingests captures from Raspberry Pi field devices (wildlife/rodent
monitoring), runs YOLOv8 detection on them in the background, and provides browser UIs for
annotation, thermal/RGB calibration, and capture review — plus a full annotate → export →
train → promote loop for building custom models.

Two classes: `rat` and `human` (`classes.json`).

This is one half of a two-repo system. The other half is **`Rodent-client`** (the Pi-side
recorder/uploader) — checked out at `/home/haamer/Rodent-client` on sentinel-compute, with its
own `AGENTS.md`. Read that file before changing anything on the wire; it documents the client
side of the same contract. The client talks to this server over a SIM800 GSM modem at
~1.8 KB/s with no inbound route. That constraint explains most of the odd-looking design decisions here — base64
bundles, multipart tokens, 80×62 thermal frames. When a change touches the wire format of
`/upload` or `/client-update/*`, the client repo has to change with it.

**Read [docs/decisions.md](docs/decisions.md) before changing core pipeline logic.** It records
what was tried, what broke, and why the current shape was chosen. Several entries document
failure modes that look like reasonable "improvements" from the code alone.

## Layout

| Path | Role |
|---|---|
| [app.py](app.py) | FastAPI app, lifespan startup, `watch_uploads()` background watcher |
| [config.py](config.py) | Every path, threshold, and model default. Nothing else holds constants |
| [state.py](state.py) | Shared mutable watcher state (`_processed`, `_failed_counts`, `processing_enabled`) |
| [inference.py](inference.py) | YOLO predict wrapper; also a CLI |
| [trainer.py](trainer.py) | Training daemon thread + status under a `threading.Lock` |
| [thermal_align.py](thermal_align.py) | Calibration profiles, homography/affine fitting, warping, corruption detection. Also a CLI (`calibrate`, `align`, `auto-calibrate`) |
| [thermal_color.py](thermal_color.py) | Decodes stored 0-255 thermal bytes back to °C and re-renders them. Owns the `thermal_norm_*` invariant below — read it before touching any thermal display path |
| [repair_squashed_thermal.py](repair_squashed_thermal.py) | One-off archive repair for frames stored 62×80 by the Aug 2026 `fpa_shape` axis bug. Its docstring is the reference for that failure |
| `Dockerfile`, `docker-compose.yml` | Deployment. Code is baked into the image; only data dirs and logs are mounted |
| [routers/](routers/) | One module per endpoint group; all mounted in `app.py` |
| [static/](static/) | `annotate.html`, `thermal_calibrate.html` — single-page UIs, no build step |
| [docs/](docs/) | `architecture.md`, `api.md`, `decisions.md`, `models.md` + published HTML decks/viewers |
| [deploy/systemd/](deploy/systemd/) | ngrok user units for the deployment host |

Runtime data directories (`uploads/`, `thermal/`, `processed/`, `annotated/`, `failed/`,
`dataset/`, `models/`, `runs/`) are all gitignored and bind-mounted in Docker. `calibration/`
is gitignored except for `profiles.json`, which is committed.

## Running

**Check `hostname` first — this repo exists in two places.** There is a Windows development
checkout, and there is `/home/haamer/Server` on the deployment host itself (`sentinel-compute`),
which is very likely where you are reading this. On sentinel-compute the server is already
running as the `server-server-1` container holding port 8000: do **not** run the uvicorn line
below there — it will fail to bind, or worse, race the live process over the same data
directories. Use `docker compose` there, and read logs with `docker compose logs -f`.

Development (Windows, from repo root):

```bash
uvicorn app:app --host 0.0.0.0 --port 8000
curl http://localhost:8000/health
```

Deployment (`haamer@129.206.4.33`, "sentinel-compute"): Docker Compose. **Application code is
baked into the image** (`COPY *.py`, `routers/`, `static/`) — a code change requires
`docker compose up -d --build`, not a restart. Only the data directories and log files are
mounted. Weights (`models/best.pt`) are copied to the host manually; they are gitignored.

**Check `df -h /` before any rebuild.** The disk has filled before: `uploads/` grows ~440 MB/day
and nothing prunes it, and `pip install torch` needs 2–3 GB of scratch. Under ~3 GB free the
build fails partway with `ENOSPC`.

Browser access to the UIs goes through an SSH tunnel (`ssh -N -L 8000:localhost:8000 …`), not
ngrok. ngrok exists for the Pi, which needs a genuinely public plain-HTTP endpoint.

## Testing

**There is no test suite** — no pytest, no CI. Verification here is empirical and end-to-end,
and the `decisions.md` entries show the expected standard: POST a synthesized capture to
`/upload` and check the artifacts that appear; run a transform over real archive frames and
report the measured numbers; check file mtimes to prove which captures a realignment touched.

When you change pipeline behaviour, verify against real data in `uploads/`/`thermal/` and say
what you measured. Do not report a change as working because it imports cleanly.

## Invariants — do not break these

- **`workers=0` in `trainer.py`.** PyTorch's spawn-based DataLoader deadlocks when started from
  a thread inside a running asyncio app on Windows. Slower, but the alternative is a freeze.
- **Paths are anchored to `Path(__file__).parent` in `config.py`**, never relative to CWD, and
  routers import them from there rather than rebuilding their own. This fixed a real bug where
  uploads landed outside the project.
- **Always pass `project=` and `name=` to YOLO `predict`/`train`.** Without them ultralytics
  scatters output into `runs/detect/` relative to CWD.
- **`thermal/` is a sibling of `uploads/`, not a child.** That is what keeps thermal PNGs
  structurally out of the watcher's scope — it is not an incidental consequence of the
  extension filter, and nesting them would silently feed thermal frames into detection.
- **`Path(filename).name` on any client-supplied filename in `upload.py`.** `/upload` writes to
  disk; without stripping directory components a crafted name escapes `UPLOAD_DIR`.
- **Decode thermal pixels with `thermal_norm_min_c`/`thermal_norm_max_c`, never with
  `thermal_min_c`/`thermal_max_c`.** The `norm_` pair is the fixed window the 0-255 values were
  encoded against; the other pair is observed-range telemetry. Using the latter rescales every
  frame differently and quietly corrupts the temperatures.
- **Alignment resolves a profile by the capture's *own* timestamp**, via
  `find_profile_for_timestamp` — not "now", not the newest profile. This is what lets the camera
  move mid-deployment without corrupting frames taken before the move. Profile windows are
  validated non-overlapping on save; keep it that way.
- **Keep the `mAP50 >= MIN_MAP` promotion gate.** If a model is good enough but below the bar,
  raise `MIN_MAP` in `config.py` — do not delete the check.
- **`UPLOAD_TOKEN` comes from the environment**, never `config.py`, so the secret stays out of
  the repo. Compare it with `secrets.compare_digest`. It is accepted as a `token` multipart
  field *and* an `X-Upload-Token` header because SIM800 firmware handles custom headers
  inconsistently — do not drop the form-field path.

## Conventions

- Comments in this codebase explain **why**, at length, and often cite measured numbers. Match
  that register; a comment restating the code is worse than none. Several near-identical-looking
  constants (`_CORRUPTED_ROW_STREAK_THRESHOLD`, `_MIN_INLIER_RATIO`, `_BLOB_Z_THRESH`) were tuned
  against archive data — if you change one, say what you measured.
- Non-trivial design changes get an entry in `docs/decisions.md` under *What Worked*, *What
  Didn't Work*, or *Known Limitations*, in the existing format (**Decision / Why it worked /
  Trade-off**). Update `docs/api.md` when an endpoint's contract changes.
- Persistence is append-only tab-separated `.txt` logs (`upload_log.txt`, `processed_log.txt`,
  `annotation_log.txt`) — deliberately, not a database. Don't reintroduce one without reading the
  TrackerDB entry in `decisions.md`.
- The UIs in `static/` and the HTML embedded in `routers/thermal_view.py` and
  `routers/config_help.py` are hand-written single files with no framework or build step. Keep it
  that way; they share a dark monospace look.
- Timestamps are UTC ISO 8601 (`YYYY-MM-DDTHH:MM:SSZ`); capture stems carry `YYYYMMDDTHHMMSSZ`,
  which several code paths parse straight off the filename to avoid opening thousands of sidecars.

## Cross-repo changes

`/upload` and `/client-update/*` are a contract with the `Rodent-client` fleet, and the fleet is
the half you cannot reach to fix — over GSM the device only ever pulls. So the order matters:

1. **Change the server first, backward-compatible with the bundle the fleet is running now.**
   A field the current client does not send must not become required.
2. **Then publish the client change** with `scripts/build-update-bundle.sh` *in the client repo*
   — that is the only path that compile-checks, runs the offline test gate, and prints the
   digest. Do not hand-write a `.tgz` into `client_updates/`; it skips all three.
3. **Let probation prove it.** The device promotes an update only after a completed upload, and
   `monitoring-pipeline-update-watchdog` rolls it back otherwise.

Only then drop the compatibility shim, once every device has confirmed.

`GET /client-update/manifest` serves the bundle in `client_updates/` with the newest **mtime**
(not the newest filename — [routers/client_update.py:25](routers/client_update.py#L25)), so
removing a bundle rolls the fleet back to the one before it, and anything that rewrites mtimes
in that directory silently re-elects a bundle. The client's `AGENTS.md` carries the multipart
field list.

## Known sharp edges

- `/upload` deduplicates nothing — a repeated filename overwrites, and the processed log makes
  the second copy look already-inferred.
- `state._failed_counts` is in-memory; a restart grants three more attempts. Intentional.
- `POST /dataset/export` rebuilds `dataset/` from scratch. Never hand-edit the exported dataset.
- Auto-calibration (`POST /thermal/calibrate/auto`) currently fails its own inlier guard on this
  deployment's footage — the RGB-side blob detector can't reliably localize a small rodent. Use
  manual box calibration. Further threshold tuning is not the fix; a trained RGB detector is.
- Everything except `/upload` is unauthenticated. The server is meant to sit behind a tunnel or a
  private network.
- Client update bundles are unsigned. The digest covers transit corruption, not a compromised
  server — anyone who can write `client_updates/` runs code on every device.

## Git

Work on a branch; `main` is the default and the deployment tracks it. Do not commit weights
(`*.pt`), captures, or `.env`. Published client bundles (`client_updates/*.tgz`) are
deliberately untracked — the `Rodent-client` repo is the record of what shipped.
