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

### Local (development)

```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

Expose publicly (e.g. for a Pi on cellular):
```bash
ngrok http --scheme http 8000
```

### Shared server (Docker)

Requires Docker. Model weights (`models/best.pt`) must be copied to the server manually as they are gitignored.

```bash
# On the server — first time setup
git clone -b <branch> https://github.com/madham97/Server.git ~/Server
cd ~/Server
mkdir -p models uploads processed annotated failed dataset thermal

# Copy model weights from local machine
scp /path/to/models/best.pt user@<server-ip>:~/Server/models/

# Build and start (runs in background, restarts on reboot)
docker compose up -d --build
```

Check the server is up:
```bash
curl http://localhost:8000/health
```

View logs:
```bash
docker compose logs --tail=20
```

Stop cleanly:
```bash
docker compose down
```

#### Browser access via SSH port forwarding (preferred)

For reaching the web UI (`/thermal/calibrate`, `/thermal`, `/annotate`, `/config-help`) from your own machine, **use an SSH tunnel rather than ngrok**. It needs no account, has no bandwidth cap, requires nothing installed or running on the server beyond sshd, and sidesteps the HSTS problem described below entirely.

Run this on your **local machine**, not on the server:
```bash
ssh -N -L 8000:localhost:8000 <user>@<server-ip>
```

`-N` opens the tunnel without a shell, so the command produces no output and appears to hang — that is correct. Leave it running and open:

```
http://localhost:8000/thermal/calibrate
```

Notes:
- The URL is `localhost` on your own machine. The server's LAN address (e.g. `192.168.0.2`) is not reachable from outside its network.
- Use `http://`, not `https://` — uvicorn serves plain HTTP. Because this is `localhost`, no browser HSTS upgrade applies.
- If local port 8000 is taken, `ssh` fails with `bind: Address already in use`. Change the left-hand port: `-L 8080:localhost:8000`, then browse to `localhost:8080`.
- Connecting through a jump host: keep the `-J` flag as well — `ssh -N -J <jump> -L 8000:localhost:8000 <user>@<server-ip>`.
- The tunnel dies with the SSH session, so keep that terminal open.

This does **not** replace ngrok for the Pi, which needs a genuinely public plain-HTTP endpoint reachable without an SSH client.

#### Exposing publicly via ngrok (no sudo required)

Download ngrok into the repo directory:
```bash
curl -O https://bin.equinox.io/c/bNyj1mQVY4c/ngrok-v3-stable-linux-amd64.tgz
tar -xzf ngrok-v3-stable-linux-amd64.tgz
./ngrok config add-authtoken <your-token>
```

**`--scheme http` is required for the Pi**, not optional: the GSM modem uploader can't follow redirects or negotiate TLS, so the tunnel must serve plain HTTP with no https upgrade. The tradeoff is that plain HTTP tunnels are effectively unreachable from a browser if the domain is under a gTLD with mandatory HSTS preloading (e.g. `.dev`, `.app`) — the browser force-upgrades to https before ever making a request, and no site setting or HSTS-deletion can override a TLD-level preload entry. If you need browser access (e.g. for `/thermal` or `/config-help`) on such a domain, run a **second** ngrok agent against the same domain in default mode, which is https-capable. Both agents can run simultaneously against the same static domain, split by scheme — the Pi keeps using `http://`, browsers use `https://`.

Run both as durable `systemd --user` services (survives reboots, crashes, and CLI/session disconnects — see `deploy/systemd/README.md` for why this matters over a plain `nohup ... &`):
```bash
mkdir -p ~/.config/systemd/user
cp deploy/systemd/ngrok-*.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now ngrok-http.service ngrok-https.service
loginctl enable-linger "$USER"   # required, or both die on full logout
```

Get the assigned public URL:
```bash
curl http://localhost:4040/api/tunnels
```

> **Note:** If this account has a claimed static domain, the same public URL persists across restarts. Otherwise a free-tier random URL changes every time ngrok restarts.

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
| `image` | file | JPEG or PNG (WebP accepted and converted). Thermal-fused RGBA WebP is split into JPEG + thermal sidecar (see below). |
| `device_id` | text | Identifier of the sending device |
| `mode` | text | Recording mode (`image_motion`, `image_interval`) |
| `motion_score` | text | Motion ratio that triggered capture |
| `timestamp` | text | ISO 8601 capture time |
| `format` | text | Format hint sent by the Pi (currently informational only) |
| `thermal_min_c` | text | Minimum temperature (°C) in the thermal frame, if present |
| `thermal_max_c` | text | Maximum temperature (°C) in the thermal frame, if present |
| `thermal_avg_c` | text | Average temperature (°C) in the thermal frame, if present |

Thermal-fused frames (RGBA WebP — visible image in RGB, normalized thermal map in alpha) are
split on receipt: the RGB is saved as a normal JPEG to `uploads/`, and the alpha channel is
saved as `thermal/<stem>_thermal.png` with a `thermal/<stem>_thermal.json` sidecar carrying
`thermal_min_c`/`max_c`/`avg_c` (needed to reconstruct real degrees from the 0-255 alpha).

### Utilities

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/thermal` | Browse RGB/thermal capture pairs side by side, newest first — date filters, paging, click-to-open overlay (`?start=&end=&page=&per_page=`) |
| `GET` | `/thermal/image/{name}` | Serve a raw thermal PNG from `thermal/` |
| `GET` | `/config-help` | Interactive builder for `SET key=value` SMS config commands to the Pi |

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
thermal/                — Thermal PNG + JSON sidecar split from RGBA uploads (gitignored)
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

### Upload authentication

`POST /upload` writes files to disk. It is guarded by a shared secret read from the
`UPLOAD_TOKEN` environment variable — **not** from `config.py`, so the secret never lands in
the repo.

```bash
# Generate one and put it in a .env file next to docker-compose.yml
echo "UPLOAD_TOKEN=$(openssl rand -hex 24)" >> .env
docker compose up -d
```

`.env` is gitignored; `.env.example` documents the variable and is committed. Running outside
Docker, export it instead: `UPLOAD_TOKEN=... uvicorn app:app --host 0.0.0.0 --port 8000`.

Give the Pi the same value as `upload_token` in its `client.json`. The client sends it as a
`token` multipart field; an `X-Upload-Token` header is also accepted for anything that can set
headers (`curl`, tests). A request without it gets `401`.

> **If `UPLOAD_TOKEN` is empty the endpoint is unauthenticated** and logs a warning at startup.
> That is tolerable only while the server has no route from the internet. Set it before
> forwarding a public port or handing out a tunnel URL.

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| Images not processed | Call `POST /processing?enabled=true` to enable background inference |
| Training fails immediately | Windows DataLoader issue — `workers=0` is set by default |
| YOLO saves to `runs/detect/...` | Ensure trainer uses absolute project path (already fixed) |
| Port conflict | `uvicorn app:app --port 8001` |
| Web UI unreachable remotely | Use an SSH tunnel: `ssh -N -L 8000:localhost:8000 <user>@<server-ip>`, then open `http://localhost:8000/...` |
| ngrok bandwidth limit hit | Browser access does not need ngrok — use the SSH tunnel above. Only the Pi uploader still requires a public tunnel |
| Uploads returning `401` | `UPLOAD_TOKEN` on the server and `upload_token` in the Pi's `client.json` disagree, or one is unset |
| CUDA not detected | Reinstall PyTorch with CUDA: `pip install torch --index-url https://download.pytorch.org/whl/nightly/cu126` |
