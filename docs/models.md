# Model Management

## Model Files

| Path | Purpose |
|---|---|
| `models/best.pt` | Live weights used by the background watcher and `/infer/test`. |
| `models/candidate.pt` | Output of the most recent completed training run. Not used for inference until promoted. |
| `models/runs/<timestamp>/` | Full training output: weights, confusion matrix, PR curve, `results.csv`. |
| `models/archive/<timestamp>_best.pt` | Previous live models, saved before each promotion. |

All of these are gitignored. Back them up externally if they matter.

---

## Base Models

The server fine-tunes from a YOLO checkpoint. The default is `yolov8n.pt` (nano — fast, lower accuracy). Available options from ultralytics:

| Model | Params | Notes |
|---|---|---|
| `yolov8n.pt` | 3.2M | Default. Good for prototyping and resource-constrained hardware. |
| `yolov8s.pt` | 11.2M | Better accuracy, still fast on GPU. |
| `yolov8m.pt` | 25.9M | Use when you have a substantial dataset (500+ images per class). |
| `yolov8l.pt` | 43.7M | High accuracy, slow training. |

When calling `POST /train/start`, pass `base_model=yolov8s.pt` to use a larger base. The model is downloaded automatically by ultralytics on first use.

---

## Training Workflow

### Prerequisites

1. Annotate images via the browser UI at `/annotate`.
2. Check progress: `GET /annotate/stats` — aim for a reasonable class balance.
3. Export: `POST /dataset/export?val_split=0.2`
4. Verify: `GET /dataset/stats` should show non-zero train and val counts.

### Starting a run

```
POST /train/start?epochs=100&imgsz=640&base_model=yolov8n.pt
```

Monitor progress:
```
GET /train/status
```

The `metrics` field updates after every epoch. Key metrics:

| Metric | What it means |
|---|---|
| `metrics/mAP50(B)` | Mean average precision at IoU=0.5. Primary quality signal. |
| `metrics/mAP50-95(B)` | mAP averaged over IoU 0.5–0.95. Stricter. |
| `train/box_loss` | Bounding box regression loss. |
| `train/cls_loss` | Classification loss. |

### Promotion

Once training is complete (`state: "complete"`):

```
POST /train/promote
```

This will fail if `mAP50 < 0.3` (the `MIN_MAP` threshold in `config.py`). If the model is good enough but below threshold, raise `MIN_MAP` in config — do not remove the check entirely.

After promotion, the old `best.pt` is archived and the new one is immediately live for the background watcher.

---

## Dataset Guidelines

### Minimum viable dataset

YOLOv8n can start learning with as few as 50–100 annotated images per class, but results will be noisy. Aim for:

- 200+ images per class for a reliable model.
- Varied lighting, angles, and backgrounds.
- Roughly balanced class counts (within 2:1 ratio).

### Validation split

The default `val_split=0.2` reserves 20% for validation. Do not go below 0.1 or above 0.4. With very small datasets (< 50 total images), use 0.15 to preserve more training examples.

### Negative examples

Images with no detections (empty annotation) are valid training examples. They help reduce false positives. Include a proportion of blank-field images.

---

## Inference Configuration

Set in `config.py`. These apply to both the background watcher and `/infer/test`:

| Setting | Default | Effect |
|---|---|---|
| `MODEL_CONF` | `0.25` | Confidence threshold. Lower = more detections, more false positives. |
| `MODEL_IOU` | `0.45` | NMS IoU threshold. Lower = fewer overlapping boxes. |
| `MODEL_IMGSZ` | `640` | Inference image size. Match your training `imgsz`. |

Tune `MODEL_CONF` first if you're seeing too many or too few detections in the field.

---

## Rollback

To roll back to a previous model:

```powershell
Copy-Item models\archive\<timestamp>_best.pt models\best.pt
```

The server picks up the new weights on the next inference call — no restart required, because `process_image` loads the model fresh each time it is called from the background watcher (a new `YOLO(weights)` instance per call). For high-throughput use, consider caching the model in `state.py`.

---

## Known Issues

- **workers=0 is mandatory on Windows.** PyTorch's multiprocessing DataLoader freezes when spawned from a thread inside a running asyncio app. This is set in `trainer.py` and must not be removed.
- **Training saves to `models/runs/` not `runs/`.** The absolute `project_dir` path in `trainer._run()` prevents YOLO from falling back to a `runs/` directory in the CWD. If you ever see a `runs/` directory appear at the project root, the path passed to `YOLO.train()` is wrong.
- **Candidate is overwritten by each new run.** `models/candidate.pt` is replaced when any training completes. If you want to preserve a candidate before starting a new run, archive it manually first.
