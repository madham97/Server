from pathlib import Path

# Directory configuration
_BASE = Path(__file__).parent

UPLOAD_DIR = _BASE / "uploads"
PROCESSED_DIR = _BASE / "processed"
ANNOTATED_DIR = _BASE / "annotated"
FAILED_DIR = _BASE / "failed"

# Processing configuration
POLL_INTERVAL = 5
MAX_PROCESS_ATTEMPTS = 3
UPLOAD_LOG = _BASE / "upload_log.txt"
PROCESSED_LOG = _BASE / "processed_log.txt"
ANNOTATION_LOG = _BASE / "annotation_log.txt"

# Model configuration
MODEL_WEIGHTS = str(_BASE / "models/best.pt")
MODEL_CONF = 0.25
MODEL_IOU = 0.45
MODEL_IMGSZ = 640
BASE_MODEL = "yolov8n.pt"
TRAIN_EPOCHS = 100
TRAIN_IMGSZ = 640
MIN_MAP = 0.3
