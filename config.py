import os
from pathlib import Path

# Directory configuration
_BASE = Path(__file__).parent

# Shared secret required by POST /upload. Set via the UPLOAD_TOKEN environment variable.
# Empty means the endpoint stays open — acceptable only while the server is unreachable from
# the internet (e.g. behind a tunnel whose URL is not published). Set it before forwarding a
# public port: /upload writes files to disk and is otherwise unauthenticated.
UPLOAD_TOKEN = os.environ.get("UPLOAD_TOKEN", "").strip()

UPLOAD_DIR = _BASE / "uploads"
PROCESSED_DIR = _BASE / "processed"
ANNOTATED_DIR = _BASE / "annotated"
FAILED_DIR = _BASE / "failed"
THERMAL_DIR = _BASE / "thermal"
CALIBRATION_DIR = _BASE / "calibration"
CALIBRATION_FILE = CALIBRATION_DIR / "homography.json"  # legacy single-calibration file; migrated into PROFILES_FILE on first read, see thermal_align.load_profiles
CALIBRATION_PROFILES_FILE = CALIBRATION_DIR / "profiles.json"

# Processing configuration
POLL_INTERVAL = 5
MAX_PROCESS_ATTEMPTS = 3
UPLOAD_LOG = _BASE / "upload_log.txt"
PROCESSED_LOG = _BASE / "processed_log.txt"
ANNOTATION_LOG = _BASE / "annotation_log.txt"

# Dataset
# Client update bundles published to the field devices. Drop a .tgz here (newest wins) and the
# devices pick it up on their next poll — see routers/client_update.py.
CLIENT_UPDATE_DIR = _BASE / "client_updates"

DATASET_DIR = _BASE / "dataset"
CLASSES_FILE = _BASE / "classes.json"

# Model configuration
MODEL_WEIGHTS = str(_BASE / "models/best.pt")
CANDIDATE_WEIGHTS = _BASE / "models" / "candidate.pt"
ARCHIVE_DIR = _BASE / "models" / "archive"
TRAINING_RUNS_DIR = _BASE / "models" / "runs"
MODEL_CONF = 0.25
MODEL_IOU = 0.45
MODEL_IMGSZ = 640
BASE_MODEL = "yolov8n.pt"
TRAIN_EPOCHS = 100
TRAIN_IMGSZ = 640
MIN_MAP = 0.3
