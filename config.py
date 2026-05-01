from pathlib import Path

# Configuration for inference processing
ENABLE_PROCESSING = False

# Directory configuration
UPLOAD_DIR = Path("uploads")
PROCESSED_DIR = Path("processed")
FAILED_DIR = Path("failed")

# Processing configuration
POLL_INTERVAL = 5
MAX_PROCESS_ATTEMPTS = 3
UPLOAD_LOG = Path("upload_log.txt")
PROCESSED_LOG = Path("processed_log.txt")

# Model configuration
MODEL_WEIGHTS = "models/best.pt"
MODEL_CONF = 0.25
MODEL_IOU = 0.45
