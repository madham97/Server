from datetime import datetime, timezone

from config import PROCESSED_LOG

_processed: set[str] = set()
_failed_counts: dict[str, int] = {}
processing_enabled: bool = False


def load_processed_log() -> set[str]:
    if not PROCESSED_LOG.exists():
        return set()
    names = set()
    with open(PROCESSED_LOG) as f:
        for line in f:
            parts = line.strip().split('\t')
            if parts:
                names.add(parts[-1])
    return names


def record_processed(filename: str):
    processed_at = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
    with open(PROCESSED_LOG, 'a') as f:
        f.write(f"{processed_at}\t{filename}\n")
    _processed.add(filename)
