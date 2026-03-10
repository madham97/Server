import os
import sys
import logging
import re
from pathlib import Path
from datetime import datetime
from ultralytics import YOLO
from tracker import TrackerDB, parse_timestamp_from_filename

# Add scripts directory to path for config import
sys.path.insert(0, str(Path(__file__).parent))

# Common video extensions we consider
VIDEO_EXTS = {'.mp4', '.mov', '.avi', '.mkv', '.flv', '.webm'}

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')


def process_video(video_path: str, weights: str = "models/best.pt", project: str = 'processed', device: str = "cuda", conf: float = 0.25, iou: float = 0.45, model: YOLO | None = None):
    """Run YOLO inference on a single video file and save results into `project`.

    Returns the YOLO results object.
    """
    if not os.path.exists(weights):
        raise FileNotFoundError(f'Weights file not found: {weights}')

    if not os.path.exists(video_path):
        raise FileNotFoundError(f'Video file not found: {video_path}')

    # Create proper folder structure for this specific video
    os.makedirs(project, exist_ok=True)
    video_name = Path(video_path).stem
    video_output_dir = os.path.join(project, video_name)

    if model is None:
        logging.info('Loading model...')
        model = YOLO(weights)

    logging.info(f'Running inference on {video_path}...')
    results = model.track(
        source=video_path,
        imgsz=640,
        device=device,
        conf=conf,
        iou=iou,
        save=True,
        project=project,
        name=video_name,
        verbose=False,
        save_txt=True,
        save_conf=True,
        # stream=True
    )
    logging.info(f'Video inference complete. Results saved to {video_output_dir}')
    return results


def main():
    """If run as a script, process either a single file or all supported files in a directory.

    This version sorts files by embedded timestamp (if present) and uses a TrackerDB to
    link tracks across sequential video chunks.
    """
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('path', nargs='?', default='uploads', help='Video file or directory to process')
    parser.add_argument('--project', default='processed')
    parser.add_argument('--db', default='objects.db', help='SQLite DB file for tracking objects')
    parser.add_argument('--max-gap', type=int, default=300, help='Max seconds allowed between chunks for linking')
    args = parser.parse_args()

    path = Path(args.path)
    if not path.exists():
        raise SystemExit(f'Path not found: {path}')

    db = TrackerDB(args.db)

    # Load model once to speed up repeated inference
    weights = "models/best.pt"
    logging.info('Loading YOLO model for batch processing...')
    model = YOLO(weights)

    try:
        if path.is_dir():
            files = [p for p in path.iterdir() if p.suffix.lower() in VIDEO_EXTS and p.is_file()]
            file_entries = []
            for p in files:
                ts = parse_timestamp_from_filename(p.name)
                # Try to extract prefix (camera or feed name) by removing timestamp portion
                m = re.match(r"(?P<prefix>.*?)(?:_\d{8}T\d{6}Z)?(?:\.|$)", p.stem)
                prefix = m.group('prefix') if m else p.stem
                file_entries.append((prefix, ts, p))

            # Sort by prefix and timestamp (None timestamps will be sorted by name)
            file_entries.sort(key=lambda x: (x[0], x[1] or datetime.fromtimestamp(x[2].stat().st_mtime)))

            for prefix, ts, p in file_entries:
                logging.info(f'Processing {p.name} (prefix={prefix}, ts={ts})')
                process_video(str(p), project=args.project, model=model)
                db.process_yolo_labels_for_video(args.project, p.stem, p.name, ts, max_gap_seconds=args.max_gap)
                db.close_inactive_tracks(older_than_seconds=args.max_gap * 2)
        else:
            ts = parse_timestamp_from_filename(path.name)
            process_video(str(path), project=args.project, model=model)
            db.process_yolo_labels_for_video(args.project, path.stem, path.name, ts, max_gap_seconds=args.max_gap)
    finally:
        db.close()


if __name__ == '__main__':
    main()
