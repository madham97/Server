import os
import sys
import logging
from pathlib import Path
from datetime import datetime

import torch
from ultralytics import YOLO
from tracker import TrackerDB, parse_timestamp_from_filename

sys.path.insert(0, str(Path(__file__).parent))

IMAGE_EXTS = {'.jpg', '.jpeg', '.png'}

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

_DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def process_image(image_path: str, weights: str = "models/best.pt", project: str = 'processed',
                  device: str = _DEFAULT_DEVICE, conf: float = 0.25, iou: float = 0.45,
                  model: YOLO | None = None):
    """Run YOLO inference on a single image and save results into `project`."""
    if not os.path.exists(weights):
        raise FileNotFoundError(f'Weights file not found: {weights}')
    if not os.path.exists(image_path):
        raise FileNotFoundError(f'Image file not found: {image_path}')

    os.makedirs(project, exist_ok=True)
    image_name = Path(image_path).stem

    if model is None:
        logging.info('Loading model...')
        model = YOLO(weights)

    logging.info(f'Running inference on {image_path}...')
    results = model.predict(
        source=image_path,
        imgsz=640,
        device=device,
        conf=conf,
        iou=iou,
        save=True,
        project=project,
        name=image_name,
        exist_ok=True,
        verbose=False,
        save_txt=True,
        save_conf=True,
    )
    logging.info(f'Inference complete. Results saved to {os.path.join(project, image_name)}')
    return results


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('path', nargs='?', default='uploads', help='Image file or directory to process')
    parser.add_argument('--project', default='processed')
    parser.add_argument('--db', default='objects.db')
    parser.add_argument('--max-gap', type=int, default=300)
    args = parser.parse_args()

    path = Path(args.path)
    if not path.exists():
        raise SystemExit(f'Path not found: {path}')

    db = TrackerDB(args.db)
    logging.info('Loading YOLO model for batch processing...')
    model = YOLO("models/best.pt")

    try:
        if path.is_dir():
            files = sorted([p for p in path.iterdir() if p.suffix.lower() in IMAGE_EXTS and p.is_file()])
            for p in files:
                ts = parse_timestamp_from_filename(p.name)
                logging.info(f'Processing {p.name} (ts={ts})')
                process_image(str(p), project=args.project, model=model)
                db.process_yolo_labels_for_video(args.project, p.stem, p.name, ts, max_gap_seconds=args.max_gap)
                db.close_inactive_tracks(older_than_seconds=args.max_gap * 2)
        else:
            ts = parse_timestamp_from_filename(path.name)
            process_image(str(path), project=args.project, model=model)
            db.process_yolo_labels_for_video(args.project, path.stem, path.name, ts, max_gap_seconds=args.max_gap)
    finally:
        db.close()


if __name__ == '__main__':
    main()
