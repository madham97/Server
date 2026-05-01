import os
import sys
import logging
from pathlib import Path

import torch
from ultralytics import YOLO

from config import MODEL_WEIGHTS, MODEL_CONF, MODEL_IOU, MODEL_IMGSZ, PROCESSED_DIR

sys.path.insert(0, str(Path(__file__).parent))

IMAGE_EXTS = {'.jpg', '.jpeg', '.png'}

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

_DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def process_image(image_path: str, weights: str = MODEL_WEIGHTS, project: str = str(PROCESSED_DIR),
                  device: str = _DEFAULT_DEVICE, conf: float = MODEL_CONF, iou: float = MODEL_IOU,
                  imgsz: int = MODEL_IMGSZ, model: YOLO | None = None):
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
        imgsz=imgsz,
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
    args = parser.parse_args()

    path = Path(args.path)
    if not path.exists():
        raise SystemExit(f'Path not found: {path}')

    logging.info('Loading YOLO model for batch processing...')
    model = YOLO("models/best.pt")

    if path.is_dir():
        files = sorted([p for p in path.iterdir() if p.suffix.lower() in IMAGE_EXTS and p.is_file()])
        for p in files:
            logging.info(f'Processing {p.name}')
            process_image(str(p), project=args.project, model=model)
    else:
        process_image(str(path), project=args.project, model=model)


if __name__ == '__main__':
    main()
