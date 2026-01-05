import os
import sys
import logging
from pathlib import Path
from ultralytics import YOLO

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

    os.makedirs(project, exist_ok=True)

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
        project="processed",
        name=project,
        verbose=False,
        save_txt=True,
        save_conf=True,
        # stream=True
    )
    logging.info(f'Video inference complete. Results saved to {project}')
    return results


def main():
    """If run as a script, process either a single file or all supported files in a directory."""
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('path', nargs='?', default='uploads', help='Video file or directory to process')
    parser.add_argument('--project', default='processed')
    args = parser.parse_args()

    path = Path(args.path)
    if not path.exists():
        raise SystemExit(f'Path not found: {path}')

    if path.is_dir():
        # Process alphabetically
        files = sorted([p for p in path.iterdir() if p.suffix.lower() in VIDEO_EXTS and p.is_file()])
        for f in files:
            process_video(str(f), project=args.project)
    else:
        process_video(str(path), project=args.project)


if __name__ == '__main__':
    main()
