import shutil
import threading
from datetime import datetime

from ultralytics import YOLO

from config import TRAINING_RUNS_DIR, CANDIDATE_WEIGHTS

_status: dict = {
    "state": "idle",   # idle | running | complete | failed
    "epoch": 0,
    "total_epochs": 0,
    "run_dir": None,
    "metrics": {},
    "error": None,
}
_lock = threading.Lock()


def get_status() -> dict:
    with _lock:
        return dict(_status)


def start_training(base_model: str, epochs: int, imgsz: int, dataset_yaml: str) -> bool:
    with _lock:
        if _status["state"] == "running":
            return False
        _status.update({
            "state": "running",
            "epoch": 0,
            "total_epochs": epochs,
            "run_dir": None,
            "metrics": {},
            "error": None,
        })
    thread = threading.Thread(
        target=_run,
        args=(base_model, epochs, imgsz, dataset_yaml),
        daemon=True,
    )
    thread.start()
    return True


def _run(base_model: str, epochs: int, imgsz: int, dataset_yaml: str):
    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    TRAINING_RUNS_DIR.mkdir(parents=True, exist_ok=True)
    run_dir = TRAINING_RUNS_DIR / timestamp

    with _lock:
        _status["run_dir"] = str(run_dir)

    try:
        model = YOLO(base_model)

        def on_epoch_end(trainer):
            with _lock:
                _status["epoch"] = trainer.epoch + 1
                _status["metrics"] = {
                    k: float(v)
                    for k, v in trainer.metrics.items()
                    if isinstance(v, (int, float))
                }

        model.add_callback("on_train_epoch_end", on_epoch_end)
        model.train(
            data=dataset_yaml,
            epochs=epochs,
            imgsz=imgsz,
            project=str(TRAINING_RUNS_DIR),
            name=timestamp,
            exist_ok=True,
            workers=0,
        )

        best_pt = run_dir / "weights" / "best.pt"
        if best_pt.exists():
            CANDIDATE_WEIGHTS.parent.mkdir(exist_ok=True)
            shutil.copy2(best_pt, CANDIDATE_WEIGHTS)

        with _lock:
            _status["state"] = "complete"

    except Exception as e:
        with _lock:
            _status["state"] = "failed"
            _status["error"] = str(e)
