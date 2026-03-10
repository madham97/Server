import os
import re
import json
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional


def parse_timestamp_from_filename(filename: str) -> Optional[datetime]:
    """Parse timestamps in filenames like: prefix_YYYYMMDDThhmmssZ
    Returns a naive UTC datetime or None.
    """
    m = re.search(r"(?P<ts>\d{8}T\d{6}Z)", filename)
    if not m:
        return None
    ts = m.group('ts')  # e.g. 20260103T112511Z
    return datetime.strptime(ts, "%Y%m%dT%H%M%SZ")


def iou_normalized(boxA: Tuple[float, float, float, float], boxB: Tuple[float, float, float, float]) -> float:
    # boxes are x_center, y_center, w, h (normalized)
    def to_xyxy(b):
        x, y, w, h = b
        x1 = x - w / 2
        y1 = y - h / 2
        x2 = x + w / 2
        y2 = y + h / 2
        return x1, y1, x2, y2

    a = to_xyxy(boxA)
    b = to_xyxy(boxB)
    xA = max(a[0], b[0])
    yA = max(a[1], b[1])
    xB = min(a[2], b[2])
    yB = min(a[3], b[3])

    interW = max(0.0, xB - xA)
    interH = max(0.0, yB - yA)
    interArea = interW * interH

    areaA = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    areaB = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])

    denom = areaA + areaB - interArea
    if denom <= 0:
        return 0.0
    return interArea / denom


class TrackerDB:
    def __init__(self, db_path: str = 'objects.db'):
        self.db_path = db_path
        self._conn = sqlite3.connect(self.db_path, detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES)
        self._conn.row_factory = sqlite3.Row
        self._ensure_schema()

    def _ensure_schema(self):
        c = self._conn.cursor()
        c.execute('''
        CREATE TABLE IF NOT EXISTS objects (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            class INTEGER,
            first_seen TEXT,
            last_seen TEXT,
            first_conf REAL,
            last_conf REAL,
            frames_seen INTEGER DEFAULT 0,
            avg_conf REAL DEFAULT 0.0,
            metadata TEXT
        )
        ''')
        c.execute('''
        CREATE TABLE IF NOT EXISTS sightings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            object_id INTEGER,
            file_name TEXT,
            frame INTEGER,
            bbox TEXT,
            conf REAL,
            class INTEGER,
            time TEXT
        )
        ''')
        c.execute('''
        CREATE TABLE IF NOT EXISTS active_tracks (
            object_id INTEGER PRIMARY KEY,
            last_file TEXT,
            last_frame INTEGER,
            last_bbox TEXT,
            last_seen TEXT,
            last_conf REAL
        )
        ''')
        self._conn.commit()

    def close(self):
        self._conn.close()

    def _insert_object(self, class_id: int, time_iso: str, conf: float) -> int:
        c = self._conn.cursor()
        c.execute('INSERT INTO objects (class, first_seen, last_seen, first_conf, last_conf, frames_seen, avg_conf) VALUES (?, ?, ?, ?, ?, ?, ?)',
                  (class_id, time_iso, time_iso, conf, conf, 0, conf))
        self._conn.commit()
        return c.lastrowid

    def _update_object_on_add_sighting(self, object_id: int, conf: float, time_iso: str):
        c = self._conn.cursor()
        # update frames_seen and avg_conf, and last_seen/last_conf
        c.execute('SELECT frames_seen, avg_conf FROM objects WHERE id = ?', (object_id,))
        row = c.fetchone()
        if row is None:
            return
        frames = row['frames_seen'] or 0
        avg = row['avg_conf'] or 0.0
        new_frames = frames + 1
        new_avg = ((avg * frames) + conf) / new_frames
        c.execute('UPDATE objects SET frames_seen = ?, avg_conf = ?, last_seen = ?, last_conf = ? WHERE id = ?',
                  (new_frames, new_avg, time_iso, conf, object_id))
        self._conn.commit()

    def add_sighting(self, object_id: int, file_name: str, frame: int, bbox: Tuple[float, float, float, float], conf: float, class_id: int, time_iso: str):
        c = self._conn.cursor()
        c.execute('INSERT INTO sightings (object_id, file_name, frame, bbox, conf, class, time) VALUES (?, ?, ?, ?, ?, ?, ?)',
                  (object_id, file_name, frame, json.dumps(bbox), conf, class_id, time_iso))
        self._conn.commit()
        # update object aggregates
        self._update_object_on_add_sighting(object_id, conf, time_iso)

    def update_active_track(self, object_id: int, last_file: str, last_frame: int, last_bbox: Tuple[float, float, float, float], last_seen_iso: str, last_conf: float):
        c = self._conn.cursor()
        c.execute('INSERT OR REPLACE INTO active_tracks (object_id, last_file, last_frame, last_bbox, last_seen, last_conf) VALUES (?, ?, ?, ?, ?, ?)',
                  (object_id, last_file, last_frame, json.dumps(last_bbox), last_seen_iso, last_conf))
        self._conn.commit()

    def remove_active_track(self, object_id: int):
        c = self._conn.cursor()
        c.execute('DELETE FROM active_tracks WHERE object_id = ?', (object_id,))
        self._conn.commit()

    def match_active(self, bbox: Tuple[float, float, float, float], class_id: int, file_time: datetime, max_gap_seconds: int = 300, iou_threshold: float = 0.25) -> Optional[int]:
        c = self._conn.cursor()
        rows = c.execute('SELECT * FROM active_tracks').fetchall()
        best_match = None
        best_score = 0.0
        for r in rows:
            try:
                last_seen = datetime.fromisoformat(r['last_seen'])
            except Exception:
                continue
            gap = (file_time - last_seen).total_seconds()
            if gap < 0:
                gap = abs(gap)
            if gap > max_gap_seconds:
                continue
            last_bbox = json.loads(r['last_bbox'])
            score = iou_normalized(bbox, tuple(last_bbox))
            if score > best_score and score >= iou_threshold:
                best_score = score
                best_match = r['object_id']
        return best_match

    def process_yolo_labels_for_video(self, project: str, video_basename: str, video_filename: str, video_time: Optional[datetime], max_gap_seconds: int = 300):
        """Parse saved YOLO label files for a processed video and update DB/link tracks across files.
        Expects labels at: {project}/{video_basename}/labels/{video_basename}_*.txt (flat YOLO output format)
        """
        labels_dir = os.path.join(project, video_basename, 'labels')
        if not os.path.isdir(labels_dir):
            return
        # gather frame label files matching the video basename (e.g. video_20260103T112511Z_*.txt)
        # Sort numerically by frame number, not alphabetically
        frame_files = [p for p in os.listdir(labels_dir) if p.startswith(video_basename) and p.lower().endswith('.txt')]
        # Extract frame numbers and sort numerically
        def get_frame_number(filename):
            match = re.search(r'_(\d+)\.txt$', filename)
            return int(match.group(1)) if match else 0
        frames = sorted(frame_files, key=get_frame_number)
        tracks: Dict[str, List[Dict]] = {}
        for f in frames:
            # Extract frame number from filename like video_20260103T112511Z_100.txt
            frame_matches = re.findall(r"_(\d+)\.txt$", f)
            frame_no = int(frame_matches[0]) if frame_matches else 0
            path = os.path.join(labels_dir, f)
            with open(path, 'r') as fh:
                for line in fh:
                    parts = line.strip().split()
                    if not parts:
                        continue
                    # YOLO txt with tracking: class track_id x y w h conf
                    if len(parts) == 7:
                        class_id = int(parts[0])
                        track_id = parts[1]
                        x, y, w, h = map(float, parts[2:6])
                        conf = float(parts[6])
                    elif len(parts) == 6:
                        # no track id, use generated per-detection track id
                        class_id = int(parts[0])
                        track_id = f"det_{frame_no}_{len(tracks)}"
                        x, y, w, h = map(float, parts[1:5])
                        conf = float(parts[5])
                    else:
                        # unexpected format; skip
                        continue
                    tracks.setdefault(track_id, []).append({
                        'frame': frame_no,
                        'bbox': (x, y, w, h),
                        'conf': conf,
                        'class': class_id
                    })
        # For each track, decide if it maps to an existing persistent object
        for track_id, sightings in tracks.items():
            first = sightings[0]
            last = sightings[-1]
            # choose time = video_time if available
            time_iso = video_time.isoformat() if video_time else datetime.utcnow().isoformat()
            matched = self.match_active(first['bbox'], first['class'], video_time or datetime.utcnow(), max_gap_seconds=max_gap_seconds)
            if matched is None:
                # create new object
                obj_id = self._insert_object(first['class'], time_iso, first['conf'])
            else:
                obj_id = matched
            # add all sightings
            for s in sightings:
                self.add_sighting(obj_id, video_filename, s['frame'], s['bbox'], s['conf'], s['class'], time_iso)
            # update last info in active_tracks
            self.update_active_track(obj_id, video_filename, last['frame'], last['bbox'], time_iso, last['conf'])

    def close_inactive_tracks(self, older_than_seconds: int = 600):
        c = self._conn.cursor()
        rows = c.execute('SELECT * FROM active_tracks').fetchall()
        cutoff = datetime.utcnow() - timedelta(seconds=older_than_seconds)
        for r in rows:
            try:
                last_seen = datetime.fromisoformat(r['last_seen'])
            except Exception:
                continue
            if last_seen < cutoff:
                # remove from active
                self.remove_active_track(r['object_id'])
                # nothing else required for now; objects table preserves last_seen


if __name__ == '__main__':
    # small smoke test
    db = TrackerDB(':memory:')
    now = datetime.utcnow()
    obj = db._insert_object(0, now.isoformat(), 0.8)
    db.add_sighting(obj, 'video_20260103T112511Z.mp4', 1, (0.5, 0.5, 0.1, 0.1), 0.8, 0, now.isoformat())
    db.update_active_track(obj, 'video_20260103T112511Z.mp4', 100, (0.5, 0.5, 0.1, 0.1), now.isoformat(), 0.8)
    print('ok')
