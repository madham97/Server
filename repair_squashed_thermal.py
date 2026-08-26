#!/usr/bin/env python3
"""
Repair thermal frames stored with their axes squashed.

Between 2026-08-22 12:17 and the geometry fix, the recorder derived its native output size by
unpacking `mi48.fpa_shape` as (rows, cols). The MI48 reports it as (width, height) — (80, 62) —
so every native frame was resized to a 62x80 portrait canvas: the same landscape scene,
anisotropically squashed.

Nothing downstream complained, which is why it ran for a day. `align_thermal` scales each axis
independently, and the two errors cancel exactly:

    stored 62x80 gets    x 30.968  y 13.500
    squash baked in      x  0.775  y  1.290
    net                  x 24.000  y 17.419   = a correct 80x62 frame

So the *aligned* outputs were right all along. What is wrong is the stored native frame itself —
the `?native=1` view, anything measuring on the sensor grid, and the `thermal_geometry` label,
which reads `upsampled_rgb` because the size does not equal SENSOR_FRAME_SIZE.

**The repair is lossy in x, and cannot not be.** The squash resampled 80 columns down to 62;
those 18 columns are gone. Undoing it restores correct geometry and 80 columns, but they are
interpolated from 62 samples. Vertically nothing was lost (62 was stretched to 80 and comes
back). Frames are marked so this is never mistaken for pristine sensor data.

Selection is on the stored dimensions being exactly 62x80, not on the `thermal_geometry` label:
genuine upsampled legacy frames (1920x1080, 1440x810) also carry that label and must not be
touched.

Usage:
    python3 repair_squashed_thermal.py              # dry run, reports what it would do
    python3 repair_squashed_thermal.py --apply      # repair, backing originals up first
    python3 repair_squashed_thermal.py --apply --no-realign
"""

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

from config import THERMAL_DIR
from thermal_align import (SENSOR_FRAME_SIZE, align_thermal, capture_timestamp,
                            find_profile_for_timestamp)

# (width, height) as the squashed frames were written — the transpose of the real sensor grid.
SQUASHED_SIZE = (SENSOR_FRAME_SIZE[1], SENSOR_FRAME_SIZE[0])

BACKUP_DIR = THERMAL_DIR / '.squashed_originals'


def png_size(path: Path) -> tuple[int, int] | None:
    """(width, height) from a PNG's IHDR, without decoding the image.

    The archive holds ~10k frames, most of them 1920x1080; decoding every one to read two
    integers takes minutes, while the header is the first 24 bytes of the file."""
    try:
        with path.open('rb') as f:
            head = f.read(24)
    except OSError:
        return None
    if len(head) < 24 or head[:8] != b'\x89PNG\r\n\x1a\n' or head[12:16] != b'IHDR':
        return None
    return (int.from_bytes(head[16:20], 'big'), int.from_bytes(head[20:24], 'big'))


def find_squashed(thermal_dir: Path) -> list[Path]:
    """Every stored thermal frame that is exactly 62x80. Selecting on the real dimensions rather
    than on the sidecar label means a frame already repaired is not picked up twice, and genuine
    upsampled legacy frames — which share the `upsampled_rgb` label — are never touched."""
    hits = []
    for png in sorted(thermal_dir.glob('*_thermal.png')):
        if png.name.endswith('_aligned.png'):
            continue
        size = png_size(png)
        if size is None:
            logging.warning(f'Unreadable PNG header, skipping: {png.name}')
            continue
        if size == SQUASHED_SIZE:
            hits.append(png)
    return hits


def repair_frame(png: Path, backup_dir: Path, realign: bool) -> str:
    """Restore one frame's geometry, update its sidecar, and regenerate its aligned counterpart.

    The original is copied out before anything is overwritten — the resample is not perfectly
    invertible, so the pre-repair file is the only record of exactly what was received."""
    img = cv2.imread(str(png), cv2.IMREAD_UNCHANGED)
    backup_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(png, backup_dir / png.name)

    # INTER_CUBIC to mirror the cubic that produced the squash. One axis is an upsample and the
    # other a downsample, so no single "correct" kernel exists; matching the forward operation
    # keeps the error symmetric rather than compounding a different one on top.
    repaired = cv2.resize(img, SENSOR_FRAME_SIZE, interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(str(png), repaired)

    stem = png.name.removesuffix('_thermal.png')
    sidecar = THERMAL_DIR / f'{stem}_thermal.json'
    if sidecar.exists():
        try:
            meta = json.loads(sidecar.read_text())
        except json.JSONDecodeError:
            meta = {}
        meta.update({
            'thermal_geometry': 'native_sensor',
            'thermal_width': SENSOR_FRAME_SIZE[0],
            'thermal_height': SENSOR_FRAME_SIZE[1],
            # Provenance, so this frame is never read as pristine sensor output: its horizontal
            # detail was resampled 80 -> 62 -> 80 and cannot be recovered.
            'thermal_geometry_repaired': True,
            'thermal_geometry_repair_note': (
                'stored 62x62-transposed (62x80) by a recorder that misread fpa_shape; '
                'geometry restored, horizontal detail interpolated from 62 columns'
            ),
        })
        sidecar.write_text(json.dumps(meta))

    if not realign:
        return 'repaired'

    aligned_path = THERMAL_DIR / f'{stem}_thermal_aligned.png'
    if not aligned_path.exists():
        return 'repaired (no aligned counterpart)'
    timestamp = capture_timestamp(stem)
    profile = find_profile_for_timestamp(timestamp) if timestamp else None
    if profile is None:
        return 'repaired (no covering calibration profile; aligned left as-is)'
    homography = np.array(profile['homography'], dtype=np.float64)
    aligned = align_thermal(png, homography, ref_size=profile.get('ref_size'))
    cv2.imwrite(str(aligned_path), aligned)
    return 'repaired + realigned'


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--apply', action='store_true',
                        help='actually repair (default is a dry run)')
    parser.add_argument('--no-realign', action='store_true',
                        help='skip regenerating aligned counterparts')
    parser.add_argument('--thermal-dir', default=str(THERMAL_DIR))
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
    thermal_dir = Path(args.thermal_dir)

    squashed = find_squashed(thermal_dir)
    print(f'Frames stored at {SQUASHED_SIZE[0]}x{SQUASHED_SIZE[1]} (squashed): {len(squashed)}')
    if not squashed:
        print('Nothing to repair.')
        return 0
    print(f'  first: {squashed[0].name}')
    print(f'  last : {squashed[-1].name}')

    if not args.apply:
        print(f'\nDry run. Would restore each to {SENSOR_FRAME_SIZE[0]}x{SENSOR_FRAME_SIZE[1]}, '
              f'back originals up to {BACKUP_DIR}/,')
        print('update sidecars, and regenerate aligned counterparts. Re-run with --apply.')
        return 0

    counts: dict[str, int] = {}
    for i, png in enumerate(squashed, 1):
        try:
            outcome = repair_frame(png, BACKUP_DIR, realign=not args.no_realign)
        except Exception as e:
            logging.exception(f'Failed on {png.name}')
            outcome = f'failed: {type(e).__name__}'
        counts[outcome] = counts.get(outcome, 0) + 1
        if i % 100 == 0:
            print(f'  ...{i}/{len(squashed)}')

    print()
    for outcome, n in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f'  {n:5d}  {outcome}')
    print(f'\nOriginals preserved in {BACKUP_DIR}/')
    return 0


if __name__ == '__main__':
    sys.exit(main())
