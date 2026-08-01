import json
import logging
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from config import CALIBRATION_DIR, CALIBRATION_FILE, CALIBRATION_PROFILES_FILE, THERMAL_DIR, UPLOAD_DIR

sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

_CORNER_CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)


def _parse_ts(ts: str) -> datetime:
    """Parses a 'YYYY-MM-DDTHH:MM:SSZ'-style timestamp (the format used throughout this
    codebase — sidecar JSON, calibrated_at, effective_from/until) into an aware datetime."""
    return datetime.fromisoformat(ts.replace('Z', '+00:00'))


def load_profiles() -> list[dict]:
    """All saved calibration profiles, sorted by effective_from.

    A profile is a homography/affine transform plus the time window it applies to
    (`effective_from`, and `effective_until` or None for open-ended) — see
    `find_profile_for_timestamp`. On first call, if no profiles file exists yet but the old
    single-calibration CALIBRATION_FILE does, migrates it into one profile covering all time
    (effective_from the Unix epoch, open-ended), so an existing deployment's calibration
    keeps working unchanged until a real recalibration (e.g. after moving the camera)
    creates a second, properly time-scoped profile."""
    if not CALIBRATION_PROFILES_FILE.exists():
        if CALIBRATION_FILE.exists():
            legacy = json.loads(CALIBRATION_FILE.read_text())
            profile = {
                'id': 'legacy',
                'label': 'Migrated from single-file calibration',
                'effective_from': '1970-01-01T00:00:00Z',
                'effective_until': None,
                **legacy,
            }
            save_profiles([profile])
            logging.info(f'Migrated legacy {CALIBRATION_FILE} into a single all-time calibration profile')
            return [profile]
        return []
    return json.loads(CALIBRATION_PROFILES_FILE.read_text())


def save_profiles(profiles: list[dict]) -> None:
    CALIBRATION_DIR.mkdir(exist_ok=True)
    CALIBRATION_PROFILES_FILE.write_text(json.dumps(profiles, indent=2))


def _windows_overlap(a_from: datetime, a_until: datetime | None,
                      b_from: datetime, b_until: datetime | None) -> bool:
    a_until = a_until or datetime.max.replace(tzinfo=timezone.utc)
    b_until = b_until or datetime.max.replace(tzinfo=timezone.utc)
    return a_from < b_until and b_from < a_until


def _validate_window(profiles: list[dict], effective_from: str, effective_until: str | None,
                      exclude_id: str | None = None) -> None:
    """Raises ValueError if [effective_from, effective_until) would overlap any other saved
    profile's window — two profiles both claiming the same moment is ambiguous, since a
    capture at that moment couldn't tell which one to use."""
    new_from = _parse_ts(effective_from)
    new_until = _parse_ts(effective_until) if effective_until else None
    if new_until and new_until <= new_from:
        raise ValueError('effective_until must be after effective_from')

    for p in profiles:
        if p['id'] == exclude_id:
            continue
        p_from = _parse_ts(p['effective_from'])
        p_until = _parse_ts(p['effective_until']) if p.get('effective_until') else None
        if _windows_overlap(new_from, new_until, p_from, p_until):
            raise ValueError(
                f"Effective window {effective_from} .. {effective_until or 'open-ended'} overlaps "
                f"existing profile {p['id']!r} ({p['effective_from']} .. {p.get('effective_until') or 'open-ended'})"
            )


def find_profile_for_timestamp(timestamp: str, profiles: list[dict] | None = None) -> dict | None:
    """Which saved calibration profile applies to a capture at this ISO timestamp — the one
    whose [effective_from, effective_until) window contains it. Returns None if no profile
    covers it (nothing calibrated yet, or a genuine gap between two profiles' windows) —
    callers should treat that as 'don't align this one', not an error."""
    if profiles is None:
        profiles = load_profiles()
    ts = _parse_ts(timestamp)
    for p in profiles:
        p_from = _parse_ts(p['effective_from'])
        p_until = _parse_ts(p['effective_until']) if p.get('effective_until') else None
        if p_from <= ts and (p_until is None or ts < p_until):
            return p
    return None


def _upsert_profile(new_fields: dict, effective_from: str, effective_until: str | None,
                     profile_id: str | None = None, label: str = "") -> dict:
    """Creates a new calibration profile, or replaces an existing one in place (by id) —
    the "always editable" model: refitting an old profile with more box/point pairs updates
    it in place rather than creating a duplicate, but its effective window can also move
    (still validated against every *other* profile). Either way the window is validated
    against every other saved profile first. Returns the saved profile."""
    profiles = load_profiles()
    _validate_window(profiles, effective_from, effective_until, exclude_id=profile_id)

    now = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
    if profile_id:
        existing = next((p for p in profiles if p['id'] == profile_id), None)
        if existing is None:
            raise ValueError(f'No existing calibration profile with id {profile_id!r}')
        profile = {
            **existing, **new_fields,
            'effective_from': effective_from,
            'effective_until': effective_until,
            'calibrated_at': now,
            'label': label or existing.get('label', ''),
        }
        profiles = [profile if p['id'] == profile_id else p for p in profiles]
    else:
        profile = {
            'id': f"prof_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}_{uuid.uuid4().hex[:6]}",
            'label': label,
            'effective_from': effective_from,
            'effective_until': effective_until,
            'calibrated_at': now,
            **new_fields,
        }
        profiles.append(profile)

    profiles.sort(key=lambda p: p['effective_from'])
    save_profiles(profiles)
    logging.info(f"Saved calibration profile {profile['id']} "
                 f"({effective_from} .. {effective_until or 'open-ended'})")
    return profile


def _find_corners(image_path: str, pattern_size: tuple[int, int]) -> np.ndarray:
    """Locate checkerboard inner-corner points in a single image. Raises if not found."""
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f'Could not read image: {image_path}')

    found, corners = cv2.findChessboardCorners(
        img, pattern_size,
        flags=cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE,
    )
    if not found:
        raise RuntimeError(
            f'Checkerboard ({pattern_size[0]}x{pattern_size[1]} inner corners) not found in {image_path}'
        )

    corners = cv2.cornerSubPix(img, corners, (11, 11), (-1, -1), _CORNER_CRITERIA)
    return corners.reshape(-1, 2)


def _score_points(homography: np.ndarray, thermal_pts: np.ndarray, rgb_pts: np.ndarray,
                   mask: np.ndarray | None, stems: list[str] | None = None) -> list[dict]:
    """Per-point reprojection error and RANSAC inlier/outlier status — the diagnostics
    attached to a point-based calibration profile. Pure function; does not save anything
    (see `_upsert_profile` for that)."""
    inlier_flags = mask.ravel().astype(bool).tolist() if mask is not None else [True] * len(thermal_pts)

    ones = np.ones((len(thermal_pts), 1))
    projected = (homography @ np.hstack([thermal_pts, ones]).T).T
    projected = projected[:, :2] / projected[:, 2:3]
    errors = np.linalg.norm(projected - rgb_pts, axis=1)

    points = [{
        'stem': stems[i] if stems else None,
        'thermal': thermal_pts[i].tolist(),
        'rgb': rgb_pts[i].tolist(),
        'inlier': bool(inlier_flags[i]),
        'error_px': round(float(errors[i]), 2),
    } for i in range(len(thermal_pts))]

    inliers = sum(inlier_flags)
    logging.info(f'Homography solved from {len(thermal_pts)} point pairs ({inliers} inliers)')
    return points


def _capture_size(stem: str) -> tuple[int, int] | None:
    """(width, height) of a capture's thermal PNG — by construction (see the Pi client's
    fusion step) the paired RGB frame is resized to exactly this size, so one lookup
    covers both."""
    img = cv2.imread(str(THERMAL_DIR / f'{stem}_thermal.png'), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    h, w = img.shape[:2]
    return w, h


def _normalize_coords_to_reference(entries: list[list[float]],
                                    stems: list[str]) -> tuple[list[list[float]], tuple[int, int]]:
    """Rescales each entry (a flat [x,y] point or [x1,y1,x2,y2] box, in that capture's own
    pixel coordinates) onto one common reference resolution — the most common size among
    the given stems — so pairs drawn on captures from different device/firmware resolutions
    can be fit together instead of silently corrupting the fit as if they shared one
    coordinate system. Raises if a differently-sized capture has a different aspect ratio
    than the reference, since a per-axis scale factor can't correct for that."""
    from collections import Counter

    sizes = {}
    for stem in set(stems):
        size = _capture_size(stem)
        if size is None:
            raise ValueError(f'Could not determine capture resolution for {stem}')
        sizes[stem] = size

    ref_w, ref_h = Counter(sizes.values()).most_common(1)[0][0]
    ref_aspect = ref_w / ref_h

    normalized = []
    for entry, stem in zip(entries, stems):
        w, h = sizes[stem]
        if (w, h) == (ref_w, ref_h):
            normalized.append(entry)
            continue
        aspect = w / h
        if abs(aspect - ref_aspect) / ref_aspect > 0.02:
            raise ValueError(
                f'{stem} is {w}x{h} ({aspect:.3f} aspect), too different from the '
                f'{ref_w}x{ref_h} ({ref_aspect:.3f} aspect) reference used by other pairs '
                f'in this submission to rescale — remove it or calibrate separately'
            )
        kx, ky = ref_w / w, ref_h / h
        normalized.append([v * (kx if i % 2 == 0 else ky) for i, v in enumerate(entry)])

    return normalized, (ref_w, ref_h)


def calibrate(rgb_calib_path: str, thermal_calib_path: str, effective_from: str,
              effective_until: str | None = None, pattern_size: tuple[int, int] = (9, 6),
              profile_id: str | None = None, label: str = "") -> np.ndarray:
    """Compute the homography that warps thermal-frame pixels into RGB-frame pixels,
    from a single photo pair of a checkerboard visible in both spectra. Saves the result as
    a calibration profile effective over [effective_from, effective_until) and returns the
    3x3 matrix — see `_upsert_profile`."""
    rgb_pts = _find_corners(rgb_calib_path, pattern_size)
    thermal_pts = _find_corners(thermal_calib_path, pattern_size)

    homography, mask = cv2.findHomography(thermal_pts, rgb_pts, cv2.RANSAC, 3.0)
    if homography is None:
        raise RuntimeError('Homography estimation failed')

    points = _score_points(homography, thermal_pts, rgb_pts, mask)
    _upsert_profile({
        'homography': homography.tolist(),
        'point_count': len(thermal_pts),
        'inlier_count': sum(1 for p in points if p['inlier']),
        'points': points,
        'method': 'checkerboard',
        'pattern_size': list(pattern_size),
        'rgb_calib_image': str(rgb_calib_path),
        'thermal_calib_image': str(thermal_calib_path),
    }, effective_from, effective_until, profile_id=profile_id, label=label)
    return homography


def calibrate_from_points(thermal_points: list[list[float]], rgb_points: list[list[float]],
                           effective_from: str, effective_until: str | None = None,
                           stems: list[str] | None = None, source_stem: str = "",
                           profile_id: str | None = None, label: str = "") -> tuple[np.ndarray, list[dict], str]:
    """Compute the thermal-to-RGB homography from manually picked corresponding points
    on an existing (non-checkerboard) capture pair. Needs at least 4 point pairs; more,
    spread across the frame, gives a more reliable fit. Saves as a calibration profile
    effective over [effective_from, effective_until), including per-point reprojection
    error and RANSAC inlier/outlier status — see `_upsert_profile`. Returns
    (homography, points, profile_id) — the last is the new or updated profile's id, for
    scoping a subsequent `align_all(profile_id=...)` to just the frames it governs."""
    if len(thermal_points) != len(rgb_points):
        raise ValueError('thermal_points and rgb_points must be the same length')
    if len(thermal_points) < 4:
        raise ValueError('Need at least 4 point pairs to solve a homography')

    ref_size = None
    if stems:
        thermal_points, ref_size = _normalize_coords_to_reference(thermal_points, stems)
        rgb_points, _ = _normalize_coords_to_reference(rgb_points, stems)

    thermal_pts = np.array(thermal_points, dtype=np.float64)
    rgb_pts = np.array(rgb_points, dtype=np.float64)

    method = cv2.RANSAC if len(thermal_pts) > 4 else 0
    homography, mask = cv2.findHomography(thermal_pts, rgb_pts, method, 5.0)
    if homography is None:
        raise RuntimeError('Homography estimation failed')

    points = _score_points(homography, thermal_pts, rgb_pts, mask, stems=stems)
    profile = _upsert_profile({
        'homography': homography.tolist(),
        'point_count': len(thermal_pts),
        'inlier_count': sum(1 for p in points if p['inlier']),
        'points': points,
        'method': 'manual_points',
        'source_stem': source_stem,
        'ref_size': list(ref_size) if ref_size else None,
    }, effective_from, effective_until, profile_id=profile_id, label=label)
    return homography, points, profile['id']


def _similarity_transform(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Closed-form least-squares similarity transform (uniform scale + rotation +
    translation, no shear/perspective) mapping src points onto dst points — the Umeyama
    method, via SVD. Returns a 3x3 homogeneous matrix compatible with cv2.warpPerspective.
    Fewer degrees of freedom than a full homography, so it needs less (and less precise)
    input data to fit stably — appropriate for a fixed rig with no real perspective
    distortion between the two cameras."""
    mu_src, mu_dst = src.mean(axis=0), dst.mean(axis=0)
    src_c, dst_c = src - mu_src, dst - mu_dst

    var_src = (src_c ** 2).sum() / len(src)
    cov = (dst_c.T @ src_c) / len(src)

    U, S, Vt = np.linalg.svd(cov)
    d = np.ones(2)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        d[-1] = -1
    R = U @ np.diag(d) @ Vt
    scale = np.trace(np.diag(S) @ np.diag(d)) / var_src if var_src > 0 else 1.0
    t = mu_dst - scale * R @ mu_src

    H = np.eye(3)
    H[:2, :2] = scale * R
    H[:2, 2] = t
    return H


def _affine_transform(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Closed-form ordinary-least-squares affine transform (independent x/y scale,
    rotation, shear, translation — 6 DOF) mapping src points onto dst points. Unlike a
    similarity transform this allows different scale factors per axis, which matters
    here: the thermal channel is resized on-device from its native 80x62 sensor to a
    different aspect ratio than the RGB frame (see docs/decisions or the Pi client repo),
    so the two channels are already stretched non-uniformly relative to each other before
    calibration ever sees them."""
    n = len(src)
    design = np.hstack([src, np.ones((n, 1))])  # [x, y, 1] per row
    coeffs_x, *_ = np.linalg.lstsq(design, dst[:, 0], rcond=None)
    coeffs_y, *_ = np.linalg.lstsq(design, dst[:, 1], rcond=None)
    H = np.eye(3)
    H[0, :] = coeffs_x
    H[1, :] = coeffs_y
    return H


def _box_iou(box_a: list[float], box_b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _transform_box(box: list[float], H: np.ndarray) -> list[float]:
    """Applies H to all 4 corners of an (axis-aligned) box and returns the axis-aligned
    bounding box of the result — handles the small rotation component of a similarity
    transform without assuming corners map straight across."""
    x1, y1, x2, y2 = box
    corners = np.array([[x1, y1, 1], [x2, y1, 1], [x2, y2, 1], [x1, y2, 1]])
    projected = (H @ corners.T).T
    projected = projected[:, :2] / projected[:, 2:3]
    return [
        float(projected[:, 0].min()), float(projected[:, 1].min()),
        float(projected[:, 0].max()), float(projected[:, 1].max()),
    ]


def calibrate_from_boxes(thermal_boxes: list[list[float]], rgb_boxes: list[list[float]],
                          effective_from: str, effective_until: str | None = None,
                          stems: list[str] | None = None, source_stem: str = "",
                          inlier_iou: float = 0.15, profile_id: str | None = None,
                          label: str = "") -> tuple[np.ndarray, list[dict], str]:
    """Fits an affine transform from matched bounding boxes (one drawn around the animal
    in each spectrum per capture) instead of exact point clicks. A box's center and size
    both constrain the fit, which tolerates imprecise drawing far better than single-pixel
    clicking on a blurry, low-resolution thermal blob. Affine (rather than a similarity
    transform) allows independent x/y scale, which the on-device thermal resize pipeline
    makes necessary — see `_affine_transform`. Needs at least 3 box pairs (6 corner
    points, comfortably above the 6-DOF minimum of 3 points); more, and spread across
    several images, is better."""
    if len(thermal_boxes) != len(rgb_boxes):
        raise ValueError('thermal_boxes and rgb_boxes must be the same length')
    if len(thermal_boxes) < 3:
        raise ValueError('Need at least 3 box pairs to fit an affine transform')

    ref_size = None
    if stems:
        thermal_boxes, ref_size = _normalize_coords_to_reference(thermal_boxes, stems)
        rgb_boxes, _ = _normalize_coords_to_reference(rgb_boxes, stems)

    def _normalize(box):
        x1, y1, x2, y2 = box
        return [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]

    thermal_boxes = [_normalize(b) for b in thermal_boxes]
    rgb_boxes = [_normalize(b) for b in rgb_boxes]

    # Each box contributes its top-left and bottom-right corners as point correspondences —
    # doubling the constraints per box pair without adding UI complexity.
    thermal_corners = np.array([[b[0], b[1]] for b in thermal_boxes] +
                                [[b[2], b[3]] for b in thermal_boxes], dtype=np.float64)
    rgb_corners = np.array([[b[0], b[1]] for b in rgb_boxes] +
                            [[b[2], b[3]] for b in rgb_boxes], dtype=np.float64)

    homography = _affine_transform(thermal_corners, rgb_corners)

    boxes_detail = []
    for i in range(len(thermal_boxes)):
        transformed = _transform_box(thermal_boxes[i], homography)
        iou = _box_iou(transformed, rgb_boxes[i])
        boxes_detail.append({
            'stem': stems[i] if stems else None,
            'thermal': thermal_boxes[i],
            'rgb': rgb_boxes[i],
            'transformed_thermal': transformed,
            'iou': round(iou, 3),
            'inlier': bool(iou >= inlier_iou),
        })

    inliers = sum(1 for b in boxes_detail if b['inlier'])
    logging.info(f'Affine transform solved from {len(thermal_boxes)} box pairs '
                 f'({inliers} above {inlier_iou} IoU)')

    profile = _upsert_profile({
        'homography': homography.tolist(),
        'point_count': len(thermal_boxes),
        'inlier_count': inliers,
        'boxes': boxes_detail,
        'method': 'manual_boxes',
        'source_stem': source_stem,
        'ref_size': list(ref_size) if ref_size else None,
    }, effective_from, effective_until, profile_id=profile_id, label=label)

    return homography, boxes_detail, profile['id']


def _capture_timestamp(stem: str) -> str | None:
    """The capture-time ISO timestamp recorded in a thermal sidecar JSON, or None if the
    sidecar is missing/unreadable. This is what a capture's applicable calibration profile
    is looked up by — not upload time or file mtime, since a batch re-upload or a delayed
    sync would otherwise misattribute captures to whichever profile happens to be current
    at upload time rather than the one that was actually in effect when the photo was taken."""
    sidecar = THERMAL_DIR / f'{stem}_thermal.json'
    if not sidecar.exists():
        return None
    try:
        return json.loads(sidecar.read_text()).get('timestamp') or None
    except (json.JSONDecodeError, OSError):
        return None


def align_thermal(thermal_png_path: str, homography: np.ndarray,
                   ref_size: tuple[int, int] | None = None) -> np.ndarray:
    """Warp a single thermal PNG into RGB pixel space. Does not touch the original file.

    `ref_size` is the (width, height) the homography was actually calibrated against. If
    this image is a different size — different device/firmware capture resolution — the
    homography's translation terms are scaled to match, on the assumption the aspect ratio
    is unchanged (see `_normalize_coords_to_reference`); an affine/similarity transform's
    linear part is scale-covariant so it doesn't need adjusting, only the translation."""
    img = cv2.imread(str(thermal_png_path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f'Could not read image: {thermal_png_path}')
    h, w = img.shape[:2]

    H = homography
    if ref_size and (w, h) != tuple(ref_size):
        ref_w, ref_h = ref_size
        kx, ky = w / ref_w, h / ref_h
        if abs(kx - ky) / ((kx + ky) / 2) > 0.02:
            logging.warning(
                f'{thermal_png_path}: {w}x{h} has a different aspect ratio than the '
                f'{ref_w}x{ref_h} calibration reference — alignment will be off'
            )
        H = homography.copy()
        H[:2, 2] *= (kx + ky) / 2

    return cv2.warpPerspective(img, H, (w, h))


def align_all(thermal_dir: Path = THERMAL_DIR, force: bool = False,
               profile_id: str | None = None) -> int:
    """Align every *_thermal.png in thermal_dir that doesn't already have an aligned
    counterpart, writing *_thermal_aligned.png alongside the original. Each capture is
    aligned with whichever calibration profile's [effective_from, effective_until) window
    contains *its own* capture timestamp — not necessarily the newest profile — so a camera
    move that starts a new profile only ever affects the frames actually taken after the
    move; frames from before it keep using the old profile they were correctly aligned
    with. A capture with no covering profile (a gap, or nothing calibrated yet for that
    period) is left unaligned.

    If `profile_id` is given, only re-aligns captures whose applicable profile is that one
    — used after editing an existing profile, so the refit only touches the frames it
    actually governs rather than every frame on disk regardless of which profile applies to
    it. Captures whose applicable profile is a *different* one are left untouched either way.

    Returns the number of files aligned."""
    profiles = load_profiles()
    count = 0
    for src in sorted(thermal_dir.glob('*_thermal.png')):
        stem = src.name.removesuffix('_thermal.png')
        timestamp = _capture_timestamp(stem)
        if timestamp is None:
            continue
        profile = find_profile_for_timestamp(timestamp, profiles=profiles)
        if profile is None:
            continue
        if profile_id is not None and profile['id'] != profile_id:
            continue

        out_path = thermal_dir / f'{stem}_thermal_aligned.png'
        if out_path.exists() and not force:
            continue
        homography = np.array(profile['homography'], dtype=np.float64)
        aligned = align_thermal(src, homography, ref_size=profile.get('ref_size'))
        cv2.imwrite(str(out_path), aligned)
        logging.info(f'Aligned {src.name} -> {out_path.name} (profile {profile["id"]})')
        count += 1
    return count


_CORRUPTED_ROW_STREAK_THRESHOLD = 0.6


def is_thermal_frame_corrupted(thermal_png_path: str) -> bool:
    """Flags thermal frames that are SPI/CRC read garbage rather than real sensor data —
    a known failure mode on this hardware (see the Pi client's thermal_common.py). A
    corrupted read shows strong, high-frequency row-to-row banding across the *entire*
    frame, regardless of content; a real (if extremely blurry, upsampled-from-80x62)
    frame doesn't, since even heavy interpolation keeps adjacent rows close to each other.
    Measured on real data: clean frames score ~0.1-0.3, corrupted ones ~1.0-1.3."""
    img = cv2.imread(str(thermal_png_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return True
    row_means = img.astype(np.float32).mean(axis=1)
    return float(np.std(np.diff(row_means))) >= _CORRUPTED_ROW_STREAK_THRESHOLD


def _list_capture_pairs(skip_corrupted: bool = True) -> list[tuple[str, Path, Path]]:
    """All (stem, rgb_path, thermal_path) with both files present on disk, excluding
    corrupted thermal reads by default."""
    pairs = []
    skipped = 0
    for sidecar in sorted(THERMAL_DIR.glob('*_thermal.json')):
        stem = sidecar.name.removesuffix('_thermal.json')
        meta = json.loads(sidecar.read_text())
        source_image = meta.get('source_image', '')
        rgb_path = UPLOAD_DIR / source_image if source_image else None
        thermal_path = THERMAL_DIR / f'{stem}_thermal.png'
        if not (rgb_path and rgb_path.exists() and thermal_path.exists()):
            continue
        if skip_corrupted and is_thermal_frame_corrupted(thermal_path):
            skipped += 1
            continue
        pairs.append((stem, rgb_path, thermal_path))
    if skipped:
        logging.info(f'Skipped {skipped} capture(s) with a corrupted thermal read')
    return pairs


def _dominant_resolution_pairs(pairs: list[tuple[str, Path, Path]]) -> list[tuple[str, Path, Path]]:
    """Keeps only the pairs matching whichever (rgb_shape, thermal_shape) combo is most
    common. Different device/firmware versions over time can produce different frame
    sizes, and a per-pixel background reference needs a consistent shape to stack against."""
    from collections import defaultdict
    groups: dict[tuple, list] = defaultdict(list)
    for stem, rgb_path, thermal_path in pairs:
        rgb_img = cv2.imread(str(rgb_path), cv2.IMREAD_GRAYSCALE)
        thermal_img = cv2.imread(str(thermal_path), cv2.IMREAD_GRAYSCALE)
        if rgb_img is None or thermal_img is None:
            continue
        groups[(rgb_img.shape, thermal_img.shape)].append((stem, rgb_path, thermal_path))

    if not groups:
        return []
    best_key = max(groups, key=lambda k: len(groups[k]))
    dropped = sum(len(v) for k, v in groups.items() if k != best_key)
    if dropped:
        logging.info(f'Dropped {dropped} capture(s) with a different resolution than the dominant '
                     f'{len(groups[best_key])}-frame group ({len(groups)} distinct resolutions seen)')
    return groups[best_key]


def _background_stats(paths: list[Path], sample_size: int) -> tuple[np.ndarray, np.ndarray]:
    """Per-pixel median and standard deviation across a sample of grayscale frames. The
    median is the 'no animal' reference; the std map captures how noisy/variable each pixel
    normally is (sensor noise, flickering lighting, cage bars with inconsistent brightness)
    so a real foreground object can be told apart from a pixel that's simply always noisy."""
    if len(paths) > sample_size:
        idx = np.linspace(0, len(paths) - 1, sample_size).astype(int)
        paths = [paths[i] for i in idx]
    stack = np.stack([cv2.imread(str(p), cv2.IMREAD_GRAYSCALE).astype(np.float32) for p in paths])
    return np.median(stack, axis=0), np.std(stack, axis=0)


_BLOB_Z_THRESH = 2.5
_BLOB_MIN_CIRCULARITY = 0.25


def _blob_centroid(gray: np.ndarray, background: np.ndarray, background_std: np.ndarray,
                    min_area_frac: float, max_area_frac: float,
                    blur_ksize: int = 9) -> tuple[float, float] | None:
    """Finds the centroid of the largest foreground blob, or None if nothing blob-shaped
    and plausibly-sized stands out. Thresholds each pixel's deviation from the background
    against that pixel's own normal variability (z-score) rather than a single Otsu cut on
    raw brightness difference — plain Otsu-on-absdiff falls apart on this data: thermal
    frames carry substantial per-pixel sensor noise that Otsu happily calls 'foreground'
    everywhere, and the RGB background subtraction alone can't distinguish a genuinely
    unusual pixel from one that's simply in a high-variance area (cage bars, flickering
    light). A circularity filter on top rejects the elongated non-blob regions (bars,
    diagonal shadow edges) that a compact resting/moving animal doesn't produce."""
    diff = np.abs(gray.astype(np.float32) - background)
    diff_blur = cv2.GaussianBlur(diff, (blur_ksize, blur_ksize), 0)
    std_blur = cv2.GaussianBlur(background_std, (blur_ksize, blur_ksize), 0)
    z = diff_blur / (std_blur + 3.0)
    mask = (z >= _BLOB_Z_THRESH).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
    if num_labels <= 1:
        return None

    total_px = gray.shape[0] * gray.shape[1]
    best_label, best_area = None, 0
    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]
        if not (min_area_frac * total_px <= area <= max_area_frac * total_px):
            continue
        contours, _ = cv2.findContours((labels == label).astype(np.uint8),
                                        cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        perimeter = cv2.arcLength(contours[0], True) if contours else 0
        circularity = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0
        if circularity < _BLOB_MIN_CIRCULARITY:
            continue
        if area > best_area:
            best_label, best_area = label, area

    if best_label is None:
        return None
    cx, cy = centroids[best_label]
    return float(cx), float(cy)


_MIN_INLIER_RATIO = 0.3


def auto_calibrate(effective_from: str, effective_until: str | None = None, min_pairs: int = 8,
                    sample_size: int = 80, min_area_frac: float = 0.0001, max_area_frac: float = 0.05,
                    profile_id: str | None = None, label: str = "") -> tuple[np.ndarray, list[dict], dict, str]:
    """Automatically extracts thermal<->RGB correspondences with no manual clicking:
    for every capture, finds the centroid of whatever stands out most from a median
    background in each spectrum independently, and keeps the frame only if both sides
    found one plausible, similarly-sized blob. Fixed camera rig means the same
    background-subtraction approach works for both RGB and thermal. Feeds the pooled
    centroids into the same RANSAC homography fit as manual calibration."""
    all_pairs = _list_capture_pairs()
    if len(all_pairs) < min_pairs:
        raise RuntimeError(f'Only {len(all_pairs)} capture pairs on disk, need at least {min_pairs}')

    pairs = _dominant_resolution_pairs(all_pairs)
    if len(pairs) < min_pairs:
        raise RuntimeError(
            f'Only {len(pairs)} capture pairs share a common resolution (out of {len(all_pairs)} total), '
            f'need at least {min_pairs}'
        )

    rgb_bg, rgb_std = _background_stats([p[1] for p in pairs], sample_size)
    thermal_bg, thermal_std = _background_stats([p[2] for p in pairs], sample_size)

    thermal_points, rgb_points, stems = [], [], []
    for stem, rgb_path, thermal_path in pairs:
        rgb_gray = cv2.imread(str(rgb_path), cv2.IMREAD_GRAYSCALE)
        thermal_gray = cv2.imread(str(thermal_path), cv2.IMREAD_GRAYSCALE)
        if rgb_gray is None or thermal_gray is None:
            continue

        rgb_centroid = _blob_centroid(rgb_gray, rgb_bg, rgb_std, min_area_frac, max_area_frac, blur_ksize=15)
        thermal_centroid = _blob_centroid(thermal_gray, thermal_bg, thermal_std, min_area_frac, max_area_frac)
        if rgb_centroid is None or thermal_centroid is None:
            continue

        rgb_points.append(rgb_centroid)
        thermal_points.append(thermal_centroid)
        stems.append(stem)

    diagnostics = {'pairs_total': len(all_pairs), 'pairs_considered': len(pairs), 'pairs_matched': len(stems)}
    if len(stems) < min_pairs:
        raise RuntimeError(
            f'Only found {len(stems)} usable frames (both sides showed a clear blob) '
            f'out of {len(pairs)} considered; need at least {min_pairs}'
        )

    thermal_pts = np.array(thermal_points, dtype=np.float64)
    rgb_pts = np.array(rgb_points, dtype=np.float64)

    # Fit an affine transform (see _affine_transform: this fixed rig has no real perspective
    # distortion, only independent x/y scale) via cv2's built-in RANSAC, rather than a full
    # projective homography. Centroid matches from independent background-subtraction in each
    # spectrum are frequently wrong (mismatched blobs, shadows, noise), and a projective fit
    # through that many outliers can produce a near-singular perspective row that blows up
    # warpPerspective into a radial starburst rather than just a bad-but-bounded misalignment.
    affine, mask = cv2.estimateAffine2D(thermal_pts, rgb_pts, method=cv2.RANSAC,
                                         ransacReprojThreshold=8.0)
    if affine is None:
        raise RuntimeError('Affine estimation failed')
    homography = np.vstack([affine, [0.0, 0.0, 1.0]])

    inlier_ratio = mask.ravel().sum() / len(mask)
    if inlier_ratio < _MIN_INLIER_RATIO:
        raise RuntimeError(
            f'Only {int(mask.ravel().sum())}/{len(mask)} correspondences ({inlier_ratio:.0%}) agreed on a '
            f'fit — auto-calibration data is too noisy to trust. Try manual point/box calibration instead.'
        )

    ref_h, ref_w = thermal_gray.shape[:2]
    points = _score_points(homography, thermal_pts, rgb_pts, mask, stems=stems)
    profile = _upsert_profile({
        'homography': homography.tolist(),
        'point_count': len(thermal_pts),
        'inlier_count': sum(1 for p in points if p['inlier']),
        'points': points,
        'method': 'auto_background_subtraction',
        'source_stem': f'auto:{len(stems)}_frames',
        'ref_size': [ref_w, ref_h],
    }, effective_from, effective_until, profile_id=profile_id, label=label)
    return homography, points, diagnostics, profile['id']


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Calibrate and apply thermal-to-RGB alignment')
    sub = parser.add_subparsers(dest='command', required=True)

    p_cal = sub.add_parser('calibrate', help='Compute homography from a checkerboard photo pair')
    p_cal.add_argument('rgb_image', help='Path to the RGB photo of the checkerboard')
    p_cal.add_argument('thermal_image', help='Path to the thermal photo of the checkerboard')
    p_cal.add_argument('--pattern-size', default='9x6', help='Inner corners WxH, e.g. 9x6')
    p_cal.add_argument('--effective-from', required=True,
                        help='ISO timestamp (e.g. 2026-08-01T00:00:00Z) this calibration profile applies from')
    p_cal.add_argument('--effective-until', default=None,
                        help='ISO timestamp this profile applies until (default: open-ended)')
    p_cal.add_argument('--profile-id', default=None, help='Edit an existing profile in place instead of creating a new one')
    p_cal.add_argument('--label', default='', help='Optional human-readable label for this profile')

    p_align = sub.add_parser('align', help="Apply each capture's applicable calibration profile")
    p_align.add_argument('path', nargs='?', default=str(THERMAL_DIR),
                          help='A *_thermal.png file, or a directory to batch-process (default: thermal/)')
    p_align.add_argument('--force', action='store_true', help='Re-align even if an aligned file already exists')
    p_align.add_argument('--profile-id', default=None,
                          help='Only re-align captures whose applicable profile is this one')

    p_profiles = sub.add_parser('profiles', help='List saved calibration profiles')

    p_auto = sub.add_parser('auto-calibrate',
                             help='Extract correspondences automatically via background subtraction, no manual clicking')
    p_auto.add_argument('--effective-from', required=True,
                         help='ISO timestamp this calibration profile applies from')
    p_auto.add_argument('--effective-until', default=None, help='ISO timestamp this profile applies until (default: open-ended)')
    p_auto.add_argument('--profile-id', default=None, help='Edit an existing profile in place instead of creating a new one')
    p_auto.add_argument('--label', default='', help='Optional human-readable label for this profile')
    p_auto.add_argument('--min-pairs', type=int, default=8)
    p_auto.add_argument('--sample-size', type=int, default=80, help='Frames sampled to build the background reference')
    p_auto.add_argument('--min-area-frac', type=float, default=0.0001, help='Min blob size, as a fraction of frame area')
    p_auto.add_argument('--max-area-frac', type=float, default=0.05, help='Max blob size, as a fraction of frame area')

    args = parser.parse_args()

    if args.command == 'calibrate':
        w, h = (int(x) for x in args.pattern_size.lower().split('x'))
        calibrate(args.rgb_image, args.thermal_image, args.effective_from,
                  effective_until=args.effective_until, pattern_size=(w, h),
                  profile_id=args.profile_id, label=args.label)

    elif args.command == 'auto-calibrate':
        homography, points, diagnostics, profile_id = auto_calibrate(
            args.effective_from, effective_until=args.effective_until,
            min_pairs=args.min_pairs, sample_size=args.sample_size,
            min_area_frac=args.min_area_frac, max_area_frac=args.max_area_frac,
            profile_id=args.profile_id, label=args.label,
        )
        inliers = sum(1 for p in points if p['inlier'])
        logging.info(f"{diagnostics['pairs_matched']}/{diagnostics['pairs_considered']} frames matched, "
                     f"{inliers}/{len(points)} inliers")

    elif args.command == 'profiles':
        for p in load_profiles():
            logging.info(f"{p['id']}: {p['effective_from']} .. {p.get('effective_until') or 'open-ended'} "
                         f"[{p.get('method')}] {p.get('inlier_count')}/{p.get('point_count')} inliers"
                         f"{' — ' + p['label'] if p.get('label') else ''}")

    elif args.command == 'align':
        path = Path(args.path)
        if not path.exists():
            raise SystemExit(f'Path not found: {path}')

        if path.is_dir():
            count = align_all(path, force=args.force, profile_id=args.profile_id)
            logging.info(f'Aligned {count} file(s)')
        else:
            stem = path.name.removesuffix('_thermal.png')
            timestamp = _capture_timestamp(stem)
            profile = find_profile_for_timestamp(timestamp) if timestamp else None
            if profile is None:
                raise SystemExit(f'No calibration profile covers {stem} (capture timestamp: {timestamp})')
            homography = np.array(profile['homography'], dtype=np.float64)
            out_path = path.parent / f'{stem}_thermal_aligned.png'
            aligned = align_thermal(path, homography, ref_size=profile.get('ref_size'))
            cv2.imwrite(str(out_path), aligned)
            logging.info(f'Aligned -> {out_path} (profile {profile["id"]})')


if __name__ == '__main__':
    main()
