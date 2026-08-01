import asyncio
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel

from config import THERMAL_DIR, UPLOAD_DIR
from thermal_align import (align_all, auto_calibrate, calibrate_from_boxes, calibrate_from_points,
                            find_profile_for_timestamp, is_thermal_frame_corrupted, load_profiles,
                            save_profiles)

router = APIRouter(prefix="/thermal/calibrate")

STATIC_DIR = Path(__file__).parent.parent / "static"


@router.get("", response_class=HTMLResponse)
async def calibrate_ui():
    return FileResponse(str(STATIC_DIR / "thermal_calibrate.html"))


@router.get("/candidates")
async def candidates(limit: int = 200):
    sidecars = sorted(THERMAL_DIR.glob("*_thermal.json"), reverse=True)[:limit]
    items = []
    for sidecar in sidecars:
        meta = json.loads(sidecar.read_text())
        stem = sidecar.name.removesuffix("_thermal.json")
        source_image = meta.get("source_image", "")
        if not source_image:
            continue
        if not (UPLOAD_DIR / source_image).exists():
            continue
        thermal_png = THERMAL_DIR / f"{stem}_thermal.png"
        if not thermal_png.exists():
            continue
        if is_thermal_frame_corrupted(thermal_png):
            continue
        items.append({
            "stem": stem,
            "source_image": source_image,
            "timestamp": meta.get("timestamp", ""),
            "device_id": meta.get("device_id", ""),
        })
    return items


@router.get("/profiles")
async def list_profiles():
    """All saved calibration profiles, newest effective_from first — each with its
    [effective_from, effective_until) window, method, and fit summary. A capture is aligned
    using whichever profile's window contains its own timestamp (see
    thermal_align.find_profile_for_timestamp), not necessarily the most recent profile."""
    profiles = sorted(load_profiles(), key=lambda p: p["effective_from"], reverse=True)
    return [{
        "id": p["id"],
        "label": p.get("label", ""),
        "effective_from": p["effective_from"],
        "effective_until": p.get("effective_until"),
        "calibrated_at": p.get("calibrated_at"),
        "method": p.get("method"),
        "point_count": p.get("point_count"),
        "inlier_count": p.get("inlier_count"),
    } for p in profiles]


@router.delete("/profiles/{profile_id}")
async def delete_profile(profile_id: str):
    profiles = load_profiles()
    remaining = [p for p in profiles if p["id"] != profile_id]
    if len(remaining) == len(profiles):
        raise HTTPException(status_code=404, detail=f"No profile with id {profile_id!r}")
    save_profiles(remaining)
    return {"status": "ok", "deleted": profile_id}


class Pair(BaseModel):
    stem: str
    rgb: list[float]
    thermal: list[float]


class CalibrationSubmission(BaseModel):
    pairs: list[Pair]
    effective_from: str
    effective_until: str | None = None
    profile_id: str | None = None
    label: str = ""
    preview_stem: str | None = None


@router.post("")
async def submit_calibration(body: CalibrationSubmission):
    if len(body.pairs) < 4:
        raise HTTPException(status_code=400, detail="Need at least 4 point pairs")

    thermal_points = [p.thermal for p in body.pairs]
    rgb_points = [p.rgb for p in body.pairs]
    point_stems = [p.stem for p in body.pairs]
    stems = sorted(set(point_stems))

    try:
        _, points, profile_id = calibrate_from_points(
            thermal_points, rgb_points, body.effective_from, effective_until=body.effective_until,
            stems=point_stems, source_stem=",".join(stems), profile_id=body.profile_id, label=body.label,
        )
    except (ValueError, RuntimeError) as e:
        raise HTTPException(status_code=400, detail=str(e))

    aligned_count = align_all(force=True, profile_id=profile_id)
    logging.info(
        f"Calibrated profile {profile_id} from {len(body.pairs)} points across {len(stems)} image(s); "
        f"re-aligned {aligned_count} file(s)"
    )

    preview_stem = body.preview_stem or stems[0]
    preview = None
    if (THERMAL_DIR / f"{preview_stem}_thermal_aligned.png").exists():
        preview = f"/thermal/image/{preview_stem}_thermal_aligned.png"

    return {
        "status": "ok",
        "profile_id": profile_id,
        "point_count": len(body.pairs),
        "image_count": len(stems),
        "aligned_count": aligned_count,
        "preview": preview,
        "points": sorted(points, key=lambda p: -p["error_px"]),
    }


class BoxPair(BaseModel):
    stem: str
    rgb: list[float]
    thermal: list[float]


class BoxCalibrationSubmission(BaseModel):
    pairs: list[BoxPair]
    effective_from: str
    effective_until: str | None = None
    profile_id: str | None = None
    label: str = ""
    preview_stem: str | None = None


@router.post("/boxes")
async def submit_box_calibration(body: BoxCalibrationSubmission):
    if len(body.pairs) < 3:
        raise HTTPException(status_code=400, detail="Need at least 3 box pairs")

    thermal_boxes = [p.thermal for p in body.pairs]
    rgb_boxes = [p.rgb for p in body.pairs]
    point_stems = [p.stem for p in body.pairs]
    stems = sorted(set(point_stems))

    try:
        _, boxes, profile_id = calibrate_from_boxes(
            thermal_boxes, rgb_boxes, body.effective_from, effective_until=body.effective_until,
            stems=point_stems, source_stem=",".join(stems), profile_id=body.profile_id, label=body.label,
        )
    except (ValueError, RuntimeError) as e:
        raise HTTPException(status_code=400, detail=str(e))

    aligned_count = align_all(force=True, profile_id=profile_id)
    logging.info(
        f"Calibrated profile {profile_id} from {len(body.pairs)} boxes across {len(stems)} image(s); "
        f"re-aligned {aligned_count} file(s)"
    )

    preview_stem = body.preview_stem or stems[0]
    preview = None
    if (THERMAL_DIR / f"{preview_stem}_thermal_aligned.png").exists():
        preview = f"/thermal/image/{preview_stem}_thermal_aligned.png"

    return {
        "status": "ok",
        "profile_id": profile_id,
        "point_count": len(body.pairs),
        "image_count": len(stems),
        "aligned_count": aligned_count,
        "preview": preview,
        "boxes": sorted(boxes, key=lambda b: b["iou"]),
    }


class AutoCalibrationSubmission(BaseModel):
    effective_from: str
    effective_until: str | None = None
    profile_id: str | None = None
    label: str = ""


@router.post("/auto")
async def auto_submit_calibration(body: AutoCalibrationSubmission):
    try:
        _, points, diagnostics, profile_id = await asyncio.to_thread(
            auto_calibrate, body.effective_from, effective_until=body.effective_until,
            profile_id=body.profile_id, label=body.label,
        )
    except (ValueError, RuntimeError) as e:
        raise HTTPException(status_code=400, detail=str(e))

    aligned_count = align_all(force=True, profile_id=profile_id)
    logging.info(
        f"Auto-calibrated profile {profile_id} from {diagnostics['pairs_matched']}/{diagnostics['pairs_considered']} "
        f"frames; re-aligned {aligned_count} file(s)"
    )

    return {
        "status": "ok",
        "profile_id": profile_id,
        "point_count": len(points),
        "pairs_considered": diagnostics["pairs_considered"],
        "pairs_matched": diagnostics["pairs_matched"],
        "aligned_count": aligned_count,
        "points": sorted(points, key=lambda p: -p["error_px"]),
    }


@router.get("/stats")
async def calibration_stats(profile_id: str | None = None, at: str | None = None):
    """Fit summary for one calibration profile. Defaults to whichever profile covers `at`
    (an ISO timestamp, default now) if `profile_id` isn't given explicitly — i.e. "the
    profile that would be used for a capture right now." Use `GET /thermal/calibrate/profiles`
    to see all of them."""
    profiles = load_profiles()
    if profile_id:
        data = next((p for p in profiles if p["id"] == profile_id), None)
        if data is None:
            raise HTTPException(status_code=404, detail=f"No profile with id {profile_id!r}")
    else:
        ts = at or datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
        data = find_profile_for_timestamp(ts, profiles=profiles)
        if data is None:
            raise HTTPException(status_code=404, detail="No calibration profile covers this time")

    base = {
        "id": data.get("id"),
        "label": data.get("label", ""),
        "effective_from": data.get("effective_from"),
        "effective_until": data.get("effective_until"),
        "calibrated_at": data.get("calibrated_at"),
        "method": data.get("method"),
        "point_count": data.get("point_count"),
        "inlier_count": data.get("inlier_count"),
    }

    if "boxes" in data:
        boxes = data["boxes"]
        inliers = [b for b in boxes if b["inlier"]]
        return {
            **base,
            "kind": "boxes",
            "mean_inlier_iou": round(sum(b["iou"] for b in inliers) / len(inliers), 3) if inliers else None,
            "min_inlier_iou": min((b["iou"] for b in inliers), default=None),
            "boxes": sorted(boxes, key=lambda b: b["iou"]),
        }

    points = data.get("points", [])
    inliers = [p for p in points if p["inlier"]]
    return {
        **base,
        "kind": "points",
        "mean_inlier_error_px": round(sum(p["error_px"] for p in inliers) / len(inliers), 2) if inliers else None,
        "max_inlier_error_px": max((p["error_px"] for p in inliers), default=None),
        "points": sorted(points, key=lambda p: -p["error_px"]),
    }
