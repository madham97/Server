"""Rendering a thermal frame as a temperature image rather than as raw stored bytes.

Frames are stored as 0-255 encoded linearly across a *fixed* °C window, recorded per capture in
the JSON sidecar as thermal_norm_min_c/thermal_norm_max_c (10-45 °C for everything captured
before 2026-08-22, when the client gained a configurable window). A typical enclosure scene only
occupies about 21-26 °C of that window, so every pixel lands in the bottom fifth of the byte
range and the frame renders as near-black — the sensor is fine, the display mapping is not.

Decoding back to °C and rescaling to the range the frame actually occupies is what makes a warm
body separate from cool bedding. The percentile clip matters as much as the colour map: a single
hot outlier (an IR illuminator reflection, a sun-warmed surface) would otherwise compress the
whole animal back into the low end.

That range is a property of the *capture*, not of the view: it is always measured on the stored,
unwarped frame, and every render of that capture — raw, aligned, native — is mapped through it.
Letting a warped frame pick its own range is what made the aligned pane disagree with the thermal
pane beside it; see `colorize`.
"""

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from config import THERMAL_DIR

# The window every capture up to 2026-08-22 was encoded against — kept in sync with
# routers/upload.py, which writes these same defaults into new sidecars when the device does
# not report its own.
LEGACY_NORM_MIN_C = 10.0
LEGACY_NORM_MAX_C = 45.0

# Percentile clip applied to the frame's own valid pixels before mapping to colour. The low end
# is trimmed harder than the high end because we care about preserving the top of the range,
# where the animal is.
DEFAULT_LO_PCT = 1.0
DEFAULT_HI_PCT = 99.8

# Cold -> hot control points, as (position, R, G, B). Deep blue for background so cool bedding
# reads as one flat field, then magenta/red through the middle and near-white at the top, so a
# body separates from a merely warm surface instead of both saturating.
_STOPS = (
    (0.00, (8, 14, 60)),
    (0.16, (26, 46, 150)),
    (0.34, (60, 96, 210)),
    (0.50, (140, 92, 190)),
    (0.64, (208, 62, 128)),
    (0.78, (240, 66, 52)),
    (0.90, (252, 152, 32)),
    (1.00, (255, 244, 200)),
)


def _build_lut() -> np.ndarray:
    """A 256-entry BGR lookup table, in cv2's channel order so it can be indexed straight into
    an array that cv2.imencode will write."""
    xs = np.array([s[0] for s in _STOPS])
    cs = np.array([s[1] for s in _STOPS], dtype=np.float64)
    t = np.linspace(0.0, 1.0, 256)
    lut = np.empty((256, 3), dtype=np.uint8)
    for out_ch, rgb_ch in enumerate((2, 1, 0)):  # BGR out, RGB in
        lut[:, out_ch] = np.clip(np.interp(t, xs, cs[:, rgb_ch]), 0, 255).astype(np.uint8)
    return lut


LUT = _build_lut()


def _render_version() -> str:
    """A fingerprint of the code that decides what a thermal frame looks like.

    /thermal/image renders on every request, so changing a colour stop, a percentile, or the
    aligned-frame masking changes what a URL returns without changing the URL. The renders are
    cached hard in the browser, which otherwise goes on serving the old picture for as long as
    the max-age says — the reason the first pass at the aligned colour scale appeared fixed in
    a fresh tab and unfixed in one that had already loaded the page. Folding this into the cache
    key means a deploy invalidates exactly the frames whose appearance actually changed.
    """
    digest = hashlib.blake2b(digest_size=6)
    for path in (Path(__file__), Path(__file__).parent / "routers" / "thermal_view.py"):
        try:
            digest.update(path.read_bytes())
        except OSError:  # running from a layout where the router is elsewhere; version the rest
            digest.update(b"?")
    return digest.hexdigest()


RENDER_VERSION = _render_version()


def norm_window(stem: str) -> tuple[float, float]:
    """The °C window this capture's bytes were encoded against, from its sidecar.

    Falls back to the legacy defaults when the sidecar is missing or unparseable — the same
    assumption routers/upload.py records as thermal_norm_source='assumed_legacy_default'.
    """
    try:
        meta = json.loads((THERMAL_DIR / f"{stem}_thermal.json").read_text())
        return (float(meta["thermal_norm_min_c"]), float(meta["thermal_norm_max_c"]))
    except (OSError, ValueError, KeyError, TypeError):
        return (LEGACY_NORM_MIN_C, LEGACY_NORM_MAX_C)


def to_celsius(gray: np.ndarray, norm_min_c: float, norm_max_c: float) -> np.ndarray:
    return norm_min_c + (gray.astype(np.float32) / 255.0) * (norm_max_c - norm_min_c)


def _gray(img: np.ndarray) -> np.ndarray:
    return img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def _percentile_bounds(celsius: np.ndarray, lo_pct: float, hi_pct: float) -> tuple[float, float]:
    lo = float(np.percentile(celsius, lo_pct))
    hi = float(np.percentile(celsius, hi_pct))
    if hi - lo < 1e-3:
        hi = lo + 1.0
    return lo, hi


def scale_bounds(stem: str, *, frame: np.ndarray | None = None,
                 lo_pct: float = DEFAULT_LO_PCT,
                 hi_pct: float = DEFAULT_HI_PCT) -> tuple[float, float] | None:
    """The °C range `colorize` maps across for this capture, so a caller can label the scale.

    Measured on the capture's stored *unwarped* frame, whose every pixel is a real measurement.
    An aligned frame is not usable for this: `align_thermal` resamples the sensor grid cubically
    against zero padding, so its edge carries a ramp down to the padding and a ring of over- and
    undershoot, none of it measured. Those artefacts sit far outside the scene's real spread, and
    letting them set the percentiles stretched the aligned scale over roughly 13-29 °C where the
    frame itself only spans 25-27 °C — pushing the whole scene into the top of the colour map.

    Returns None when the unwarped frame is missing, leaving the caller to fall back.
    """
    if frame is None:
        frame = cv2.imread(str(THERMAL_DIR / f"{stem}_thermal.png"), cv2.IMREAD_GRAYSCALE)
    if frame is None:
        return None
    return _percentile_bounds(to_celsius(_gray(frame), *norm_window(stem)), lo_pct, hi_pct)


def colorize(gray: np.ndarray, stem: str, *, aligned: bool = False,
             mask: np.ndarray | None = None, bounds: tuple[float, float] | None = None,
             lo_pct: float = DEFAULT_LO_PCT, hi_pct: float = DEFAULT_HI_PCT) -> np.ndarray:
    """Map a stored grayscale thermal frame to a BGR image on a blue->red temperature scale.

    The scale comes from the capture's unwarped frame (see `scale_bounds`) unless `bounds` is
    given, so an aligned frame and the raw frame beside it put the same temperature at the same
    colour — which is the only thing that makes the two panes comparable.

    `aligned` marks a frame warped into RGB coordinates, whose padding is not measurement and is
    painted black rather than coloured. Pass `mask` to say which pixels are real; without one,
    padding is guessed at as the exact zeros, which understates it, because the cubic warp ramps
    the border down through nonzero values rather than stepping to zero.
    """
    gray = _gray(gray)
    if mask is None and aligned:
        mask = gray > 0

    celsius = to_celsius(gray, *norm_window(stem))
    if bounds is None:
        bounds = scale_bounds(stem, lo_pct=lo_pct, hi_pct=hi_pct)
    if bounds is None:
        # No unwarped frame to measure; fall back to this frame's own valid pixels.
        samples = celsius[mask] if mask is not None and mask.any() else celsius
        bounds = _percentile_bounds(samples, lo_pct, hi_pct)

    lo, hi = bounds
    scaled = np.clip((celsius - lo) / (hi - lo), 0.0, 1.0)
    out = LUT[(scaled * 255).astype(np.uint8)]
    if mask is not None:
        out[~mask] = 0
    return out
