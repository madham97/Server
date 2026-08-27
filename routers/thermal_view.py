import hashlib
import json
import re
from datetime import date, datetime, timezone
from pathlib import Path
from urllib.parse import urlencode

import cv2
import numpy as np
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, Response

from config import THERMAL_DIR, UPLOAD_DIR
from thermal_align import (SENSOR_FRAME_SIZE, capture_timestamp, find_profile_for_timestamp,
                            scale_to_reference)
from thermal_color import RENDER_VERSION, colorize

router = APIRouter(prefix="/thermal")

DEFAULT_PER_PAGE = 24
MAX_PER_PAGE = 200

# A rendered frame is cached for a year and only ever replaced by a new URL, so every input that
# can change the picture has to be in that URL: see _image_version.
_IMMUTABLE_CACHE = "public, max-age=31536000, immutable"
# Someone linking /thermal/image/... by hand gets no version to pin, so their copy has to be
# allowed to go stale for minutes rather than for a day.
_UNVERSIONED_CACHE = "public, max-age=300"


def _render_inputs(name: str) -> list[Path]:
    """Every stored file the render of `name` reads.

    An aligned frame depends on its source frame as well as itself: the source sets the colour
    scale and the footprint mask, so re-aligning after a recalibration and repairing a source
    frame must both change the URL."""
    paths = [THERMAL_DIR / name]
    if name.endswith("_thermal_aligned.png"):
        paths.append(THERMAL_DIR / f"{_stem_of(name)}_thermal.png")
    return paths


def _image_version(name: str) -> str:
    """The cache key for a rendered thermal frame: the renderer's fingerprint over the stored
    bytes it reads. Changing the colour pipeline or rewriting a frame on disk both produce a new
    token, and nothing else does."""
    digest = hashlib.blake2b(digest_size=8)
    digest.update(RENDER_VERSION.encode())
    for path in _render_inputs(name):
        try:
            st = path.stat()
            digest.update(f"{path.name}:{st.st_mtime_ns}:{st.st_size}".encode())
        except OSError:
            digest.update(f"{path.name}:missing".encode())
    return digest.hexdigest()


def image_url(name: str) -> str:
    """A thermal image URL carrying the version of the render it points at."""
    return f"/thermal/image/{name}?v={_image_version(name)}"


# Sidecars are named "<prefix>_YYYYMMDDTHHMMSSZ_thermal.json" by the uploader, so the capture
# date is readable straight off the filename — date filtering never has to open 10k JSON files.
_STEM_TS_RE = re.compile(r"(\d{8})T(\d{6})Z")


def _stem_datetime(stem: str) -> datetime | None:
    m = _STEM_TS_RE.search(stem)
    if not m:
        return None
    try:
        return datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def _parse_date(value: str | None) -> date | None:
    if not value:
        return None
    try:
        return date.fromisoformat(value)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid date {value!r}, expected YYYY-MM-DD")


def _parse_hour(value: int | None, field: str) -> int | None:
    if value is None:
        return None
    if not 0 <= value <= 23:
        raise HTTPException(status_code=400, detail=f"Invalid {field} {value!r}, expected an hour 0-23")
    return value


def _hour_in_window(hour: int, start: int, end: int) -> bool:
    """Inclusive at both ends, and wraps through midnight when start > end — 18→06 means the
    night, not the empty set. Wrapping is the point of this filter: the animals are nocturnal,
    so the interesting window always straddles 00:00 UTC."""
    if start <= end:
        return start <= hour <= end
    return hour >= start or hour <= end


_PAGE_HEAD = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Thermal Pairs</title>
<style>
  body { font-family: monospace; max-width: 1400px; margin: 40px auto; padding: 0 20px; background: #111; color: #eee; }
  h1 { color: #7cf; margin-bottom: 4px; }
  p.sub { color: #888; margin-top: 4px; margin-bottom: 20px; }
  .grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(560px, 1fr)); gap: 20px; }
  .card { background: #1a1a1a; border: 1px solid #333; border-radius: 6px; padding: 12px; }
  .pair { display: flex; gap: 8px; }
  .pair > div { flex: 1; min-width: 0; }
  .pair img { width: 100%; height: 160px; object-fit: cover; border-radius: 4px; background: #000; display: block; cursor: zoom-in; }
  .pair p { margin: 4px 0 0; color: #888; font-size: 0.75em; text-align: center; }
  .missing { display: flex; align-items: center; justify-content: center; height: 160px; color: #f66; font-size: 0.7em; text-align: center; padding: 0 6px; border: 1px dashed #444; border-radius: 4px; }
  .meta { margin-top: 10px; font-size: 0.85em; color: #aaa; }
  .meta .stem { color: #7cf; word-break: break-all; }
  .meta .temps { color: #afa; }
  .empty { color: #888; margin-top: 40px; }

  form.filters { display: flex; align-items: flex-end; gap: 12px; flex-wrap: wrap; background: #1a1a1a; border: 1px solid #333; border-radius: 6px; padding: 12px 14px; margin-bottom: 16px; }
  form.filters label { display: flex; flex-direction: column; gap: 4px; font-size: 0.8em; color: #888; }
  form.filters input, form.filters select, form.filters button, .pager a, .pager span.cur {
    background: #2a2a2a; border: 1px solid #555; color: #eee; border-radius: 4px;
    padding: 5px 9px; font-size: 0.85em; font-family: monospace;
  }
  form.filters button { cursor: pointer; border-color: #4af; background: #245; }
  form.filters a.clear { color: #888; font-size: 0.8em; padding-bottom: 6px; }
  form.filters .count { margin-left: auto; font-size: 0.8em; color: #888; padding-bottom: 6px; text-align: right; }
  form.filters .count b { color: #afa; }
  form.filters .wrap { color: #fd6; }

  .pager { display: flex; align-items: center; gap: 8px; flex-wrap: wrap; margin: 20px 0; }
  .pager a { text-decoration: none; cursor: pointer; }
  .pager a:hover { background: #3a3a3a; }
  .pager span.cur { border-color: #4af; color: #7cf; }
  .pager span.gap, .pager span.info { color: #666; font-size: 0.85em; }
  .pager span.disabled { opacity: 0.35; border: 1px solid #444; border-radius: 4px; padding: 5px 9px; font-size: 0.85em; }

  #lightbox { position: fixed; inset: 0; background: rgba(0, 0, 0, 0.9); z-index: 50; display: none; flex-direction: column; align-items: center; justify-content: center; padding: 24px; gap: 10px; }
  #lightbox.open { display: flex; }
  #lbWrap { position: relative; background: #000; border-radius: 6px; overflow: hidden; max-width: min(1100px, 92vw); max-height: 72vh; line-height: 0; }
  #lbWrap img { display: block; max-width: min(1100px, 92vw); max-height: 72vh; width: auto; }
  #lbWrap img.top { position: absolute; inset: 0; width: 100%; height: 100%; opacity: 0.7; }
  #lbMissing { display: none; position: absolute; left: 0; right: 0; bottom: 0; background: rgba(0,0,0,0.75); color: #f88; font-size: 0.75em; line-height: 1.4; text-align: center; padding: 6px; }
  #lbControls { display: flex; align-items: center; gap: 10px; flex-wrap: wrap; color: #888; font-size: 0.8em; max-width: 92vw; }
  #lbControls input[type=range] { width: 220px; }
  #lbControls button, #lbClose { background: #2a2a2a; border: 1px solid #555; color: #eee; border-radius: 4px; padding: 5px 10px; font-family: monospace; font-size: 0.9em; cursor: pointer; }
  #lbControls button:disabled { opacity: 0.35; cursor: not-allowed; }
  #lbMeta { color: #aaa; font-size: 0.8em; text-align: center; max-width: 92vw; word-break: break-all; }
  #lbMeta .stem { color: #7cf; }
  #lbClose { position: absolute; top: 16px; right: 20px; }
  #lbLayer { color: #7cf; }

  /* Native view: the browser upscales an 80x62 frame, and `pixelated` stops it smoothing the
     result back into the interpolation we just removed. One sensor cell = one visible block. */
  body.native .pair img.thermal, body.native #lbWrap img.thermal { image-rendering: pixelated; }
  #nativeBar { display: flex; align-items: center; gap: 10px; flex-wrap: wrap; background: #1a1a1a;
    border: 1px solid #333; border-radius: 6px; padding: 8px 14px; margin-bottom: 16px; font-size: 0.8em; color: #888; }
  #nativeBar label { display: flex; align-items: center; gap: 6px; color: #eee; cursor: pointer; }
  #nativeBar .note { color: #666; }
  body.native #nativeBar { border-color: #4af; }
  body.native #nativeBar .note b { color: #fd6; }
</style>
</head>
<body>
<h1>Thermal Pairs</h1>
<p class="sub">RGB / thermal / aligned-thermal captures split from RGBA uploads, newest first. Click any image to open the overlay. &middot; <a href="/thermal/calibrate" style="color:#7cf;">calibrate alignment</a></p>

<div id="nativeBar">
  <label><input type="checkbox" id="nativeToggle"> show thermal at true sensor resolution</label>
  <span class="note">The sensor is <b>80&times;62</b> &mdash; 4,960 measurements. Stored frames were
  interpolated up to the visible frame's size on the device, which adds no information (recovering
  the sensor grid costs ~0.036&deg;C). This shows what was actually measured.</span>
</div>
"""

_LIGHTBOX = """
<div id="lightbox">
  <button id="lbClose">&times; close</button>
  <div id="lbMeta"></div>
  <div id="lbWrap">
    <img id="lbBase" src="" alt="">
    <img id="lbTop" class="top thermal" src="" alt="">
    <div id="lbMissing">Not aligned yet &mdash; this capture arrived after the last calibration ran.</div>
  </div>
  <div id="lbControls">
    <button id="lbPrev">&larr; prev</button>
    <span id="lbLayer"></span>
    <label>overlay opacity <input type="range" id="lbOpacity" min="0" max="100" value="70"></label>
    <span id="lbPct">70%</span>
    <button id="lbSwap">swap: raw thermal</button>
    <button id="lbNext">next &rarr;</button>
  </div>
  <div style="color:#666; font-size:0.75em;">&larr;/&rarr; to move &middot; &uarr;/&darr; for opacity &middot; s to swap layer &middot; esc to close</div>
</div>
<script>
// Each card carries its own image URLs; the lightbox is a single reusable overlay that
// walks the cards rendered on this page (pagination bounds the walk to one page).
const cards = [...document.querySelectorAll('.card')];
const lb = document.getElementById('lightbox');
const lbBase = document.getElementById('lbBase');
const lbTop = document.getElementById('lbTop');
const lbMissing = document.getElementById('lbMissing');
const lbOpacity = document.getElementById('lbOpacity');
const lbPct = document.getElementById('lbPct');
const lbSwap = document.getElementById('lbSwap');
const lbLayer = document.getElementById('lbLayer');
let lbIndex = -1;
let useAligned = true;   // overlay layer: aligned thermal (default) or the raw thermal frame

// Native view is a display choice, not a different capture: the same URL with ?native=1 returns
// the frame reduced to the sensor's grid. Kept in localStorage so it survives paging.
let nativeView = localStorage.getItem('thermalNative') === '1';
const nativeToggle = document.getElementById('nativeToggle');

function nativeSrc(url) {
  if (!url) return url;
  return nativeView ? url + (url.includes('?') ? '&' : '?') + 'native=1' : url;
}

function applyNativeView() {
  document.body.classList.toggle('native', nativeView);
  nativeToggle.checked = nativeView;
  cards.forEach(card => {
    const d = card.dataset;
    const thermal = card.querySelector('img.thermal[data-role=thermal]');
    const aligned = card.querySelector('img.thermal[data-role=aligned]');
    if (thermal) thermal.src = nativeSrc(d.thermal);
    if (aligned && d.aligned) aligned.src = nativeSrc(d.aligned);
  });
  if (lbIndex >= 0) showCard(lbIndex);
}

nativeToggle.addEventListener('change', () => {
  nativeView = nativeToggle.checked;
  localStorage.setItem('thermalNative', nativeView ? '1' : '0');
  applyNativeView();
});

function showCard(i) {
  if (i < 0 || i >= cards.length) return;
  lbIndex = i;
  const d = cards[i].dataset;
  const hasAligned = d.aligned !== '';
  if (!hasAligned) useAligned = false;
  const overlaySrc = useAligned && hasAligned ? d.aligned : d.thermal;

  // With no RGB to overlay onto, the thermal frame *is* the base, so it takes the native
  // treatment instead of the overlay layer.
  lbBase.src = d.rgb ? d.rgb : nativeSrc(d.thermal);
  lbBase.classList.toggle('thermal', !d.rgb);
  lbTop.src = nativeSrc(overlaySrc);
  lbTop.style.display = d.rgb ? '' : 'none';   // no RGB to overlay onto: show the thermal alone
  lbTop.style.opacity = lbOpacity.value / 100;
  lbMissing.style.display = hasAligned ? 'none' : 'block';
  lbSwap.disabled = !hasAligned || !d.rgb;
  lbSwap.textContent = useAligned ? 'swap: raw thermal' : 'swap: aligned';
  const layer = d.rgb ? (useAligned && hasAligned ? 'aligned thermal over rgb' : 'raw thermal over rgb') : 'thermal only';
  lbLayer.textContent = layer + (nativeView ? ' · 80×62 native' : '');
  document.getElementById('lbMeta').innerHTML =
    `<span class="stem">${d.stem}</span><br>${d.info} &middot; ${i + 1} of ${cards.length} on this page`;
  document.getElementById('lbPrev').disabled = i === 0;
  document.getElementById('lbNext').disabled = i === cards.length - 1;
  lb.classList.add('open');
}

function closeLb() { lb.classList.remove('open'); lbIndex = -1; }

cards.forEach((card, i) => {
  card.querySelectorAll('img').forEach(img => img.addEventListener('click', () => {
    useAligned = true;
    showCard(i);
  }));
});

document.getElementById('lbClose').addEventListener('click', closeLb);
lb.addEventListener('click', (e) => { if (e.target === lb) closeLb(); });
document.getElementById('lbPrev').addEventListener('click', () => showCard(lbIndex - 1));
document.getElementById('lbNext').addEventListener('click', () => showCard(lbIndex + 1));
lbSwap.addEventListener('click', () => { useAligned = !useAligned; showCard(lbIndex); });
lbOpacity.addEventListener('input', () => {
  lbTop.style.opacity = lbOpacity.value / 100;
  lbPct.textContent = lbOpacity.value + '%';
});

applyNativeView();

window.addEventListener('keydown', (e) => {
  // 'n' works whether or not the lightbox is open — it's a page-wide display mode.
  if (e.key === 'n' || e.key === 'N') {
    nativeToggle.checked = !nativeToggle.checked;
    nativeToggle.dispatchEvent(new Event('change'));
    return;
  }
  if (!lb.classList.contains('open')) return;
  if (e.key === 'Escape') closeLb();
  else if (e.key === 'ArrowLeft') showCard(lbIndex - 1);
  else if (e.key === 'ArrowRight') showCard(lbIndex + 1);
  else if (e.key === 'ArrowUp' || e.key === 'ArrowDown') {
    const step = e.key === 'ArrowUp' ? 5 : -5;
    lbOpacity.value = Math.max(0, Math.min(100, Number(lbOpacity.value) + step));
    lbOpacity.dispatchEvent(new Event('input'));
  } else if (e.key === 's' || e.key === 'S') {
    if (!lbSwap.disabled) { useAligned = !useAligned; showCard(lbIndex); }
  } else return;
  e.preventDefault();
});
</script>
"""

_PAGE_TAIL = """
</body>
</html>"""


def _page_url(page: int, filters: dict, per_page: int) -> str:
    params = {"page": page}
    params.update({k: v for k, v in filters.items() if v not in (None, "")})
    if per_page != DEFAULT_PER_PAGE:
        params["per_page"] = per_page
    return "/thermal?" + urlencode(params)


def _render_pager(page: int, pages: int, filters: dict, per_page: int) -> str:
    if pages <= 1:
        return ""

    def link(p: int, text: str | None = None) -> str:
        label = text or str(p)
        if p == page and text is None:
            return f'<span class="cur">{label}</span>'
        return f'<a href="{_page_url(p, filters, per_page)}">{label}</a>'

    # First, last, and a window around the current page — 400+ page links is unusable.
    window = {1, pages, page}
    window.update(p for p in range(page - 2, page + 3) if 1 <= p <= pages)
    numbers, previous = [], 0
    for p in sorted(window):
        if p - previous > 1:
            numbers.append('<span class="gap">&hellip;</span>')
        numbers.append(link(p))
        previous = p

    prev_el = link(page - 1, "&larr; prev") if page > 1 else '<span class="disabled">&larr; prev</span>'
    next_el = link(page + 1, "next &rarr;") if page < pages else '<span class="disabled">next &rarr;</span>'
    return (
        f'<div class="pager">{prev_el}{"".join(numbers)}{next_el}'
        f'<span class="info">page {page} of {pages}</span></div>'
    )


@router.get("", response_class=HTMLResponse)
async def thermal_view(
    start: str | None = None,
    end: str | None = None,
    hour_start: int | None = None,
    hour_end: int | None = None,
    page: int = 1,
    per_page: int = DEFAULT_PER_PAGE,
    limit: int | None = None,
):
    """Browse capture pairs, newest first. `start`/`end` are inclusive UTC capture dates
    (YYYY-MM-DD) and `hour_start`/`hour_end` an inclusive UTC hour-of-day window that may wrap
    through midnight; `limit` is the pre-pagination alias kept for older links and just sets
    `per_page`."""
    if limit is not None:
        per_page = limit
    per_page = max(1, min(per_page, MAX_PER_PAGE))
    page = max(1, page)
    start_date, end_date = _parse_date(start), _parse_date(end)
    hour_start = _parse_hour(hour_start, "hour_start")
    hour_end = _parse_hour(hour_end, "hour_end")
    # One bound on its own still defines a window — the other end is just the edge of the day.
    if hour_start is not None or hour_end is not None:
        hour_from = 0 if hour_start is None else hour_start
        hour_to = 23 if hour_end is None else hour_end
    else:
        hour_from = hour_to = None

    filters = {"start": start, "end": end, "hour_start": hour_start, "hour_end": hour_end}

    sidecars = sorted(THERMAL_DIR.glob("*_thermal.json"), reverse=True)
    if start_date or end_date or hour_from is not None:
        filtered = []
        for sidecar in sidecars:
            captured = _stem_datetime(sidecar.name)
            if captured is None:
                continue  # unparseable name: can't place it on the timeline, so a filter excludes it
            if start_date and captured.date() < start_date:
                continue
            if end_date and captured.date() > end_date:
                continue
            if hour_from is not None and not _hour_in_window(captured.hour, hour_from, hour_to):
                continue
            filtered.append(sidecar)
        sidecars = filtered

    total = len(sidecars)
    pages = max(1, (total + per_page - 1) // per_page)
    page = min(page, pages)
    window = sidecars[(page - 1) * per_page: page * per_page]

    def hour_options(selected: int | None) -> str:
        opts = [f'<option value=""{"" if selected is not None else " selected"}>any</option>']
        opts += [f'<option value="{h}"{" selected" if h == selected else ""}>{h:02d}:00</option>'
                 for h in range(24)]
        return "".join(opts)

    any_filter = bool(start_date or end_date or hour_from is not None)
    wrapped = hour_from is not None and hour_from > hour_to
    filter_bar = f"""
<form class="filters" method="get" action="/thermal">
  <label>from (UTC date)<input type="date" name="start" value="{start or ''}"></label>
  <label>to (UTC date)<input type="date" name="end" value="{end or ''}"></label>
  <label>hour from (UTC)<select name="hour_start">{hour_options(hour_start)}</select></label>
  <label>hour to (UTC)<select name="hour_end">{hour_options(hour_end)}</select></label>
  <label>per page<select name="per_page">
    {"".join(f'<option value="{n}"{" selected" if n == per_page else ""}>{n}</option>' for n in (12, 24, 48, 96, 200))}
  </select></label>
  <button type="submit">Apply</button>
  <a class="clear" href="/thermal">clear filters</a>
  <span class="count"><b>{total}</b> capture(s) match{"" if any_filter else " (all captures)"}
    {'<br><span class="wrap">hour window wraps through midnight</span>' if wrapped else ''}</span>
</form>"""

    if not window:
        body = ('<p class="empty">No thermal captures match these filters.</p>' if any_filter
                else '<p class="empty">No thermal captures yet.</p>')
        return _PAGE_HEAD + filter_bar + body + _PAGE_TAIL

    pager = _render_pager(page, pages, filters, per_page)

    cards = []
    for sidecar in window:
        meta = json.loads(sidecar.read_text())
        stem = sidecar.name.removesuffix("_thermal.json")
        source_image = meta.get("source_image", "")
        thermal_png = f"{stem}_thermal.png"
        aligned_png = f"{stem}_thermal_aligned.png"
        has_rgb = (UPLOAD_DIR / source_image).exists() if source_image else False
        has_aligned = (THERMAL_DIR / aligned_png).exists()
        rgb_url = f"/annotate/image/{source_image}" if has_rgb else ""
        thermal_url = image_url(thermal_png)
        aligned_url = image_url(aligned_png) if has_aligned else ""
        temps = (f"{meta.get('thermal_min_c', '?')}&ndash;{meta.get('thermal_max_c', '?')}&deg;C "
                 f"(avg {meta.get('thermal_avg_c', '?')}&deg;C)")
        rgb_side = (
            f'<img src="{rgb_url}" loading="lazy"><p>rgb</p>'
            if has_rgb else '<div class="missing">rgb missing</div><p>rgb</p>'
        )
        aligned_side = (
            f'<img class="thermal" data-role="aligned" src="{aligned_url}" loading="lazy"><p>aligned</p>'
            if has_aligned else '<div class="missing">not aligned yet</div><p>aligned</p>'
        )
        cards.append(f"""
<div class="card" data-stem="{stem}" data-rgb="{rgb_url}" data-thermal="{thermal_url}"
     data-aligned="{aligned_url}"
     data-info="{meta.get('device_id', '')} &middot; {meta.get('timestamp', '')} &middot; {temps}">
  <div class="pair">
    <div>{rgb_side}</div>
    <div><img class="thermal" data-role="thermal" src="{thermal_url}" loading="lazy"><p>thermal</p></div>
    <div>{aligned_side}</div>
  </div>
  <div class="meta">
    <div class="stem">{stem}</div>
    <div>{meta.get('device_id', '')} &middot; {meta.get('timestamp', '')}</div>
    <div class="temps">{temps}</div>
  </div>
</div>""")

    return (_PAGE_HEAD + filter_bar + pager + f'<div class="grid">{"".join(cards)}</div>'
            + pager + _LIGHTBOX + _PAGE_TAIL)


def _stem_of(name: str) -> str:
    """The capture stem a thermal filename belongs to, so its sidecar can be found."""
    for suffix in ("_thermal_aligned.png", "_thermal.png"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return Path(name).stem


def _aligned_homography(stem: str, src_shape: tuple[int, ...]) -> tuple[np.ndarray, tuple[int, int]] | None:
    """The transform `align_thermal` used for this capture, and the grid it lands on — the
    capture's own profile, not merely the newest one, so a frame from before a camera move keeps
    the geometry it was actually aligned with. None when no profile covers the capture."""
    timestamp = capture_timestamp(stem)
    profile = find_profile_for_timestamp(timestamp) if timestamp else None
    if profile is None:
        return None
    H = np.array(profile["homography"], dtype=np.float64)
    ref_w, ref_h = profile.get("ref_size") or (1920, 1080)
    h, w = src_shape[:2]
    if (w, h) != (ref_w, ref_h):
        H = H @ scale_to_reference(w, h, ref_w, ref_h)
    return H, (ref_w, ref_h)


def _aligned_footprint(H: np.ndarray, src_shape: tuple[int, ...], out_size: tuple[int, int]) -> np.ndarray:
    """Which output pixels an aligned frame actually holds a measurement for, taken from the warp
    geometry by sending an opaque source frame through the same transform.

    Not `pixel > 0`: the stored aligned frame is a *cubic* resample of the sensor grid against
    zero padding, so instead of stepping to zero at the quad edge it ramps down through a band
    about one sensor cell wide — 24 px at the 80x62 -> 1920x1080 scale. Every value in that ramp
    is nonzero, so the value test kept ~24,500 padding pixels per frame and fed them to the
    percentile clip, which is what flattened the aligned pane's colour scale."""
    h, w = src_shape[:2]
    opaque = np.full((h, w), 255, dtype=np.uint8)
    return cv2.warpPerspective(opaque, H, out_size, flags=cv2.INTER_NEAREST) > 0


def _stored_aligned_mask(name: str, out_shape: tuple[int, ...]) -> np.ndarray | None:
    """The footprint of a stored `*_thermal_aligned.png`, re-derived from its source frame and
    profile. None when either is gone, leaving `colorize` its `pixel > 0` fallback."""
    if not name.endswith("_thermal_aligned.png"):
        return None
    stem = _stem_of(name)
    source = cv2.imread(str(THERMAL_DIR / f"{stem}_thermal.png"), cv2.IMREAD_GRAYSCALE)
    if source is None:
        return None
    geometry = _aligned_homography(stem, source.shape)
    if geometry is None:
        return None
    H, _ = geometry
    out_h, out_w = out_shape[:2]
    return _aligned_footprint(H, source.shape, (out_w, out_h))


def _encode_png(img: np.ndarray, name: str, color: bool,
                mask: np.ndarray | None = None,
                headers: dict[str, str] | None = None) -> Response:
    """Encode a thermal frame for the browser, on the blue->red temperature scale unless the
    caller asked for the raw stored bytes. See thermal_color for why the raw form reads black."""
    if color:
        img = colorize(img, _stem_of(name), aligned=name.endswith("_thermal_aligned.png"),
                       mask=mask)
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise HTTPException(status_code=500, detail="Could not encode thermal frame")
    return Response(content=buf.tobytes(), media_type="image/png", headers=headers or {})


def _native_png(path: Path, name: str, color: bool = True,
                headers: dict[str, str] | None = None) -> Response:
    """The frame as the sensor actually sampled it, rather than as it was stored.

    Every capture before native transport was upsampled from 80x62 to the visible frame's size
    on the device, and that upsample added no information — reducing it back recovers the sensor
    samples to within ~0.26 gray levels (0.036 °C), measured over the archive. Serving the small
    image and letting the browser blow it up with `image-rendering: pixelated` shows the real
    measurement grid: ~4,960 samples, not the 2 million values a smooth 1920x1080 render implies.

    An *aligned* frame can't simply be reduced — warping moved the thermal content into RGB
    coordinates, so its rows and columns no longer line up with sensor cells. It is instead
    re-warped from the native grid with nearest-neighbour sampling, which shows exactly which
    RGB pixels each sensor cell covers."""
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise HTTPException(status_code=404, detail="Thermal image not readable")

    mask = None
    if name.endswith("_thermal_aligned.png"):
        stem = name.removesuffix("_thermal_aligned.png")
        raw = cv2.imread(str(THERMAL_DIR / f"{stem}_thermal.png"), cv2.IMREAD_GRAYSCALE)
        geometry = _aligned_homography(stem, SENSOR_FRAME_SIZE[::-1]) if raw is not None else None
        if geometry is None:
            # Nothing to re-derive it from; the stored aligned frame is all there is.
            return _encode_png(img, name, color, mask=_stored_aligned_mask(name, img.shape),
                               headers=headers)
        if (raw.shape[1], raw.shape[0]) != SENSOR_FRAME_SIZE:
            raw = cv2.resize(raw, SENSOR_FRAME_SIZE, interpolation=cv2.INTER_AREA)
        H, ref_size = geometry
        img = cv2.warpPerspective(raw, H, ref_size, flags=cv2.INTER_NEAREST)
        # Nearest sampling leaves no ramp, but a genuine zero-byte measurement would drop out of
        # a `pixel > 0` test; the geometric footprint keeps it.
        mask = _aligned_footprint(H, SENSOR_FRAME_SIZE[::-1], ref_size)
    elif (img.shape[1], img.shape[0]) != SENSOR_FRAME_SIZE:
        # INTER_AREA, not NEAREST: averaging each output cell over the block it came from
        # inverts the device's cubic upsample, where point-sampling would keep whichever
        # interpolated value happened to land on that pixel.
        img = cv2.resize(img, SENSOR_FRAME_SIZE, interpolation=cv2.INTER_AREA)

    return _encode_png(img, name, color, mask=mask, headers=headers)


@router.get("/image/{name}")
async def thermal_image(request: Request, name: str, native: bool = False,
                        color: bool = True, v: str | None = None):
    """Serve a thermal PNG. `native=1` returns it on the sensor's own 80x62 grid instead of the
    interpolated form it is stored in — see `_native_png`. `color=0` returns the raw stored
    bytes instead of the temperature-scaled render — see thermal_color.

    `v` is the render version the caller pinned (see `image_url`). It selects nothing: the render
    is always current, and the parameter exists so that changing the colour pipeline or rewriting
    a frame on disk moves every page's image URLs and retires the cached copies. A request that
    carries one can therefore be cached forever; one that does not gets minutes."""
    path = (THERMAL_DIR / name).resolve()
    if not path.is_relative_to(THERMAL_DIR.resolve()) or not path.exists():
        raise HTTPException(status_code=404, detail="Thermal image not found")

    # The variants are separate pictures of the same capture, so they need separate validators.
    etag = f'"{_image_version(name)}-{int(native)}{int(color)}"'
    cache = _IMMUTABLE_CACHE if v else _UNVERSIONED_CACHE
    if request.headers.get("if-none-match") == etag:
        return Response(status_code=304, headers={"ETag": etag, "Cache-Control": cache})
    headers = {"ETag": etag, "Cache-Control": cache}

    if native:
        return _native_png(path, name, color=color, headers=headers)
    if not color:
        # Untouched bytes off disk; FileResponse derives its own validator from the file.
        return FileResponse(str(path), media_type="image/png",
                            headers={"Cache-Control": cache})
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise HTTPException(status_code=404, detail="Thermal image not readable")
    return _encode_png(img, name, True, mask=_stored_aligned_mask(name, img.shape),
                       headers=headers)
