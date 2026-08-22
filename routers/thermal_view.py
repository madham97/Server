import json
import re
from datetime import date, datetime, timezone
from urllib.parse import urlencode

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, HTMLResponse

from config import THERMAL_DIR, UPLOAD_DIR

router = APIRouter(prefix="/thermal")

DEFAULT_PER_PAGE = 24
MAX_PER_PAGE = 200

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
  form.filters .count { margin-left: auto; font-size: 0.8em; color: #888; padding-bottom: 6px; }
  form.filters .count b { color: #afa; }

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
</style>
</head>
<body>
<h1>Thermal Pairs</h1>
<p class="sub">RGB / thermal / aligned-thermal captures split from RGBA uploads, newest first. Click any image to open the overlay. &middot; <a href="/thermal/calibrate" style="color:#7cf;">calibrate alignment</a></p>
"""

_LIGHTBOX = """
<div id="lightbox">
  <button id="lbClose">&times; close</button>
  <div id="lbMeta"></div>
  <div id="lbWrap">
    <img id="lbBase" src="" alt="">
    <img id="lbTop" class="top" src="" alt="">
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

function showCard(i) {
  if (i < 0 || i >= cards.length) return;
  lbIndex = i;
  const d = cards[i].dataset;
  const hasAligned = d.aligned !== '';
  if (!hasAligned) useAligned = false;
  const overlaySrc = useAligned && hasAligned ? d.aligned : d.thermal;

  lbBase.src = d.rgb || d.thermal;
  lbTop.src = overlaySrc;
  lbTop.style.display = d.rgb ? '' : 'none';   // no RGB to overlay onto: show the thermal alone
  lbTop.style.opacity = lbOpacity.value / 100;
  lbMissing.style.display = hasAligned ? 'none' : 'block';
  lbSwap.disabled = !hasAligned || !d.rgb;
  lbSwap.textContent = useAligned ? 'swap: raw thermal' : 'swap: aligned';
  lbLayer.textContent = d.rgb ? (useAligned && hasAligned ? 'aligned thermal over rgb' : 'raw thermal over rgb') : 'thermal only';
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

window.addEventListener('keydown', (e) => {
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


def _page_url(page: int, start: str | None, end: str | None, per_page: int) -> str:
    params = {"page": page}
    if start:
        params["start"] = start
    if end:
        params["end"] = end
    if per_page != DEFAULT_PER_PAGE:
        params["per_page"] = per_page
    return "/thermal?" + urlencode(params)


def _render_pager(page: int, pages: int, start: str | None, end: str | None, per_page: int) -> str:
    if pages <= 1:
        return ""

    def link(p: int, text: str | None = None) -> str:
        label = text or str(p)
        if p == page and text is None:
            return f'<span class="cur">{label}</span>'
        return f'<a href="{_page_url(p, start, end, per_page)}">{label}</a>'

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
    page: int = 1,
    per_page: int = DEFAULT_PER_PAGE,
    limit: int | None = None,
):
    """Browse capture pairs, newest first. `start`/`end` are inclusive UTC capture dates
    (YYYY-MM-DD); `limit` is the pre-pagination alias kept for older links and just sets
    `per_page`."""
    if limit is not None:
        per_page = limit
    per_page = max(1, min(per_page, MAX_PER_PAGE))
    page = max(1, page)
    start_date, end_date = _parse_date(start), _parse_date(end)

    sidecars = sorted(THERMAL_DIR.glob("*_thermal.json"), reverse=True)
    if start_date or end_date:
        filtered = []
        for sidecar in sidecars:
            captured = _stem_datetime(sidecar.name)
            if captured is None:
                continue  # unparseable name: can't place it on the timeline, so a date filter excludes it
            if start_date and captured.date() < start_date:
                continue
            if end_date and captured.date() > end_date:
                continue
            filtered.append(sidecar)
        sidecars = filtered

    total = len(sidecars)
    pages = max(1, (total + per_page - 1) // per_page)
    page = min(page, pages)
    window = sidecars[(page - 1) * per_page: page * per_page]

    filters = f"""
<form class="filters" method="get" action="/thermal">
  <label>from (UTC date)<input type="date" name="start" value="{start or ''}"></label>
  <label>to (UTC date)<input type="date" name="end" value="{end or ''}"></label>
  <label>per page<select name="per_page">
    {"".join(f'<option value="{n}"{" selected" if n == per_page else ""}>{n}</option>' for n in (12, 24, 48, 96, 200))}
  </select></label>
  <button type="submit">Apply</button>
  <a class="clear" href="/thermal">clear filters</a>
  <span class="count"><b>{total}</b> capture(s) match{" (all captures)" if not (start_date or end_date) else ""}</span>
</form>"""

    if not window:
        body = '<p class="empty">No thermal captures match this date range.</p>' if total == 0 and (start_date or end_date) \
            else '<p class="empty">No thermal captures yet.</p>'
        return _PAGE_HEAD + filters + body + _PAGE_TAIL

    pager = _render_pager(page, pages, start, end, per_page)

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
        aligned_url = f"/thermal/image/{aligned_png}" if has_aligned else ""
        temps = (f"{meta.get('thermal_min_c', '?')}&ndash;{meta.get('thermal_max_c', '?')}&deg;C "
                 f"(avg {meta.get('thermal_avg_c', '?')}&deg;C)")
        rgb_side = (
            f'<img src="{rgb_url}" loading="lazy"><p>rgb</p>'
            if has_rgb else '<div class="missing">rgb missing</div><p>rgb</p>'
        )
        aligned_side = (
            f'<img src="{aligned_url}" loading="lazy"><p>aligned</p>'
            if has_aligned else '<div class="missing">not aligned yet</div><p>aligned</p>'
        )
        cards.append(f"""
<div class="card" data-stem="{stem}" data-rgb="{rgb_url}" data-thermal="/thermal/image/{thermal_png}"
     data-aligned="{aligned_url}"
     data-info="{meta.get('device_id', '')} &middot; {meta.get('timestamp', '')} &middot; {temps}">
  <div class="pair">
    <div>{rgb_side}</div>
    <div><img src="/thermal/image/{thermal_png}" loading="lazy"><p>thermal</p></div>
    <div>{aligned_side}</div>
  </div>
  <div class="meta">
    <div class="stem">{stem}</div>
    <div>{meta.get('device_id', '')} &middot; {meta.get('timestamp', '')}</div>
    <div class="temps">{temps}</div>
  </div>
</div>""")

    return (_PAGE_HEAD + filters + pager + f'<div class="grid">{"".join(cards)}</div>'
            + pager + _LIGHTBOX + _PAGE_TAIL)


@router.get("/image/{name}")
async def thermal_image(name: str):
    path = (THERMAL_DIR / name).resolve()
    if not path.is_relative_to(THERMAL_DIR.resolve()) or not path.exists():
        raise HTTPException(status_code=404, detail="Thermal image not found")
    return FileResponse(str(path), media_type="image/png")
