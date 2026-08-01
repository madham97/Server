import json

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, HTMLResponse

from config import THERMAL_DIR, UPLOAD_DIR

router = APIRouter(prefix="/thermal")

_PAGE_HEAD = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Thermal Pairs</title>
<style>
  body { font-family: monospace; max-width: 1400px; margin: 40px auto; padding: 0 20px; background: #111; color: #eee; }
  h1 { color: #7cf; margin-bottom: 4px; }
  p.sub { color: #888; margin-top: 4px; margin-bottom: 32px; }
  .grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(560px, 1fr)); gap: 20px; }
  .card { background: #1a1a1a; border: 1px solid #333; border-radius: 6px; padding: 12px; }
  .pair { display: flex; gap: 8px; }
  .pair > div { flex: 1; min-width: 0; }
  .pair img { width: 100%; height: 160px; object-fit: cover; border-radius: 4px; background: #000; display: block; }
  .pair p { margin: 4px 0 0; color: #888; font-size: 0.75em; text-align: center; }
  .missing { display: flex; align-items: center; justify-content: center; height: 160px; color: #f66; font-size: 0.7em; text-align: center; padding: 0 6px; border: 1px dashed #444; border-radius: 4px; }
  .meta { margin-top: 10px; font-size: 0.85em; color: #aaa; }
  .meta .stem { color: #7cf; word-break: break-all; }
  .meta .temps { color: #afa; }
  .empty { color: #888; margin-top: 40px; }
</style>
</head>
<body>
<h1>Thermal Pairs</h1>
<p class="sub">RGB / thermal / aligned-thermal captures split from RGBA uploads, newest first. &middot; <a href="/thermal/calibrate" style="color:#7cf;">calibrate alignment</a></p>
"""

_PAGE_TAIL = """
</body>
</html>"""


@router.get("", response_class=HTMLResponse)
async def thermal_view(limit: int = 100):
    sidecars = sorted(THERMAL_DIR.glob("*_thermal.json"), reverse=True)[:limit]

    if not sidecars:
        return _PAGE_HEAD + '<p class="empty">No thermal captures yet.</p>' + _PAGE_TAIL

    cards = []
    for sidecar in sidecars:
        meta = json.loads(sidecar.read_text())
        stem = sidecar.name.removesuffix("_thermal.json")
        source_image = meta.get("source_image", "")
        thermal_png = f"{stem}_thermal.png"
        aligned_png = f"{stem}_thermal_aligned.png"
        has_rgb = (UPLOAD_DIR / source_image).exists() if source_image else False
        has_aligned = (THERMAL_DIR / aligned_png).exists()
        rgb_side = (
            f'<img src="/annotate/image/{source_image}" loading="lazy"><p>rgb</p>'
            if has_rgb else '<div class="missing">rgb missing</div><p>rgb</p>'
        )
        aligned_side = (
            f'<img src="/thermal/image/{aligned_png}" loading="lazy"><p>aligned</p>'
            if has_aligned else '<div class="missing">not aligned yet</div><p>aligned</p>'
        )
        cards.append(f"""
<div class="card">
  <div class="pair">
    <div>{rgb_side}</div>
    <div><img src="/thermal/image/{thermal_png}" loading="lazy"><p>thermal</p></div>
    <div>{aligned_side}</div>
  </div>
  <div class="meta">
    <div class="stem">{stem}</div>
    <div>{meta.get('device_id', '')} &middot; {meta.get('timestamp', '')}</div>
    <div class="temps">{meta.get('thermal_min_c', '?')}&ndash;{meta.get('thermal_max_c', '?')}&deg;C (avg {meta.get('thermal_avg_c', '?')}&deg;C)</div>
  </div>
</div>""")

    return _PAGE_HEAD + f'<div class="grid">{"".join(cards)}</div>' + _PAGE_TAIL


@router.get("/image/{name}")
async def thermal_image(name: str):
    path = (THERMAL_DIR / name).resolve()
    if not path.is_relative_to(THERMAL_DIR.resolve()) or not path.exists():
        raise HTTPException(status_code=404, detail="Thermal image not found")
    return FileResponse(str(path), media_type="image/png")
