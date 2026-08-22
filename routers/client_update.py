import base64
import hashlib
import logging
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import PlainTextResponse

from config import CLIENT_UPDATE_DIR

router = APIRouter(prefix="/client-update")

# Bundles are served base64-encoded. The devices fetch over a 2G modem's AT+HTTPREAD, which
# hands the payload back across a serial link as text; raw tar.gz bytes would have to survive
# that path intact, and a single mangled byte fails the whole transfer with no cheap way to tell
# where. Base64 costs 33% — on a ~39KB bundle at the ~1.8KB/s this link delivers, about 7
# seconds — to make the payload immune to the transport.
_MANIFEST_CACHE: dict[str, tuple[float, dict]] = {}


def _current_bundle() -> Path | None:
    """Newest .tgz in CLIENT_UPDATE_DIR, or None if no update has been published."""
    if not CLIENT_UPDATE_DIR.exists():
        return None
    bundles = sorted(CLIENT_UPDATE_DIR.glob("*.tgz"), key=lambda p: p.stat().st_mtime, reverse=True)
    return bundles[0] if bundles else None


def _manifest_for(path: Path) -> dict:
    """Version, size and digest of a bundle. Cached on (path, mtime) since hashing a bundle on
    every device poll would re-read it needlessly — the fleet polls far more often than the
    bundle changes."""
    key = str(path)
    mtime = path.stat().st_mtime
    cached = _MANIFEST_CACHE.get(key)
    if cached and cached[0] == mtime:
        return cached[1]

    raw = path.read_bytes()
    encoded = base64.b64encode(raw)
    manifest = {
        "version": path.stem,
        # Digest of the *decoded* bundle, so a device verifies what it will actually install
        # rather than the wire representation of it.
        "sha256": hashlib.sha256(raw).hexdigest(),
        "size": len(raw),
        "encoded_size": len(encoded),
    }
    _MANIFEST_CACHE[key] = (mtime, manifest)
    return manifest


@router.get("/manifest")
async def manifest(device_id: str = ""):
    """What the currently published client bundle is, so a device can decide whether it already
    has it. Deliberately tiny: this is polled far more often than a bundle is downloaded, and
    every byte crosses a 2G link."""
    bundle = _current_bundle()
    if bundle is None:
        return {"version": None}
    data = _manifest_for(bundle)
    if device_id:
        logging.info(f"Update manifest served to {device_id}: {data['version']}")
    return data


@router.get("/bundle", response_class=PlainTextResponse)
async def bundle(device_id: str = ""):
    """The published bundle, base64-encoded. 404 when nothing is published."""
    path = _current_bundle()
    if path is None:
        raise HTTPException(status_code=404, detail="No client update published")
    data = _manifest_for(path)
    logging.info(
        f"Serving client bundle {data['version']} "
        f"({data['encoded_size']} b64 bytes) to {device_id or 'unknown device'}"
    )
    return base64.b64encode(path.read_bytes()).decode("ascii")
