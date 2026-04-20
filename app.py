import asyncio
import logging
import shutil
import time
from pathlib import Path
from datetime import datetime
from fastapi import FastAPI, UploadFile, File, Form, HTTPException

from inference import process_video, VIDEO_EXTS
from tracker import TrackerDB, parse_timestamp_from_filename

app = FastAPI()

UPLOAD_DIR = Path("uploads")
PROCESSED_DIR = Path("processed")
UPLOAD_DIR.mkdir(exist_ok=True)
PROCESSED_DIR.mkdir(exist_ok=True)

UPLOAD_LOG = Path("upload_log.txt")

POLL_INTERVAL = 5  # seconds to wait between polling for new files
DB_PATH = "objects.db"
MAX_GAP_SECONDS = 300  # max time between video chunks for track linking

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

# Global database connection
db: TrackerDB | None = None


from PIL import Image as PILImage
import io

@app.post("/upload")
async def upload(
    video: UploadFile = File(None),
    image: UploadFile = File(None),
    timestamp: str = Form(None),
    mode: str = Form(None),
    motion_score: str = Form(None),
    device_id: str = Form(None),
):
    file = video or image
    if file is None:
        raise HTTPException(status_code=422, detail="No file provided")

    data = await file.read()

    # Convert WebP to JPEG transparently
    if file.filename.lower().endswith('.webp'):
        img = PILImage.open(io.BytesIO(data))
        buf = io.BytesIO()
        img.save(buf, format='JPEG', quality=95)
        data = buf.getvalue()
        filename = Path(file.filename).stem + '.jpg'
    else:
        filename = file.filename

    file_path = UPLOAD_DIR / filename
    file_path.write_bytes(data)

    meta_parts = [p for p in [
        device_id,
        mode,
        f"motion={motion_score}" if motion_score else None,
        timestamp,
    ] if p]
    logging.info(f"Received upload: {filename}" +
                 (f" [{', '.join(meta_parts)}]" if meta_parts else ""))

    received_at = datetime.now(datetime.UTC).strftime('%Y-%m-%dT%H:%M:%SZ')
    with open(UPLOAD_LOG, 'a') as f:
        f.write(f"{received_at}\t{filename}\t{device_id or ''}\t{mode or ''}\t{motion_score or ''}\t{timestamp or ''}\n")

    return {"status": "ok"}



@app.get("/health")
async def health():
    return {"status": "ok"}


from fastapi.responses import HTMLResponse

@app.get("/config-help", response_class=HTMLResponse)
async def config_help():
    return """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>SMS Config Helper</title>
<style>
  body { font-family: monospace; max-width: 720px; margin: 40px auto; padding: 0 20px; background: #111; color: #eee; }
  h1 { color: #7cf; margin-bottom: 4px; }
  p.sub { color: #888; margin-top: 4px; margin-bottom: 32px; }
  h2 { color: #aaa; font-size: 0.85em; text-transform: uppercase; letter-spacing: 2px; margin: 32px 0 8px; }
  table { width: 100%; border-collapse: collapse; margin-bottom: 24px; }
  th { text-align: left; color: #888; font-size: 0.8em; padding: 4px 8px; border-bottom: 1px solid #333; }
  td { padding: 6px 8px; border-bottom: 1px solid #222; font-size: 0.9em; vertical-align: top; }
  td.key { color: #7cf; white-space: nowrap; }
  td.note { color: #f90; font-size: 0.8em; }
  select, input[type=text], input[type=number] { background: #222; color: #eee; border: 1px solid #444; padding: 8px; border-radius: 4px; font-family: monospace; font-size: 1em; }
  select { width: 100%; margin-bottom: 12px; }
  .row { display: flex; gap: 12px; margin-bottom: 12px; align-items: center; }
  .row label { color: #888; white-space: nowrap; }
  .row input { flex: 1; }
  #output { background: #1a1a1a; border: 1px solid #444; border-radius: 4px; padding: 16px; font-size: 1.1em; color: #afa; word-break: break-all; min-height: 48px; margin: 12px 0; }
  button { background: #7cf; color: #111; border: none; padding: 10px 24px; border-radius: 4px; font-family: monospace; font-size: 1em; cursor: pointer; font-weight: bold; }
  button:active { background: #5ad; }
  .copied { color: #afa; margin-left: 12px; display: none; }
</style>
</head>
<body>
<h1>SMS Config Helper</h1>
<p class="sub">Build a JSON patch to text to the device's SIM number. Changes apply within 60 seconds.</p>

<h2>Recording</h2>
<table>
  <tr><th>Key</th><th>Type</th><th>Example</th><th></th></tr>
  <tr><td class="key">recording.mode</td><td>string</td><td>image_motion, image_interval, segment, motion</td><td class="note">restart</td></tr>
  <tr><td class="key">recording.motion_threshold</td><td>float 0–1</td><td>0.015 = sensitive, 0.05 = less sensitive</td><td class="note">restart</td></tr>
  <tr><td class="key">recording.motion_cooldown</td><td>seconds</td><td>60</td><td class="note">restart</td></tr>
  <tr><td class="key">recording.detection_interval</td><td>seconds</td><td>1</td><td class="note">restart</td></tr>
  <tr><td class="key">recording.image_interval</td><td>seconds</td><td>30</td><td class="note">restart</td></tr>
  <tr><td class="key">recording.image_quality</td><td>int 1–100</td><td>75</td><td class="note">restart</td></tr>
  <tr><td class="key">recording.chunk_duration</td><td>seconds</td><td>30</td><td class="note">restart</td></tr>
  <tr><td class="key">recording.width</td><td>pixels</td><td>1280</td><td class="note">restart</td></tr>
  <tr><td class="key">recording.height</td><td>pixels</td><td>720</td><td class="note">restart</td></tr>
  <tr><td class="key">recording.framerate</td><td>fps</td><td>15</td><td class="note">restart</td></tr>
  <tr><td class="key">recording.motion_debug</td><td>bool</td><td>true / false</td><td class="note">restart</td></tr>
  <tr><td class="key">recording.enabled</td><td>bool</td><td>true / false</td><td class="note">restart</td></tr>
</table>

<h2>Uploader (live — no restart needed)</h2>
<table>
  <tr><th>Key</th><th>Type</th><th>Example</th><th></th></tr>
  <tr><td class="key">webp_quality</td><td>int 1–100</td><td>80</td><td></td></tr>
  <tr><td class="key">webp_compress</td><td>bool</td><td>true / false</td><td></td></tr>
  <tr><td class="key">poll_interval</td><td>seconds</td><td>10</td><td></td></tr>
  <tr><td class="key">max_retries</td><td>int</td><td>3</td><td></td></tr>
  <tr><td class="key">retry_delay</td><td>seconds</td><td>10</td><td></td></tr>
</table>

<h2>Build your SMS</h2>
<select id="keySelect" onchange="onKeyChange()">
  <optgroup label="Recording (requires restart)">
    <option value='{"recording":{"mode":"VALUE"}}' data-type="string" data-placeholder="image_motion">recording.mode</option>
    <option value='{"recording":{"motion_threshold":VALUE}}' data-type="float" data-placeholder="0.015">recording.motion_threshold</option>
    <option value='{"recording":{"motion_cooldown":VALUE}}' data-type="float" data-placeholder="60">recording.motion_cooldown</option>
    <option value='{"recording":{"detection_interval":VALUE}}' data-type="float" data-placeholder="1">recording.detection_interval</option>
    <option value='{"recording":{"image_interval":VALUE}}' data-type="float" data-placeholder="30">recording.image_interval</option>
    <option value='{"recording":{"image_quality":VALUE}}' data-type="int" data-placeholder="75">recording.image_quality</option>
    <option value='{"recording":{"chunk_duration":VALUE}}' data-type="int" data-placeholder="30">recording.chunk_duration</option>
    <option value='{"recording":{"width":VALUE}}' data-type="int" data-placeholder="1280">recording.width</option>
    <option value='{"recording":{"height":VALUE}}' data-type="int" data-placeholder="720">recording.height</option>
    <option value='{"recording":{"framerate":VALUE}}' data-type="int" data-placeholder="15">recording.framerate</option>
    <option value='{"recording":{"motion_debug":VALUE}}' data-type="bool" data-placeholder="false">recording.motion_debug</option>
    <option value='{"recording":{"enabled":VALUE}}' data-type="bool" data-placeholder="true">recording.enabled</option>
  </optgroup>
  <optgroup label="Uploader (live)">
    <option value='{"webp_quality":VALUE}' data-type="int" data-placeholder="80">webp_quality</option>
    <option value='{"webp_compress":VALUE}' data-type="bool" data-placeholder="true">webp_compress</option>
    <option value='{"poll_interval":VALUE}' data-type="int" data-placeholder="10">poll_interval</option>
    <option value='{"max_retries":VALUE}' data-type="int" data-placeholder="3">max_retries</option>
    <option value='{"retry_delay":VALUE}' data-type="int" data-placeholder="10">retry_delay</option>
  </optgroup>
</select>

<div class="row">
  <label>Value:</label>
  <input type="text" id="valueInput" placeholder="enter value" oninput="buildSms()">
</div>

<div id="output">Select a key and enter a value above.</div>
<button onclick="copySms()">Copy</button>
<span class="copied" id="copiedMsg">Copied!</span>

<script>
function onKeyChange() {
  const sel = document.getElementById('keySelect');
  const opt = sel.options[sel.selectedIndex];
  const input = document.getElementById('valueInput');
  input.placeholder = opt.dataset.placeholder || '';
  input.value = '';
  buildSms();
}

function buildSms() {
  const sel = document.getElementById('keySelect');
  const opt = sel.options[sel.selectedIndex];
  const template = opt.value;
  const raw = document.getElementById('valueInput').value.trim();
  const type = opt.dataset.type;
  const out = document.getElementById('output');

  if (!raw) { out.textContent = 'Select a key and enter a value above.'; return; }

  let value;
  if (type === 'string') {
    value = JSON.stringify(raw);
  } else if (type === 'bool') {
    value = raw.toLowerCase() === 'true' ? 'true' : 'false';
  } else {
    value = raw;
  }

  const sms = template.replace('VALUE', value);
  try {
    JSON.parse(sms);
    out.textContent = sms;
    out.style.color = '#afa';
  } catch {
    out.textContent = 'Invalid — check your value';
    out.style.color = '#f66';
  }
}

function copySms() {
  const text = document.getElementById('output').textContent;
  navigator.clipboard.writeText(text).then(() => {
    const msg = document.getElementById('copiedMsg');
    msg.style.display = 'inline';
    setTimeout(() => msg.style.display = 'none', 2000);
  });
}
</script>
</body>
</html>"""


async def _is_file_stable(path: Path, wait: float = 1.0) -> bool:
    """Ensure file size isn't changing (simple protection against incomplete uploads)."""
    try:
        size1 = path.stat().st_size
    except FileNotFoundError:
        return False
    await asyncio.sleep(wait)
    try:
        size2 = path.stat().st_size
    except FileNotFoundError:
        return False
    return size1 == size2


async def watch_uploads():
    """Background task: poll the uploads folder, process new videos alphabetically, and move originals to processed."""
    logging.info('Starting uploads watcher')
    while True:
        try:
            # gather files matching known extensions, sorted alphabetically
            files = sorted([p for p in UPLOAD_DIR.iterdir() if p.is_file() and p.suffix.lower() in VIDEO_EXTS])
            for video_path in files:
                # check file is stable (likely finished uploading)
                stable = await _is_file_stable(video_path)
                if not stable:
                    logging.info(f'Skipping unstable file: {video_path.name}')
                    continue

                logging.info(f'Processing {video_path.name}...')
                try:
                    # run blocking processing in a thread so the event loop isn't blocked
                    await asyncio.to_thread(process_video, str(video_path), project=str(PROCESSED_DIR))

                    # update database with detections from this video
                    if db is not None:
                        video_time = parse_timestamp_from_filename(video_path.name)
                        db.process_yolo_labels_for_video(
                            str(PROCESSED_DIR),
                            video_path.stem,
                            video_path.name,
                            video_time,
                            max_gap_seconds=MAX_GAP_SECONDS
                        )
                        db.close_inactive_tracks(older_than_seconds=MAX_GAP_SECONDS * 2)
                        logging.info(f'Updated database for {video_path.name}')

                    # move original into its video subdirectory (ensure no overwrite)
                    video_subdir = PROCESSED_DIR / video_path.stem
                    video_subdir.mkdir(exist_ok=True)
                    dest = video_subdir / video_path.name
                    if dest.exists():
                        dest = video_subdir / f"{video_path.stem}-{int(time.time())}{video_path.suffix}"
                    shutil.move(str(video_path), str(dest))
                    logging.info(f'Moved original to {dest}')
                except Exception:
                    logging.exception(f'Failed to process {video_path.name}')
        except Exception:
            logging.exception('Error while watching uploads')

        await asyncio.sleep(POLL_INTERVAL)


@app.on_event("startup")
async def startup_event():
    global db
    # Initialize database
    db = TrackerDB(DB_PATH)
    logging.info(f'Initialized database: {DB_PATH}')
    # Launch background watcher
    asyncio.create_task(watch_uploads())
    logging.info('Background upload watcher started')


@app.on_event("shutdown")
async def shutdown_event():
    global db
    if db is not None:
        db.close()
        logging.info('Database closed')
