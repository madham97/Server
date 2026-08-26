from fastapi import APIRouter
from fastapi.responses import HTMLResponse

router = APIRouter()

_PAGE = """<!DOCTYPE html>
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
  select, input[type=text] { background: #222; color: #eee; border: 1px solid #444; padding: 8px; border-radius: 4px; font-family: monospace; font-size: 1em; }
  select { width: 100%; margin-bottom: 12px; }
  .row { display: flex; gap: 12px; margin-bottom: 12px; align-items: center; }
  .row label { color: #888; white-space: nowrap; }
  .row input { flex: 1; }
  #pending { background: #1a1a1a; border: 1px solid #444; border-radius: 4px; padding: 12px 16px; margin: 12px 0; min-height: 24px; }
  #pending .empty { color: #666; }
  #pending .pair { display: inline-block; background: #223; color: #afa; border-radius: 4px; padding: 4px 10px; margin: 4px 6px 4px 0; }
  #pending .pair button { background: none; border: none; color: #f66; cursor: pointer; margin-left: 6px; font-family: monospace; }
  #output { background: #1a1a1a; border: 1px solid #444; border-radius: 4px; padding: 16px; font-size: 1.1em; color: #afa; word-break: break-all; min-height: 48px; margin: 12px 0; }
  button.main { background: #7cf; color: #111; border: none; padding: 10px 24px; border-radius: 4px; font-family: monospace; font-size: 1em; cursor: pointer; font-weight: bold; }
  button.main:active { background: #5ad; }
  button.secondary { background: #333; color: #eee; border: 1px solid #555; padding: 10px 20px; border-radius: 4px; font-family: monospace; font-size: 1em; cursor: pointer; }
  .copied { color: #afa; margin-left: 12px; display: none; }
</style>
</head>
<body>
<h1>SMS Config Helper</h1>
<p class="sub">Build a <code>SET key=value ...</code> SMS to text to the device's SIM number. Live settings apply immediately; restart settings apply on the recorder's next restart.</p>

<h2>Recorder (requires restart)</h2>
<table>
  <tr><th>Key</th><th>Type</th><th>Example</th></tr>
  <tr><td class="key">mode</td><td>string</td><td>image_motion, image_interval</td></tr>
  <tr><td class="key">motion_threshold</td><td>float</td><td>0.015 = sensitive, 0.05 = less sensitive</td></tr>
  <tr><td class="key">motion_cooldown</td><td>seconds</td><td>60</td></tr>
  <tr><td class="key">detection_interval</td><td>seconds</td><td>1</td></tr>
  <tr><td class="key">image_interval</td><td>seconds</td><td>30</td></tr>
  <tr><td class="key">image_quality</td><td>int 1-100</td><td>75</td></tr>
</table>

<h2>Uploader (live — no restart needed)</h2>
<table>
  <tr><th>Key</th><th>Type</th><th>Example</th></tr>
  <tr><td class="key">webp_compress</td><td>bool</td><td>true / false</td></tr>
  <tr><td class="key">webp_quality</td><td>int 1-100</td><td>80</td></tr>
</table>

<h2>Build your SMS</h2>
<select id="keySelect" onchange="onKeyChange()">
  <optgroup label="Recorder (requires restart)">
    <option value="mode" data-type="string" data-placeholder="image_motion">mode</option>
    <option value="motion_threshold" data-type="float" data-placeholder="0.015">motion_threshold</option>
    <option value="motion_cooldown" data-type="float" data-placeholder="60">motion_cooldown</option>
    <option value="detection_interval" data-type="float" data-placeholder="1">detection_interval</option>
    <option value="image_interval" data-type="float" data-placeholder="30">image_interval</option>
    <option value="image_quality" data-type="int" data-placeholder="75">image_quality</option>
  </optgroup>
  <optgroup label="Uploader (live)">
    <option value="webp_compress" data-type="bool" data-placeholder="true">webp_compress</option>
    <option value="webp_quality" data-type="int" data-placeholder="80">webp_quality</option>
  </optgroup>
</select>

<div class="row">
  <label>Value:</label>
  <input type="text" id="valueInput" placeholder="enter value">
</div>
<button class="secondary" onclick="addPair()">Add to SMS</button>

<h2>Pending keys</h2>
<div id="pending"><span class="empty">None added yet — SMS will just read STATUS.</span></div>

<div id="output">STATUS</div>
<button class="main" onclick="copySms()">Copy</button>
<button class="secondary" onclick="clearPairs()">Clear</button>
<span class="copied" id="copiedMsg">Copied!</span>

<script>
let pairs = {};

function onKeyChange() {
  const sel = document.getElementById('keySelect');
  const opt = sel.options[sel.selectedIndex];
  const input = document.getElementById('valueInput');
  input.placeholder = opt.dataset.placeholder || '';
  input.value = '';
}

function formatValue(raw, type) {
  if (type === 'bool') return raw.toLowerCase() === 'true' ? 'true' : 'false';
  return raw;
}

function addPair() {
  const sel = document.getElementById('keySelect');
  const opt = sel.options[sel.selectedIndex];
  const key = opt.value;
  const type = opt.dataset.type;
  const raw = document.getElementById('valueInput').value.trim();
  if (!raw) return;
  pairs[key] = formatValue(raw, type);
  document.getElementById('valueInput').value = '';
  renderPending();
  buildSms();
}

function removePair(key) {
  delete pairs[key];
  renderPending();
  buildSms();
}

function clearPairs() {
  pairs = {};
  renderPending();
  buildSms();
}

function renderPending() {
  const el = document.getElementById('pending');
  const keys = Object.keys(pairs);
  if (keys.length === 0) {
    el.innerHTML = '<span class="empty">None added yet — SMS will just read STATUS.</span>';
    return;
  }
  el.innerHTML = keys.map(k =>
    `<span class="pair">${k}=${pairs[k]}<button onclick="removePair('${k}')">&times;</button></span>`
  ).join('');
}

function buildSms() {
  const keys = Object.keys(pairs);
  const out = document.getElementById('output');
  if (keys.length === 0) {
    out.textContent = 'STATUS';
    return;
  }
  out.textContent = 'SET ' + keys.map(k => `${k}=${pairs[k]}`).join(' ');
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


@router.get("/config-help", response_class=HTMLResponse)
async def config_help():
    return _PAGE
