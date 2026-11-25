"""Web app: upload a photo of a waste item, get its class and the bin it goes in.

    python app.py                # then open http://127.0.0.1:5000
    python app.py --port 8080

POST /predict  (multipart form field "photo")  ->  JSON with class, bin, tip and the top-3 scores.
"""
import argparse
import io
import sys
from pathlib import Path

from flask import Flask, jsonify, render_template_string, request
from PIL import Image, UnidentifiedImageError

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from binsort import BinRules, WasteClassifier  # noqa: E402

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 15 * 1024 * 1024
clf = WasteClassifier()
rules = BinRules()

PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Which bin?</title>
<style>
  :root { --bg:#f6f7f4; --card:#fff; --ink:#1d2a1f; --muted:#5d6b60; --line:#dfe4dc; --accent:#2f7d4f; --bar:#cfe5d6; }
  @media (prefers-color-scheme: dark) { :root { --bg:#141a15; --card:#1c241e; --ink:#e8efe9; --muted:#9bab9f; --line:#2d392f; --accent:#5cc287; --bar:#2b4a37; } }
  * { box-sizing:border-box; }
  body { margin:0; background:var(--bg); color:var(--ink); font:16px/1.5 system-ui,-apple-system,Segoe UI,sans-serif; }
  main { max-width:720px; margin:0 auto; padding:32px 16px 48px; }
  h1 { font-size:1.6rem; margin:0 0 4px; } p.sub { color:var(--muted); margin:0 0 24px; }
  .card { background:var(--card); border:1px solid var(--line); border-radius:14px; padding:20px; }
  #drop { display:block; border:2px dashed var(--line); border-radius:12px; padding:28px 16px; text-align:center; color:var(--muted); cursor:pointer; }
  #drop.over { border-color:var(--accent); color:var(--ink); }
  #drop strong { color:var(--ink); }
  #preview { display:none; width:100%; max-height:340px; object-fit:contain; border-radius:10px; margin-top:16px; background:#0001; }
  #result { display:none; margin-top:20px; }
  .bin { font-size:1.5rem; font-weight:700; color:var(--accent); margin:2px 0; }
  .meta { color:var(--muted); }
  .row { display:grid; grid-template-columns:96px 1fr 52px; gap:10px; align-items:center; margin:8px 0; font-size:.95rem; }
  .track { background:var(--bar); border-radius:6px; height:12px; overflow:hidden; } .fill { background:var(--accent); height:100%; }
  .num { text-align:right; font-variant-numeric:tabular-nums; }
  .note { font-size:.85rem; color:var(--muted); margin-top:14px; }
  #err { color:#b3261e; margin-top:12px; display:none; }
</style></head>
<body><main>
  <h1>Which bin does it go in?</h1>
  <p class="sub">Upload a photo of one waste item. The model names the material and the bin it belongs in.</p>
  <div class="card">
    <label id="drop" for="file"><strong>Choose a photo</strong> or drop one here<br><span>JPG or PNG, one item per photo works best</span></label>
    <input id="file" type="file" accept="image/*" hidden>
    <img id="preview" alt="Your photo">
    <div id="err"></div>
    <div id="result">
      <div class="meta">Predicted class: <strong id="cls"></strong></div>
      <div class="bin" id="bin"></div>
      <div class="meta" id="tip"></div>
      <h3 style="margin:18px 0 4px;font-size:1rem">Top 3 model scores</h3>
      <div id="bars"></div>
      <div class="note">Scores are the ensemble's softmax outputs. They rank the classes but are not calibrated probabilities. The model only knows cardboard, glass, metal, paper, plastic and trash, so food waste and electronics are not covered.</div>
    </div>
  </div>
</main>
<script>
const file = document.getElementById('file'), drop = document.getElementById('drop');
const $ = id => document.getElementById(id);
async function run(f) {
  $('err').style.display = 'none'; $('result').style.display = 'none';
  $('preview').src = URL.createObjectURL(f); $('preview').style.display = 'block';
  const fd = new FormData(); fd.append('photo', f);
  try {
    const r = await fetch('/predict', { method: 'POST', body: fd });
    const j = await r.json();
    if (!r.ok) throw new Error(j.error || 'Something went wrong');
    $('cls').textContent = j.label; $('bin').textContent = j.bin; $('tip').textContent = j.tip;
    $('bars').innerHTML = j.top.map(t => `<div class="row"><span>${t.label}</span><div class="track"><div class="fill" style="width:${(t.score*100).toFixed(1)}%"></div></div><span class="num">${t.score.toFixed(2)}</span></div>`).join('');
    $('result').style.display = 'block';
  } catch (e) { $('err').textContent = e.message; $('err').style.display = 'block'; }
}
file.addEventListener('change', () => file.files[0] && run(file.files[0]));
['dragenter','dragover'].forEach(ev => drop.addEventListener(ev, e => { e.preventDefault(); drop.classList.add('over'); }));
['dragleave','drop'].forEach(ev => drop.addEventListener(ev, e => { e.preventDefault(); drop.classList.remove('over'); }));
drop.addEventListener('drop', e => e.dataTransfer.files[0] && run(e.dataTransfer.files[0]));
</script></body></html>"""


@app.get("/")
def index():
    return render_template_string(PAGE)


@app.post("/predict")
def predict():
    f = request.files.get("photo")
    if f is None:
        return jsonify(error="No photo was uploaded."), 400
    try:
        img = Image.open(io.BytesIO(f.read()))
        img.load()
    except (UnidentifiedImageError, OSError):
        return jsonify(error="That file is not an image the app can read."), 400
    r = clf.predict(img)
    b = rules.lookup(r["label"])
    return jsonify(label=r["label"], bin=b["bin"], tip=b["tip"], top=r["top"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=5000)
    args = ap.parse_args()
    app.run(host=args.host, port=args.port)
