"""
SAM 3 YOLO Auto-Labeler — Local Server
Runs entirely on your local machine with GPU acceleration.
No Google Colab or Ngrok required.
"""

import os
import json
import shutil
import base64
import sys
import torch
import cv2
import zipfile
import gc
import urllib.parse
import numpy as np
from pathlib import Path
from contextlib import nullcontext
from flask import Flask, render_template, request, jsonify, Response
from flask_cors import CORS
from PIL import Image
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --- SSL fix for Windows machines with certificate issues ---
import ssl
try:
    import httpx
except ModuleNotFoundError:
    httpx = None

ssl._create_default_https_context = ssl._create_unverified_context
os.environ["CURL_CA_BUNDLE"] = ""
os.environ["REQUESTS_CA_BUNDLE"] = ""
os.environ["HF_HUB_DISABLE_SSL_VERIFY"] = "1"
# Patch httpx (used internally by huggingface_hub) to skip SSL verify if available
if httpx is not None:
    _original_client_init = httpx.Client.__init__
    def _patched_client_init(self, *args, **kwargs):
        kwargs.setdefault("verify", False)
        _original_client_init(self, *args, **kwargs)
    httpx.Client.__init__ = _patched_client_init
else:
    print("--> httpx not installed; skipping SSL patch", flush=True)

# Ensure HF token is available for gated model downloads
_hf_token = os.environ.get("HF_TOKEN", "")
if _hf_token:
    # Write token to HF cache so hf_hub_download can find it
    from pathlib import Path as _Path
    _token_path = _Path.home() / ".cache" / "huggingface" / "token"
    _token_path.parent.mkdir(parents=True, exist_ok=True)
    _token_path.write_text(_hf_token)
# --- End SSL fix ---

app = Flask(__name__)
CORS(app)

# ------ Configuration ------
BASE_DIR = Path(__file__).parent.resolve()
CONFIG_FILE = BASE_DIR / "config.json"

DEFAULT_CONFIG = {
    "input_dir": str(BASE_DIR / "data" / "raw_images"),
    "output_dir": str(BASE_DIR / "data" / "dataset"),
    "classes": [
        {"id": 0, "name": "Foreground", "prompt": "the main subject", "invert": False, "color": "#00ff00"}
    ]
}


def load_config():
    if CONFIG_FILE.exists():
        with open(CONFIG_FILE, 'r') as f:
            cfg = json.load(f)
        # Resolve relative paths against BASE_DIR
        for key in ('input_dir', 'output_dir'):
            if not os.path.isabs(cfg[key]):
                cfg[key] = str(BASE_DIR / cfg[key])
        return cfg
    return DEFAULT_CONFIG.copy()


def save_config(cfg):
    CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(CONFIG_FILE, 'w') as f:
        json.dump(cfg, f, indent=4)


# ------ SAM 3 Model Loading ------
print("--> Loading SAM 3 Model (this may take a minute)...", flush=True)

# Locate BPE vocab file (adjust path if sam3 is cloned elsewhere)
SAM3_DIR = None
for candidate in [BASE_DIR / "sam3", Path.cwd() / "sam3", Path(__file__).parent.parent / "sam3"]:
    if candidate.exists():
        SAM3_DIR = candidate
        break

if SAM3_DIR is None:
    raise FileNotFoundError(
        "SAM 3 repository not found. Please clone it:\n"
        "  git clone https://github.com/facebookresearch/sam3.git\n"
        "  cd sam3 && pip install -e ."
    )

sys.path.insert(0, str(SAM3_DIR))

BPE_PATH = str(SAM3_DIR / "sam3" / "assets" / "bpe_simple_vocab_16e6.txt.gz")

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = build_sam3_image_model(bpe_path=BPE_PATH).to(device).eval()


def make_fp32_safe(m):
    """Convert bfloat16/float16 params to float32 for stable inference."""
    for name, param in m.named_parameters(recurse=False):
        if param.dtype in [torch.bfloat16, torch.float16]:
            param.data = param.data.to(torch.float32)
    for name, buf in m.named_buffers(recurse=False):
        if buf.dtype in [torch.bfloat16, torch.float16]:
            buf.data = buf.data.to(torch.float32)


for m in model.modules():
    make_fp32_safe(m)

processor = Sam3Processor(model)
if hasattr(processor, 'autocast') and device.type == "cuda":
    processor.autocast = torch.autocast(device_type="cuda", dtype=torch.float32)

print("--> Model loaded successfully!", flush=True)


# ------ Utility Functions ------

def secure_rel_path(path):
    """Sanitize relative path to prevent directory traversal."""
    clean = os.path.normpath(path).replace('..', '')
    return clean.lstrip('/\\')


def get_paired_paths(cfg, rel_path):
    """Returns (input_img, output_img, output_txt) for YOLO folder structure."""
    in_img = os.path.join(cfg['input_dir'], rel_path)
    out_img = os.path.join(cfg['output_dir'], "images", rel_path)
    out_txt = os.path.join(cfg['output_dir'], "labels", os.path.splitext(rel_path)[0] + ".txt")
    return in_img, out_img, out_txt


def mask_to_yolo_polygons(binary_mask, invert=False):
    """Convert binary mask to normalized YOLO polygon coordinates."""
    binary_mask = np.squeeze(binary_mask)
    if binary_mask.ndim != 2:
        return []
    if invert:
        binary_mask = np.logical_not(binary_mask)
    h, w = binary_mask.shape
    mask_uint8 = np.ascontiguousarray((binary_mask * 255).astype(np.uint8))
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    polygons = []
    for contour in contours:
        epsilon = 0.002 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        if len(approx) < 3:
            continue
        polygons.append([{"x": float(pt[0][0]) / w, "y": float(pt[0][1]) / h} for pt in approx])
    return polygons


def parse_yolo_txt(txt_path):
    """Read YOLO polygon annotation file."""
    polygons = []
    if os.path.exists(txt_path):
        with open(txt_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 7:
                    cls_id = int(parts[0])
                    pts = [{"x": float(parts[i]), "y": float(parts[i + 1])} for i in range(1, len(parts), 2)]
                    polygons.append({"classId": cls_id, "points": pts})
    return polygons


def infer_image(img_path, prompts):
    """Run SAM 3 inference on an image with given text prompts."""
    image = Image.open(img_path).convert("RGB")
    results = []
    autocast_ctx = torch.autocast(device_type="cuda", dtype=torch.float32) if device.type == "cuda" else nullcontext()
    with torch.inference_mode(), autocast_ctx:
        inference_state = processor.set_image(image)
        for cls in prompts:
            output = processor.set_text_prompt(state=inference_state, prompt=cls['prompt'])
            masks = output["masks"].cpu().numpy()
            scores = output["scores"].cpu().numpy()
            for i, mask in enumerate(masks):
                if scores[i] < 0.50:
                    continue
                polys = mask_to_yolo_polygons(mask, invert=cls.get('invert', False))
                for p in polys:
                    results.append({"classId": cls['id'], "points": p})
    del image
    del inference_state
    return results


# ------ API Routes ------

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/config', methods=['GET', 'POST'])
def api_config():
    if request.method == 'GET':
        return jsonify(load_config())
    save_config(request.json)
    return jsonify({"success": True})


@app.route('/api/gallery', methods=['GET'])
def api_gallery():
    cfg = load_config()
    images = []
    input_dir = cfg['input_dir']
    os.makedirs(input_dir, exist_ok=True)

    for root, dirs, files in os.walk(input_dir):
        for f in files:
            if f == ".keep":
                continue
            if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                full_path = os.path.join(root, f)
                rel_path = os.path.relpath(full_path, input_dir).replace('\\', '/')
                txt_path = os.path.join(cfg['output_dir'], "labels", os.path.splitext(rel_path)[0] + ".txt")
                status = "approved" if os.path.exists(txt_path) else "pending"
                images.append({"filename": rel_path, "status": status})

    images = sorted(images, key=lambda x: x['filename'])
    return jsonify({"success": True, "images": images})


@app.route('/api/mkdir', methods=['POST'])
def api_mkdir():
    data = request.json
    target = secure_rel_path(data.get('path', ''))
    if not target:
        return jsonify({"success": False, "error": "Invalid path"})
    full_path = os.path.join(load_config()['input_dir'], target)
    os.makedirs(full_path, exist_ok=True)
    with open(os.path.join(full_path, ".keep"), 'w') as f:
        f.write("")
    return jsonify({"success": True})


@app.route('/api/file_ops', methods=['POST'])
def api_file_ops():
    data = request.json
    action = data.get('action')
    files = data.get('files', [])
    target = secure_rel_path(data.get('target', ''))
    cfg = load_config()

    if not files or target == '':
        return jsonify({"success": False, "error": "Missing params"})

    try:
        for rel_path in files:
            in_img, out_img, out_txt = get_paired_paths(cfg, rel_path)

            if action == "rename":
                t_in_img, t_out_img, t_out_txt = get_paired_paths(cfg, target)
            else:
                filename = os.path.basename(rel_path)
                new_rel_path = os.path.join(target, filename)
                t_in_img, t_out_img, t_out_txt = get_paired_paths(cfg, new_rel_path)

            os.makedirs(os.path.dirname(t_in_img), exist_ok=True)

            if action in ["move", "rename"]:
                if os.path.exists(in_img):
                    shutil.move(in_img, t_in_img)
                if os.path.exists(out_img):
                    os.makedirs(os.path.dirname(t_out_img), exist_ok=True)
                    shutil.move(out_img, t_out_img)
                if os.path.exists(out_txt):
                    os.makedirs(os.path.dirname(t_out_txt), exist_ok=True)
                    shutil.move(out_txt, t_out_txt)
            elif action == "copy":
                if os.path.exists(in_img):
                    shutil.copy2(in_img, t_in_img)
                if os.path.exists(out_img):
                    os.makedirs(os.path.dirname(t_out_img), exist_ok=True)
                    shutil.copy2(out_img, t_out_img)
                if os.path.exists(out_txt):
                    os.makedirs(os.path.dirname(t_out_txt), exist_ok=True)
                    shutil.copy2(out_txt, t_out_txt)

        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})


@app.route('/api/upload', methods=['POST'])
def api_upload():
    cfg = load_config()
    count = 0
    for file in request.files.getlist('images'):
        if file.filename == '':
            continue
        safe_path = secure_rel_path(file.filename)
        dest = os.path.join(cfg['input_dir'], safe_path)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        file.save(dest)
        count += 1
    return jsonify({"success": True, "uploaded": count})


@app.route('/api/upload_zip', methods=['POST'])
def api_upload_zip():
    cfg = load_config()
    zip_file = request.files.get('zip')
    target_path = request.form.get('target_path', '')
    if not zip_file:
        return jsonify({"success": False})

    tmp_path = os.path.join(cfg['input_dir'], "temp_upload.zip")
    zip_file.save(tmp_path)
    count = 0
    with zipfile.ZipFile(tmp_path, 'r') as zip_ref:
        for member in zip_ref.namelist():
            if member.lower().endswith(('.png', '.jpg', '.jpeg')) and not member.startswith('__MACOSX'):
                safe_name = secure_rel_path(os.path.join(target_path, member))
                dest = os.path.join(cfg['input_dir'], safe_name)
                os.makedirs(os.path.dirname(dest), exist_ok=True)
                with zip_ref.open(member) as src, open(dest, 'wb') as dst:
                    shutil.copyfileobj(src, dst)
                count += 1
    os.remove(tmp_path)
    return jsonify({"success": True, "uploaded": count})


@app.route('/api/image/<path:filename>', methods=['DELETE'])
def api_delete(filename):
    cfg = load_config()
    fname = urllib.parse.unquote(filename)
    in_img, out_img, out_txt = get_paired_paths(cfg, fname)

    for p in [in_img, out_img, out_txt]:
        if os.path.exists(p):
            os.remove(p)
    return jsonify({"success": True})


@app.route('/api/image/<path:filename>/data', methods=['GET'])
def api_image_data(filename):
    cfg = load_config()
    fname = urllib.parse.unquote(filename)
    img_path = os.path.join(cfg['input_dir'], fname)
    txt_path = os.path.join(cfg['output_dir'], "labels", os.path.splitext(fname)[0] + ".txt")

    if not os.path.exists(img_path):
        return jsonify({"success": False, "error": "Not found"})
    with open(img_path, "rb") as f:
        b64_string = base64.b64encode(f.read()).decode('utf-8')
    return jsonify({"success": True, "image_b64": b64_string, "annotations": parse_yolo_txt(txt_path)})


@app.route('/api/auto_label', methods=['POST'])
def api_auto_label():
    data = request.json
    try:
        results = infer_image(os.path.join(load_config()['input_dir'], data['filename']), data['prompts'])
        torch.cuda.empty_cache()
        gc.collect()
        return jsonify({"success": True, "polygons": results})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})


@app.route('/api/auto_label_bulk', methods=['POST'])
def api_auto_label_bulk():
    data = request.json
    files = data['filenames']
    prompts = data['prompts']
    cfg = load_config()

    def generate():
        yield f"data: {json.dumps({'status': 'start', 'total': len(files)})}\n\n"
        for i, fname in enumerate(files):
            try:
                results = infer_image(os.path.join(cfg['input_dir'], fname), prompts)

                out_lbl_dir = os.path.join(cfg['output_dir'], "labels", os.path.dirname(fname))
                out_img_dir = os.path.join(cfg['output_dir'], "images", os.path.dirname(fname))
                os.makedirs(out_lbl_dir, exist_ok=True)
                os.makedirs(out_img_dir, exist_ok=True)

                dst_img = os.path.join(cfg['output_dir'], "images", fname)
                src_img = os.path.join(cfg['input_dir'], fname)
                if not os.path.exists(dst_img) and os.path.exists(src_img):
                    shutil.copy(src_img, dst_img)

                txt_path = os.path.join(cfg['output_dir'], "labels", os.path.splitext(fname)[0] + ".txt")
                with open(txt_path, 'w') as f:
                    for ann in results:
                        pstr = " ".join([f"{pt['x']:.6f} {pt['y']:.6f}" for pt in ann['points']])
                        f.write(f"{ann['classId']} {pstr}\n")

                yield f"data: {json.dumps({'status': 'progress', 'current': i + 1, 'filename': fname})}\n\n"
            except Exception as e:
                yield f"data: {json.dumps({'status': 'error', 'filename': fname, 'error': str(e)})}\n\n"
            finally:
                torch.cuda.empty_cache()
                gc.collect()

        yield f"data: {json.dumps({'status': 'done'})}\n\n"

    return Response(generate(), mimetype='text/event-stream')


@app.route('/api/save', methods=['POST'])
def api_save():
    data = request.json
    fname = data['filename']
    cfg = load_config()

    out_lbl_dir = os.path.join(cfg['output_dir'], "labels", os.path.dirname(fname))
    out_img_dir = os.path.join(cfg['output_dir'], "images", os.path.dirname(fname))
    os.makedirs(out_lbl_dir, exist_ok=True)
    os.makedirs(out_img_dir, exist_ok=True)

    try:
        src_img = os.path.join(cfg['input_dir'], fname)
        dst_img = os.path.join(cfg['output_dir'], "images", fname)
        if not os.path.exists(dst_img) and os.path.exists(src_img):
            shutil.copy(src_img, dst_img)

        txt_path = os.path.join(cfg['output_dir'], "labels", os.path.splitext(fname)[0] + ".txt")
        with open(txt_path, 'w') as f:
            for ann in data['annotations']:
                pstr = " ".join([f"{pt['x']:.6f} {pt['y']:.6f}" for pt in ann['points']])
                f.write(f"{ann['classId']} {pstr}\n")
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})


# ------ Entry Point ------

if __name__ == '__main__':
    # Ensure data directories exist
    cfg = load_config()
    os.makedirs(cfg['input_dir'], exist_ok=True)
    os.makedirs(os.path.join(cfg['output_dir'], "images"), exist_ok=True)
    os.makedirs(os.path.join(cfg['output_dir'], "labels"), exist_ok=True)

    print("=" * 60)
    print("  SAM 3 YOLO Auto-Labeler — Local Server")
    print("  Open in browser: http://localhost:5000")
    print("=" * 60)
    app.run(host='0.0.0.0', port=5000, debug=False)
