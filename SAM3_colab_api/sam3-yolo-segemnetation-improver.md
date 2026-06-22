# YOLO -> SAM 3 Automated Pseudo-Labeling Studio

### Prerequisites:
1. **Hugging Face Token:** Generate at [huggingface.co](https://huggingface.co/settings/tokens) and accept SAM 3 terms.
2. **Ngrok Token:** Get from [dashboard.ngrok.com](https://dashboard.ngrok.com/get-started/your-authtoken).
3. **Colab Secrets:** Add `HF_TOKEN` and `NGROK_TOKEN` to Colab's secret manager (🔑 icon).

---

### Cell 1: Install Dependencies
```bash
!mkdir -p templates
!git clone https://github.com/facebookresearch/sam3.git
%cd sam3
!pip install -e .
!pip install flask flask-cors pyngrok opencv-python-headless ultralytics
```

---

### Cell 2: Mount Google Drive
```python
from google.colab import drive
drive.mount('/content/drive')
print("✅ Google Drive mounted successfully!")
```

---

### Cell 3: The Backend Server (`app.py`)
*Note: Run this cell to write the backend Python script.*

```python
%%writefile app.py
import os
import json
import base64
import torch
import cv2
import gc
import threading
import time
import shutil
import zipfile
import numpy as np
from flask import Flask, render_template, request, jsonify, Response
from flask_cors import CORS
from PIL import Image
from ultralytics import YOLO

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})

# ==========================================
# 1. GLOBAL STATE & MODELS
# ==========================================
print("--> Initializing SAM 3 Model...", flush=True)

BPE_PATH = "/content/sam3/sam3/assets/bpe_simple_vocab_16e6.txt.gz"
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

sam_model = build_sam3_image_model(bpe_path=BPE_PATH).cuda().eval()
def make_fp32_safe(m):
    for name, param in m.named_parameters(recurse=False):
        if param.dtype in [torch.bfloat16, torch.float16]: param.data = param.data.to(torch.float32)
    for name, buf in m.named_buffers(recurse=False):
        if buf.dtype in [torch.bfloat16, torch.float16]: buf.data = buf.data.to(torch.float32)
for m in sam_model.modules(): make_fp32_safe(m)

sam_processor = Sam3Processor(sam_model)
if hasattr(sam_processor, 'autocast'): sam_processor.autocast = torch.autocast(device_type="cuda", dtype=torch.float32)

yolo_model = None
yolo_classes = {}

# Engine State
PIPELINE_STATE = {
    "is_running": False,
    "total": 0,
    "current": 0,
    "current_file": "",
    "errors": 0,
    "logs": [],
    "preview_b64": "",
    "eta_seconds": 0
}

WORKER_QUEUE = []
OUTPUT_DIR = "/content/drive/MyDrive/AutoLabeled_Dataset"

APP_CONFIG = {
    "conf_threshold": 0.50,
    "active_classes": []
}

# ==========================================
# 2. UTILITY FUNCTIONS
# ==========================================
def log_msg(msg):
    print(msg, flush=True)
    PIPELINE_STATE["logs"].append(msg)
    if len(PIPELINE_STATE["logs"]) > 50: PIPELINE_STATE["logs"].pop(0)

def mask_to_yolo_polygons(binary_mask, img_w, img_h):
    # binary_mask expected shape: (H, W)
    binary_mask = np.squeeze(binary_mask)
    if binary_mask.ndim != 2: return []
    mask_uint8 = np.ascontiguousarray((binary_mask * 255).astype(np.uint8))
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    polygons = []
    for contour in contours:
        epsilon = 0.002 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        if len(approx) < 3: continue
        polys = [{"x": float(pt[0][0]) / img_w, "y": float(pt[0][1]) / img_h} for pt in approx]
        polygons.append(polys)
    return polygons

def generate_preview(img, boxes, polygons):
    h, w = img.shape[:2]
    scale = 640.0 / max(w, h)
    if scale < 1:
        img = cv2.resize(img, (int(w * scale), int(h * scale)))
        w, h = img.shape[:2]
    
    for b in boxes:
        x1, y1, x2, y2 = [int(v * scale) for v in b]
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        
    for ann in polygons:
        pts = np.array([[int(pt['x'] * w), int(pt['y'] * h)] for pt in ann['points']], np.int32)
        cv2.polylines(img, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
        overlay = img.copy()
        cv2.fillPoly(overlay, [pts], (0, 255, 0))
        cv2.addWeighted(overlay, 0.3, img, 0.7, 0, img)

    _, buffer = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 60])
    return base64.b64encode(buffer).decode('utf-8')

# ==========================================
# 3. BACKGROUND PIPELINE ENGINE
# ==========================================
def pipeline_worker():
    global yolo_model, sam_processor, WORKER_QUEUE
    PIPELINE_STATE["is_running"] = True
    PIPELINE_STATE["errors"] = 0
    PIPELINE_STATE["logs"] = []
    PIPELINE_STATE["total"] = len(WORKER_QUEUE)
    
    out_lbl_dir = os.path.join(OUTPUT_DIR, "labels")
    out_img_dir = os.path.join(OUTPUT_DIR, "images")
    os.makedirs(out_lbl_dir, exist_ok=True)
    os.makedirs(out_img_dir, exist_ok=True)
    
    start_time = time.time()
    
    for i, img_path in enumerate(WORKER_QUEUE):
        if not PIPELINE_STATE["is_running"]: break
        
        fname = os.path.basename(img_path)
        PIPELINE_STATE["current"] = i + 1
        PIPELINE_STATE["current_file"] = fname
        
        elapsed = time.time() - start_time
        avg_time = elapsed / (i + 1)
        PIPELINE_STATE["eta_seconds"] = int(avg_time * (PIPELINE_STATE["total"] - (i + 1)))

        txt_path = os.path.join(out_lbl_dir, os.path.splitext(fname)[0] + ".txt")
        dst_img = os.path.join(out_img_dir, fname)

        if os.path.exists(txt_path):
            log_msg(f"Skipping {fname} (Already exists)")
            continue

        try:
            img_cv2 = cv2.imread(img_path)
            img_h, img_w = img_cv2.shape[:2]
            
            yolo_res = yolo_model(img_cv2, conf=APP_CONFIG["conf_threshold"], verbose=False)[0]
            boxes = yolo_res.boxes.xyxy.cpu().numpy()
            classes = yolo_res.boxes.cls.cpu().numpy()
            
            final_polygons = []
            allowed_classes = set(APP_CONFIG["active_classes"])
            
            if len(boxes) > 0:
                image_pil = Image.fromarray(cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB))
                with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float32):
                    inference_state = sam_processor.set_image(image_pil)
                    
                    for idx, box in enumerate(boxes):
                        cls_id = int(classes[idx])
                        if len(allowed_classes) > 0 and cls_id not in allowed_classes: continue
                        
                        # Convert YOLO [x1, y1, x2, y2] to SAM3 [cx, cy, w, h] normalized [0, 1]
                        x1, y1, x2, y2 = box
                        cx = ((x1 + x2) / 2.0) / img_w
                        cy = ((y1 + y2) / 2.0) / img_h
                        bw = (x2 - x1) / img_w
                        bh = (y2 - y1) / img_h
                        norm_box = [float(cx), float(cy), float(bw), float(bh)]
                        
                        sam_processor.reset_all_prompts(state=inference_state)
                        output = sam_processor.add_geometric_prompt(box=norm_box, label=True, state=inference_state)
                        
                        # Fix: Handle 3 multimasks by extracting scores and finding the best mask
                        masks = output["masks"].cpu().numpy()[0]   # Shape: (3, H, W)
                        scores = output["scores"].cpu().numpy()[0] # Shape: (3,)
                        best_idx = np.argmax(scores)
                        best_mask = masks[best_idx]                # Shape: (H, W)
                        
                        polys = mask_to_yolo_polygons(best_mask, img_w, img_h)
                        for p in polys:
                            final_polygons.append({"classId": cls_id, "points": p})

            with open(txt_path, 'w') as f:
                for ann in final_polygons:
                    pstr = " ".join([f"{pt['x']:.6f} {pt['y']:.6f}" for pt in ann['points']])
                    f.write(f"{ann['classId']} {pstr}\n")
            
            if not os.path.exists(dst_img):
                shutil.copy(img_path, dst_img)

            PIPELINE_STATE["preview_b64"] = generate_preview(img_cv2, boxes, final_polygons)
            log_msg(f"Processed {fname} ({len(final_polygons)} polygons)")

        except Exception as e:
            log_msg(f"Error on {fname}: {str(e)}")
            PIPELINE_STATE["errors"] += 1
            
        finally:
            torch.cuda.empty_cache()
            gc.collect()

    PIPELINE_STATE["is_running"] = False
    WORKER_QUEUE.clear()
    log_msg("Pipeline finished!")

# ==========================================
# 4. API ENDPOINTS
# ==========================================
@app.route('/')
def index(): return render_template('index.html')

@app.route('/api/upload_model', methods=['POST'])
def api_upload_model():
    global yolo_model, yolo_classes
    if 'file' not in request.files: return jsonify({"error": "No file uploaded"})
    file = request.files['file']
    save_path = "/content/uploaded_model.pt"
    file.save(save_path)
    try:
        yolo_model = YOLO(save_path)
        yolo_classes = yolo_model.names
        return jsonify({"success": True, "classes": yolo_classes})
    except Exception as e: return jsonify({"error": str(e)})

# --- Filesystem Endpoints ---
@app.route('/api/gallery')
def api_gallery():
    target_dir = request.args.get('path', '/content/drive/MyDrive')
    if not os.path.exists(target_dir): return jsonify({"error": "Path not found"})
    items = []
    try:
        for f in os.listdir(target_dir):
            full_path = os.path.join(target_dir, f)
            if os.path.isdir(full_path):
                items.append({"name": f, "path": full_path, "type": "folder"})
            elif f.lower().endswith(('.png', '.jpg', '.jpeg')):
                items.append({"name": f, "path": full_path, "type": "image"})
    except Exception as e: return jsonify({"error": str(e)})
    return jsonify(sorted(items, key=lambda x: (x['type'] != 'folder', x['name'])))

@app.route('/api/upload_files', methods=['POST'])
def api_upload_files():
    target_dir = request.form.get('path', '/content/drive/MyDrive')
    os.makedirs(target_dir, exist_ok=True)
    count = 0
    for file in request.files.getlist('files'):
        if file.filename:
            file.save(os.path.join(target_dir, file.filename))
            count += 1
    return jsonify({"success": True, "uploaded": count})

@app.route('/api/upload_zip', methods=['POST'])
def api_upload_zip():
    target_dir = request.form.get('path', '/content/drive/MyDrive')
    zip_file = request.files.get('zip')
    if not zip_file: return jsonify({"success": False})
    
    tmp_path = "/content/temp_upload.zip"
    zip_file.save(tmp_path)
    count = 0
    with zipfile.ZipFile(tmp_path, 'r') as zip_ref:
        for member in zip_ref.namelist():
            if member.lower().endswith(('.png', '.jpg', '.jpeg')) and not member.startswith('__MACOSX'):
                dest = os.path.join(target_dir, os.path.basename(member))
                with zip_ref.open(member) as src, open(dest, 'wb') as dst:
                    shutil.copyfileobj(src, dst)
                count += 1
    os.remove(tmp_path)
    return jsonify({"success": True, "uploaded": count})

@app.route('/api/file_ops', methods=['POST'])
def api_file_ops():
    data = request.json
    action, files, target = data.get('action'), data.get('files', []), data.get('target', '')
    try:
        for f in files:
            if not os.path.exists(f): continue
            if action == "rename": os.rename(f, target)
            elif action == "move": shutil.move(f, target)
            elif action == "copy":
                if os.path.isdir(f): shutil.copytree(f, os.path.join(target, os.path.basename(f)))
                else: shutil.copy2(f, target)
            elif action == "delete":
                if os.path.isdir(f): shutil.rmtree(f)
                else: os.remove(f)
        return jsonify({"success": True})
    except Exception as e: return jsonify({"success": False, "error": str(e)})

@app.route('/api/mkdir', methods=['POST'])
def api_mkdir():
    try:
        os.makedirs(request.json.get('path'), exist_ok=True)
        return jsonify({"success": True})
    except Exception as e: return jsonify({"success": False, "error": str(e)})

# --- Queue & Config ---
@app.route('/api/queue/add', methods=['POST'])
def api_queue_add():
    global WORKER_QUEUE
    paths = request.json.get('paths', [])
    added = 0
    for p in paths:
        if os.path.isfile(p) and p.lower().endswith(('.png', '.jpg', '.jpeg')):
            if p not in WORKER_QUEUE: WORKER_QUEUE.append(p); added += 1
        elif os.path.isdir(p):
            for root, _, files in os.walk(p):
                for f in files:
                    if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                        full = os.path.join(root, f)
                        if full not in WORKER_QUEUE: WORKER_QUEUE.append(full); added += 1
    return jsonify({"success": True, "queue_size": len(WORKER_QUEUE), "added": added})

@app.route('/api/queue/clear', methods=['POST'])
def api_queue_clear():
    global WORKER_QUEUE
    WORKER_QUEUE.clear()
    return jsonify({"success": True})

@app.route('/api/config', methods=['POST'])
def api_config():
    global APP_CONFIG
    APP_CONFIG.update(request.json)
    return jsonify({"success": True})

# --- Pipeline Control ---
@app.route('/api/pipeline/start', methods=['POST'])
def api_pipeline_start():
    if PIPELINE_STATE["is_running"]: return jsonify({"error": "Already running"})
    if yolo_model is None: return jsonify({"error": "YOLO model not loaded"})
    if len(WORKER_QUEUE) == 0: return jsonify({"error": "Queue is empty"})
    threading.Thread(target=pipeline_worker, daemon=True).start()
    return jsonify({"success": True})

@app.route('/api/pipeline/stop', methods=['POST'])
def api_pipeline_stop():
    PIPELINE_STATE["is_running"] = False
    return jsonify({"success": True})

@app.route('/api/pipeline/status')
def api_pipeline_status():
    def generate():
        while True:
            yield f"data: {json.dumps(PIPELINE_STATE)}\n\n"
            time.sleep(1)
    return Response(generate(), mimetype='text/event-stream')

# --- QA Station Endpoints ---
@app.route('/api/qa/list')
def api_qa_list():
    out_img_dir = os.path.join(OUTPUT_DIR, "images")
    out_lbl_dir = os.path.join(OUTPUT_DIR, "labels")
    if not os.path.exists(out_img_dir): return jsonify([])
    
    # Fix: Actually read images and check if label exists, solving case-sensitivity
    valid_files = []
    for fname in os.listdir(out_img_dir):
        if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
            txt_path = os.path.join(out_lbl_dir, os.path.splitext(fname)[0] + ".txt")
            if os.path.exists(txt_path):
                valid_files.append(fname)
    return jsonify(sorted(valid_files))

@app.route('/api/qa/image/<filename>')
def api_qa_image(filename):
    img_path = os.path.join(OUTPUT_DIR, "images", filename)
    txt_path = os.path.join(OUTPUT_DIR, "labels", os.path.splitext(filename)[0] + ".txt")
    
    if not os.path.exists(img_path): return jsonify({"error": "Image not found"})
    with open(img_path, "rb") as f: b64_string = base64.b64encode(f.read()).decode('utf-8')
        
    polygons = []
    if os.path.exists(txt_path):
        with open(txt_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 7:
                    cls_id = int(parts[0])
                    pts = [{"x": float(parts[i]), "y": float(parts[i+1])} for i in range(1, len(parts), 2)]
                    polygons.append({"classId": cls_id, "points": pts})
                    
    return jsonify({"success": True, "image_b64": b64_string, "polygons": polygons, "classes": yolo_classes})

@app.route('/api/qa/save', methods=['POST'])
def api_qa_save():
    data = request.json
    filename = data['filename']
    txt_path = os.path.join(OUTPUT_DIR, "labels", os.path.splitext(filename)[0] + ".txt")
    try:
        with open(txt_path, 'w') as f:
            for ann in data['polygons']:
                pstr = " ".join([f"{pt['x']:.6f} {pt['y']:.6f}" for pt in ann['points']])
                f.write(f"{ann['classId']} {pstr}\n")
        return jsonify({"success": True})
    except Exception as e: return jsonify({"error": str(e)})

@app.route('/api/qa/delete', methods=['POST'])
def api_qa_delete():
    filename = request.json.get('filename')
    img_path = os.path.join(OUTPUT_DIR, "images", filename)
    txt_path = os.path.join(OUTPUT_DIR, "labels", os.path.splitext(filename)[0] + ".txt")
    try:
        if os.path.exists(img_path): os.remove(img_path)
        if os.path.exists(txt_path): os.remove(txt_path)
        return jsonify({"success": True})
    except Exception as e: return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

---

### Cell 4: The Frontend Interface (`index.html`)
*Note: Run this cell to generate the UI.*

```html
%%writefile templates/index.html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Pipeline Mission Control</title>
    <style>
        :root { --bg: #111; --panel: #1e1e1e; --text: #eee; --accent: #3b82f6; --success: #22c55e; --danger: #ef4444; --border: #333; }
        body { margin: 0; font-family: system-ui, sans-serif; background: var(--bg); color: var(--text); display: flex; height: 100vh; overflow: hidden; }
        
        nav { width: 70px; background: #000; display: flex; flex-direction: column; align-items: center; padding-top: 20px; gap: 20px; border-right: 1px solid var(--border); z-index: 50;}
        .nav-btn { cursor: pointer; background: none; border: none; font-size: 24px; opacity: 0.4; transition: 0.2s; padding: 10px; border-radius: 8px;}
        .nav-btn.active, .nav-btn:hover { opacity: 1; background: #222; }
        
        .view-panel { flex-grow: 1; display: none; flex-direction: column; overflow-y: auto; padding: 30px; box-sizing: border-box; }
        .view-panel.active { display: flex; }
        
        h1, h2, h3 { margin-top: 0; color: #fff; }
        .card { background: var(--panel); border: 1px solid var(--border); border-radius: 8px; padding: 20px; margin-bottom: 20px; }
        
        button { padding: 10px 20px; border: none; border-radius: 4px; font-weight: bold; cursor: pointer; transition: 0.2s; color: white; background: #444; }
        button:hover { filter: brightness(1.2); }
        button:disabled { opacity: 0.5; cursor: not-allowed; }
        .btn-primary { background: var(--accent); }
        .btn-success { background: var(--success); }
        .btn-danger { background: var(--danger); }
        .btn-outline { background: transparent; border: 1px solid #666; color: #ccc; padding: 6px 12px; }
        .btn-outline:hover { background: #333; }

        .dropzone { border: 2px dashed #555; border-radius: 8px; padding: 40px; text-align: center; color: #888; transition: 0.2s; cursor: pointer; }
        .dropzone.dragover { border-color: var(--accent); background: #1a2333; color: var(--accent); }
        
        .explorer-path { font-family: monospace; background: #000; padding: 10px; border-radius: 4px; color: var(--accent); cursor: pointer; border: 1px solid var(--border); margin-bottom: 15px;}
        .explorer-list { max-height: 400px; overflow-y: auto; border: 1px solid var(--border); border-radius: 4px; background: #000; }
        .exp-item { display: flex; align-items: center; padding: 10px; border-bottom: 1px solid var(--border); cursor: pointer; }
        .exp-item:hover { background: #222; }
        .exp-item input[type="checkbox"] { transform: scale(1.4); margin-right: 15px; cursor: pointer; }
        
        .bulk-action-bar { display: none; background: #222; padding: 10px; border-radius: 4px; margin-bottom: 10px; align-items: center; justify-content: space-between; border: 1px solid var(--accent); }

        .progress-bg { width: 100%; height: 20px; background: #000; border-radius: 10px; overflow: hidden; border: 1px solid var(--border); margin: 20px 0;}
        .progress-fill { height: 100%; background: var(--accent); width: 0%; transition: 0.3s; }
        .live-preview { width: 100%; max-width: 640px; height: 360px; background: #000; border: 1px solid var(--border); border-radius: 8px; margin: 0 auto 20px auto; display: flex; align-items: center; justify-content: center; overflow: hidden; }
        .live-preview img { max-width: 100%; max-height: 100%; object-fit: contain; }
        #logs { background: #000; font-family: monospace; font-size: 12px; padding: 15px; border-radius: 4px; height: 150px; overflow-y: auto; color: #0f0; border: 1px solid var(--border); }

        #view-qa { flex-direction: row; padding: 0; }
        .canvas-container { flex-grow: 1; background: #050505; display: flex; align-items: center; justify-content: center; position: relative; overflow: hidden;}
        canvas { max-width: 100%; max-height: 100%; cursor: crosshair; }
        aside { width: 300px; background: var(--panel); padding: 20px; border-left: 1px solid var(--border); display: flex; flex-direction: column; gap: 15px; }
        
        #toast { position: fixed; bottom: 20px; right: 20px; background: var(--panel); padding: 15px 20px; border-radius: 4px; border-left: 5px solid var(--success); transition: transform 0.3s; transform: translateY(150%); z-index: 1000; }
        #toast.show { transform: translateY(0); }
    </style>
</head>
<body>

    <nav>
        <button class="nav-btn active" onclick="switchView('config')" title="Configuration">⚙️</button>
        <button class="nav-btn" onclick="switchView('explorer')" title="File Explorer">📁</button>
        <button class="nav-btn" onclick="switchView('runner')" title="Live Factory">🏭</button>
        <button class="nav-btn" onclick="switchView('qa')" title="QA Review Station">👁️</button>
    </nav>

    <!-- 1. CONFIG VIEW -->
    <main id="view-config" class="view-panel active">
        <div style="max-width: 800px; margin: 0 auto; width: 100%;">
            <h1>Pipeline Configuration</h1>
            <div class="card">
                <h2>Upload YOLO Model</h2>
                <div id="dropzone" class="dropzone" onclick="document.getElementById('model-file').click()">
                    Drag & Drop your YOLO .pt file here<br><span style="font-size:12px; opacity:0.5">(or click to browse)</span>
                </div>
                <input type="file" id="model-file" accept=".pt" style="display:none;" onchange="uploadModel(this.files[0])">
                <div id="model-status" style="margin-top: 15px; font-weight: bold; color: var(--accent);">No model loaded.</div>
                
                <div style="margin-top: 20px;">
                    <label style="color:#aaa; font-size:12px; font-weight:bold; text-transform:uppercase;">Active Classes to Segment:</label>
                    <div id="class-toggles" style="display: flex; flex-wrap: wrap; gap: 10px; margin-top: 5px;"></div>
                </div>
                <div style="margin-top: 20px;">
                    <label style="color:#aaa; font-size:12px; font-weight:bold; text-transform:uppercase;">YOLO Confidence Threshold: <span id="conf-val">0.50</span></label>
                    <input type="range" id="inp-conf" min="0.1" max="0.95" step="0.05" value="0.50" style="width: 100%;" oninput="updateConfig()">
                </div>
            </div>
        </div>
    </main>

    <!-- 2. EXPLORER VIEW -->
    <main id="view-explorer" class="view-panel">
        <div style="max-width: 1000px; margin: 0 auto; width: 100%;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px;">
                <h1>File Explorer</h1>
                <div style="font-weight: bold; color: var(--success); border: 1px solid var(--success); padding: 5px 10px; border-radius: 4px;">Queue Size: <span id="queue-size">0</span></div>
            </div>
            
            <div class="card">
                <div class="explorer-path" id="exp-path" onclick="loadGallery('/content/drive/MyDrive')">/content/drive/MyDrive</div>
                
                <!-- Toolbar -->
                <div style="display: flex; gap: 10px; margin-bottom: 10px; flex-wrap: wrap;">
                    <button class="btn-outline" onclick="toggleSelectAll()">☑️ Select All Visible</button>
                    <button class="btn-outline" onclick="promptNewFolder()">📁 New Folder</button>
                    <input type="file" id="file-upload" multiple accept="image/*" style="display:none;" onchange="uploadFiles(event)">
                    <input type="file" id="zip-upload" accept=".zip" style="display:none;" onchange="uploadZip(event)">
                    <button class="btn-outline" onclick="document.getElementById('file-upload').click()">📄 Upload Images</button>
                    <button class="btn-outline" onclick="document.getElementById('zip-upload').click()">📦 Upload ZIP</button>
                </div>

                <!-- Bulk Action Bar -->
                <div id="bulk-action-bar" class="bulk-action-bar">
                    <div style="color: var(--accent); font-weight: bold;"><span id="sel-count">0</span> Selected</div>
                    <div style="display: flex; gap: 10px;">
                        <button id="btn-rename" class="btn-outline" onclick="promptRename()">✏️ Rename</button>
                        <button class="btn-outline" onclick="promptMove()">📂 Move</button>
                        <button class="btn-outline" onclick="promptCopy()">📄 Copy</button>
                        <button class="btn-outline" style="border-color: var(--danger); color: #ff6b6b;" onclick="bulkDelete()">🗑️ Delete</button>
                    </div>
                </div>

                <div class="explorer-list" id="exp-list"></div>
                
                <div style="display: flex; gap: 10px; margin-top: 20px;">
                    <button class="btn-primary" onclick="addSelectedToQueue()" style="font-size: 16px;">➕ Add Selected to Queue</button>
                    <button class="btn-outline" style="border-color: var(--danger); color: var(--danger);" onclick="clearQueue()">🗑 Clear Queue</button>
                    <button class="btn-success" style="margin-left: auto; font-size: 16px;" onclick="switchView('runner')">Go to Factory ➔</button>
                </div>
            </div>
        </div>
    </main>

    <!-- 3. LIVE FACTORY VIEW -->
    <main id="view-runner" class="view-panel">
        <div style="max-width: 1000px; margin: 0 auto; width: 100%;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px;">
                <h1>Let's Start Cooking</h1>
                <div style="display: flex; gap: 10px;">
                    <button id="btn-start" class="btn-success" onclick="startPipeline()">▶ START ENGINE</button>
                    <button id="btn-stop" class="btn-danger" onclick="stopPipeline()" disabled>⏹ STOP</button>
                </div>
            </div>

            <div class="card" style="text-align: center;">
                <div class="live-preview">
                    <img id="live-img" src="" alt="Awaiting Stream..." style="display:none;">
                    <span id="live-placeholder" style="color:#555;">Waiting for processing to start...</span>
                </div>

                <div style="display: flex; justify-content: space-between; color: #aaa; font-size: 14px; font-weight: bold;">
                    <span id="run-status">STATUS: IDLE</span>
                    <span id="run-eta" style="color: var(--accent);">ETA: --:--</span>
                    <span id="run-count">0 / 0</span>
                </div>
                
                <div class="progress-bg"><div id="run-progress" class="progress-fill"></div></div>
                <h3 id="run-file" style="color: #fff; margin-bottom: 0;">-</h3>
                <div style="font-size: 12px; color: var(--danger); margin-top: 5px;" id="run-errors">Errors: 0</div>
            </div>

            <h2>Terminal Logs</h2>
            <div id="logs"></div>
        </div>
    </main>

    <!-- 4. QA REVIEW VIEW -->
    <main id="view-qa" class="view-panel">
        <div class="canvas-container">
            <canvas id="qa-canvas"></canvas>
            <div id="qa-loader" style="position:absolute; background:rgba(0,0,0,0.8); color:white; padding: 20px; border-radius: 8px; display:none;">Loading...</div>
        </div>
        <aside>
            <h2>QA Station</h2>
            <div style="font-size: 12px; color: #aaa; margin-bottom: 10px;" id="qa-counter">0 / 0</div>
            <div id="qa-filename" style="font-weight: bold; word-break: break-all; margin-bottom: 20px;">No files to review.</div>
            
            <div style="background: #000; padding: 15px; border-radius: 6px; font-size: 13px; color: #ccc; line-height: 1.6;">
                <strong style="color:white;">Controls:</strong><br><br>
                <kbd style="background:#333; padding:2px 6px; border-radius:4px;">SPACE</kbd> Next Image<br>
                <kbd style="background:#333; padding:2px 6px; border-radius:4px;">B</kbd> Previous Image<br>
                <kbd style="background:#333; padding:2px 6px; border-radius:4px;">Click</kbd> Select Polygon<br>
                <kbd style="background:#333; padding:2px 6px; border-radius:4px;">DEL</kbd> Remove Polygon<br>
                <kbd style="background:#dc2626; padding:2px 6px; border-radius:4px;">⇧ + DEL</kbd> Trash File
            </div>
            
            <div style="display: flex; gap: 10px; margin-top: auto;">
                <button class="btn-outline" style="flex:1;" onclick="navigateQA(-1)">⬅ Prev</button>
                <button class="btn-success" style="flex:1;" onclick="navigateQA(1)">Next ➡</button>
            </div>
            <button class="btn-danger" style="margin-top: 10px;" onclick="deleteCurrentQA()">🗑 Trash Image</button>
            <button class="btn-outline" style="margin-top: 10px;" onclick="initQA()">🔄 Refresh List</button>
        </aside>
    </main>

    <div id="toast"></div>

    <script>
        // --- Globals & Utils ---
        let currentPath = '/content/drive/MyDrive';
        let yoloClasses = {};
        let eventSource = null;
        let selectedPaths = new Set(); 

        function showToast(msg, isErr=false) {
            const t = document.getElementById('toast');
            t.textContent = msg;
            t.style.borderLeftColor = isErr ? 'var(--danger)' : 'var(--success)';
            t.classList.add('show');
            setTimeout(() => t.classList.remove('show'), 3000);
        }

        function switchView(view) {
            document.querySelectorAll('.view-panel').forEach(p => p.classList.remove('active'));
            document.querySelectorAll('.nav-btn').forEach(b => b.classList.remove('active'));
            document.getElementById(`view-${view}`).classList.add('active');
            event.currentTarget.classList.add('active');
            
            if(view === 'explorer') loadGallery(currentPath);
            if(view === 'qa') initQA();
        }

        // --- CONFIG ---
        const dropzone = document.getElementById('dropzone');
        dropzone.ondragover = (e) => { e.preventDefault(); dropzone.classList.add('dragover'); };
        dropzone.ondragleave = () => dropzone.classList.remove('dragover');
        dropzone.ondrop = (e) => { e.preventDefault(); dropzone.classList.remove('dragover'); uploadModel(e.dataTransfer.files[0]); };

        async function uploadModel(file) {
            if(!file || !file.name.endsWith('.pt')) return showToast("Please upload a .pt file", true);
            document.getElementById('model-status').innerHTML = "Uploading and loading into GPU... ⏳";
            
            const fd = new FormData(); fd.append('file', file);
            try {
                const res = await fetch('/api/upload_model', { method: 'POST', body: fd });
                const data = await res.json();
                if(data.success) {
                    yoloClasses = data.classes;
                    document.getElementById('model-status').innerHTML = `✅ Loaded! Detected ${Object.keys(yoloClasses).length} classes.`;
                    renderClassToggles();
                } else throw new Error(data.error);
            } catch(e) { document.getElementById('model-status').innerHTML = `❌ Error: ${e.message}`; }
        }

        function renderClassToggles() {
            const cont = document.getElementById('class-toggles');
            cont.innerHTML = '';
            Object.entries(yoloClasses).forEach(([id, name]) => {
                const lbl = document.createElement('label');
                lbl.style.display = 'flex'; lbl.style.alignItems = 'center'; lbl.style.gap = '5px'; lbl.style.background = '#222'; lbl.style.padding = '5px 10px'; lbl.style.borderRadius = '4px'; lbl.style.cursor = 'pointer';
                lbl.innerHTML = `<input type="checkbox" value="${id}" class="cls-toggle" checked onchange="updateConfig()"> ${name}`;
                cont.appendChild(lbl);
            });
            updateConfig();
        }

        function updateConfig() {
            const conf = document.getElementById('inp-conf').value;
            document.getElementById('conf-val').innerText = conf;
            const active = Array.from(document.querySelectorAll('.cls-toggle:checked')).map(cb => parseInt(cb.value));
            fetch('/api/config', { method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify({ conf_threshold: parseFloat(conf), active_classes: active }) });
        }

        // --- EXPLORER ---
        async function loadGallery(path) {
            currentPath = path;
            document.getElementById('exp-path').innerText = path;
            const res = await fetch(`/api/gallery?path=${encodeURIComponent(path)}`);
            const items = await res.json();
            
            const list = document.getElementById('exp-list');
            list.innerHTML = '';
            
            if(path !== '/content/drive/MyDrive' && path !== '/content/drive') {
                const up = document.createElement('div'); up.className = 'exp-item';
                up.innerHTML = `📁 <b>.. (Up)</b>`;
                up.onclick = () => loadGallery(path.substring(0, path.lastIndexOf('/')));
                list.appendChild(up);
            }

            items.forEach(i => {
                const isChecked = selectedPaths.has(i.path) ? 'checked' : '';
                const div = document.createElement('div'); div.className = 'exp-item';
                div.innerHTML = `<input type="checkbox" value="${i.path}" ${isChecked} onchange="toggleSelection('${i.path}', this.checked)" onclick="event.stopPropagation()"> ${i.type === 'folder' ? '📁' : '🖼️'} ${i.name}`;
                if(i.type === 'folder') div.onclick = (e) => { if(e.target.tagName !== 'INPUT') loadGallery(i.path); };
                else div.onclick = () => div.querySelector('input').click();
                list.appendChild(div);
            });
            updateBulkBar();
        }

        function toggleSelection(path, isChecked) {
            if(isChecked) selectedPaths.add(path); else selectedPaths.delete(path);
            updateBulkBar();
        }

        function toggleSelectAll() {
            const checkboxes = document.querySelectorAll('#exp-list input[type="checkbox"]');
            const allChecked = Array.from(checkboxes).every(cb => cb.checked);
            checkboxes.forEach(cb => { if (cb.checked === allChecked) cb.click(); });
        }

        function updateBulkBar() {
            const bar = document.getElementById('bulk-action-bar');
            if(selectedPaths.size > 0) {
                bar.style.display = 'flex';
                document.getElementById('sel-count').innerText = selectedPaths.size;
                document.getElementById('btn-rename').style.display = selectedPaths.size === 1 ? 'block' : 'none';
            } else bar.style.display = 'none';
        }

        async function doFileOp(action, target) {
            const files = Array.from(selectedPaths);
            if(files.length === 0) return;
            try {
                const res = await fetch('/api/file_ops', { method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify({action, files, target}) });
                if((await res.json()).success) { selectedPaths.clear(); updateBulkBar(); loadGallery(currentPath); showToast(`${action} successful`); }
            } catch(e) { showToast("Error", true); }
        }

        function promptRename() {
            const oldPath = Array.from(selectedPaths)[0];
            const oldName = oldPath.split('/').pop();
            const newName = prompt("Rename to:", oldName);
            if(newName && newName !== oldName) doFileOp('rename', oldPath.substring(0, oldPath.lastIndexOf('/') + 1) + newName);
        }

        function promptMove() { const t = prompt("Destination:", currentPath); if(t) doFileOp('move', t); }
        function promptCopy() { const t = prompt("Destination:", currentPath); if(t) doFileOp('copy', t); }
        function bulkDelete() { if(confirm(`Delete ${selectedPaths.size} items?`)) doFileOp('delete', ''); }
        
        async function promptNewFolder() {
            const name = prompt("Folder Name:");
            if(!name) return;
            await fetch('/api/mkdir', { method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify({path: currentPath + '/' + name}) });
            loadGallery(currentPath);
        }

        async function uploadFiles(e) {
            const fd = new FormData(); fd.append('path', currentPath);
            for(let f of e.target.files) fd.append('files', f);
            showToast("Uploading...");
            await fetch('/api/upload_files', { method: 'POST', body: fd });
            loadGallery(currentPath); e.target.value = "";
        }

        async function uploadZip(e) {
            if(!e.target.files[0]) return;
            const fd = new FormData(); fd.append('path', currentPath); fd.append('zip', e.target.files[0]);
            showToast("Uploading & Extracting ZIP...");
            await fetch('/api/upload_zip', { method: 'POST', body: fd });
            loadGallery(currentPath); e.target.value = "";
        }

        async function addSelectedToQueue() {
            const paths = Array.from(selectedPaths);
            if(paths.length === 0) return showToast("Select items first", true);
            const res = await fetch('/api/queue/add', { method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify({paths}) });
            const data = await res.json();
            document.getElementById('queue-size').innerText = data.queue_size;
            showToast(`Added ${data.added} images to queue.`);
            selectedPaths.clear(); updateBulkBar();
            document.querySelectorAll('#exp-list input[type="checkbox"]').forEach(cb => cb.checked = false);
        }

        async function clearQueue() { await fetch('/api/queue/clear', { method: 'POST' }); document.getElementById('queue-size').innerText = 0; showToast("Queue cleared."); }

        // --- FACTORY ---
        async function startPipeline() {
            const res = await fetch('/api/pipeline/start', {method: 'POST'});
            const data = await res.json();
            if(!data.success) return showToast(data.error, true);
            
            document.getElementById('btn-start').disabled = true; document.getElementById('btn-stop').disabled = false;
            if(eventSource) eventSource.close();
            eventSource = new EventSource('/api/pipeline/status');
            
            eventSource.onmessage = (e) => {
                const state = JSON.parse(e.data);
                if(!state.is_running && state.total > 0 && state.current === state.total) { stopPipeline(); showToast("Pipeline Finished!"); }
                
                document.getElementById('run-status').innerText = state.is_running ? "STATUS: RUNNING" : "STATUS: IDLE";
                document.getElementById('run-count').innerText = `${state.current} / ${state.total}`;
                document.getElementById('run-progress').style.width = state.total ? `${(state.current/state.total)*100}%` : '0%';
                document.getElementById('run-file').innerText = state.current_file || "Scanning...";
                document.getElementById('run-errors').innerText = `Errors: ${state.errors}`;
                
                const m = Math.floor(state.eta_seconds / 60); const s = state.eta_seconds % 60;
                document.getElementById('run-eta').innerText = `ETA: ${m}m ${s}s`;
                
                if(state.preview_b64) {
                    document.getElementById('live-placeholder').style.display = 'none';
                    document.getElementById('live-img').style.display = 'block';
                    document.getElementById('live-img').src = "data:image/jpeg;base64," + state.preview_b64;
                }
                
                const logsDiv = document.getElementById('logs');
                logsDiv.innerHTML = state.logs.join('<br>'); logsDiv.scrollTop = logsDiv.scrollHeight;
            };
        }

        async function stopPipeline() {
            await fetch('/api/pipeline/stop', {method: 'POST'});
            if(eventSource) { eventSource.close(); eventSource = null; }
            document.getElementById('btn-start').disabled = false; document.getElementById('btn-stop').disabled = true;
            document.getElementById('run-status').innerText = "STATUS: STOPPED";
        }

        // --- QA STATION (CACHED) ---
        let qaFiles = []; 
        let qaIndex = 0; 
        let qaCache = {}; // { filename: dataObject }
        let qaSelectedPoly = -1;
        const canvas = document.getElementById('qa-canvas'); 
        const ctx = canvas.getContext('2d');
        const COLORS = ["#3b82f6", "#ef4444", "#22c55e", "#f59e0b", "#8b5cf6", "#ec4899", "#14b8a6"];
        
        const CACHE_BEHIND = 5;
        const CACHE_AHEAD = 15;

        async function initQA() {
            const res = await fetch('/api/qa/list');
            qaFiles = await res.json();
            qaIndex = 0; qaCache = {};
            if(qaFiles.length > 0) { renderQALoading(); preloadWindow(); }
            else { document.getElementById('qa-filename').innerText = "Folder empty."; ctx.clearRect(0,0,canvas.width,canvas.height); }
        }

        function renderQALoading() {
            document.getElementById('qa-loader').style.display = 'block';
            document.getElementById('qa-counter').innerText = `${qaIndex + 1} / ${qaFiles.length}`;
            document.getElementById('qa-filename').innerText = qaFiles[qaIndex] || "";
            ctx.clearRect(0,0,canvas.width,canvas.height);
        }

        function preloadWindow() {
            if(qaFiles.length === 0) return;
            let start = Math.max(0, qaIndex - CACHE_BEHIND);
            let end = Math.min(qaFiles.length - 1, qaIndex + CACHE_AHEAD);
            
            // Garbage Collect old cache
            for(let fname in qaCache) {
                let idx = qaFiles.indexOf(fname);
                if(idx < start || idx > end) delete qaCache[fname];
            }

            // Fetch Window
            for(let i=start; i<=end; i++) {
                let fname = qaFiles[i];
                if(!qaCache[fname]) {
                    qaCache[fname] = "loading";
                    fetch(`/api/qa/image/${fname}`).then(r=>r.json()).then(data => {
                        if(data.success) {
                            if(Object.keys(yoloClasses).length === 0) yoloClasses = data.classes || {};
                            
                            // Pre-create Image element
                            let img = new Image();
                            img.src = "data:image/jpeg;base64," + data.image_b64;
                            data.imgElement = img;
                            
                            qaCache[fname] = data;
                            if(i === qaIndex) drawQA(); // Render if it was waiting
                        } else delete qaCache[fname];
                    }).catch(()=> delete qaCache[fname]);
                }
            }
            
            if(qaCache[qaFiles[qaIndex]] !== "loading" && qaCache[qaFiles[qaIndex]]) drawQA();
        }

        function drawQA() {
            const fname = qaFiles[qaIndex];
            const data = qaCache[fname];
            if(!data || data === "loading") { renderQALoading(); return; }
            
            document.getElementById('qa-loader').style.display = 'none';
            document.getElementById('qa-counter').innerText = `${qaIndex + 1} / ${qaFiles.length}`;
            document.getElementById('qa-filename').innerText = fname;
            
            const img = data.imgElement;
            canvas.width = img.width; canvas.height = img.height;
            ctx.clearRect(0,0,canvas.width,canvas.height);
            ctx.drawImage(img, 0, 0);
            
            data.polygons.forEach((poly, idx) => {
                ctx.beginPath();
                poly.points.forEach((p, i) => {
                    const px = p.x * canvas.width; const py = p.y * canvas.height;
                    if(i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
                });
                ctx.closePath();
                
                const color = COLORS[poly.classId % COLORS.length];
                ctx.fillStyle = color + "66"; ctx.fill();
                
                ctx.lineWidth = idx === qaSelectedPoly ? 4 : 2;
                ctx.strokeStyle = idx === qaSelectedPoly ? "#FFF" : color;
                if(idx === qaSelectedPoly) ctx.setLineDash([5,5]); else ctx.setLineDash([]);
                ctx.stroke(); ctx.setLineDash([]);
                
                if(idx !== qaSelectedPoly) {
                    const px = poly.points[0].x * canvas.width; const py = poly.points[0].y * canvas.height;
                    ctx.fillStyle = color; ctx.fillRect(px, py - 20, Math.max(80, (yoloClasses[poly.classId]||"").length*8), 20);
                    ctx.fillStyle = "#FFF"; ctx.font = "bold 12px sans-serif"; ctx.fillText(yoloClasses[poly.classId] || `Class ${poly.classId}`, px + 5, py - 5);
                }
            });
        }

        async function navigateQA(dir) {
            if(qaFiles.length === 0) return;
            
            // Auto-Save current
            const fname = qaFiles[qaIndex];
            const data = qaCache[fname];
            if(data && data !== "loading") {
                fetch('/api/qa/save', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({filename: fname, polygons: data.polygons}) });
            }
            
            qaIndex += dir;
            if(qaIndex < 0) qaIndex = 0;
            if(qaIndex >= qaFiles.length) { qaIndex = qaFiles.length - 1; showToast("End of folder."); return; }
            
            qaSelectedPoly = -1;
            renderQALoading();
            preloadWindow();
        }

        async function deleteCurrentQA() {
            if(qaFiles.length === 0) return;
            const fname = qaFiles[qaIndex];
            await fetch('/api/qa/delete', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({filename: fname}) });
            
            qaFiles.splice(qaIndex, 1);
            delete qaCache[fname];
            
            if(qaIndex >= qaFiles.length) qaIndex = qaFiles.length - 1;
            
            if(qaFiles.length > 0) { renderQALoading(); preloadWindow(); }
            else { document.getElementById('qa-filename').innerText = "Empty."; ctx.clearRect(0,0,canvas.width,canvas.height); }
            showToast("Trashed.");
        }

        // QA Canvas Interaction
        canvas.addEventListener('mousedown', (e) => {
            const fname = qaFiles[qaIndex];
            if(!qaCache[fname] || qaCache[fname] === "loading") return;
            
            const rect = canvas.getBoundingClientRect();
            const scaleX = canvas.width / rect.width; const scaleY = canvas.height / rect.height;
            const x = (e.clientX - rect.left) * scaleX / canvas.width; 
            const y = (e.clientY - rect.top) * scaleY / canvas.height;
            
            qaSelectedPoly = -1;
            const polys = qaCache[fname].polygons;
            for(let i = polys.length - 1; i >= 0; i--) {
                if(pointInPolygon({x, y}, polys[i].points)) { qaSelectedPoly = i; break; }
            }
            drawQA();
        });

        window.addEventListener('keydown', (e) => {
            if(!document.getElementById('view-qa').classList.contains('active')) return;
            
            if(e.code === 'Space' || e.code === 'ArrowRight') { e.preventDefault(); navigateQA(1); }
            if(e.code === 'KeyB' || e.code === 'ArrowLeft') { e.preventDefault(); navigateQA(-1); }
            
            if(e.shiftKey && (e.code === 'Delete' || e.code === 'Backspace')) {
                e.preventDefault(); deleteCurrentQA();
            } else if(e.code === 'Delete' || e.code === 'Backspace') {
                e.preventDefault();
                const fname = qaFiles[qaIndex];
                if(qaSelectedPoly !== -1 && qaCache[fname] && qaCache[fname] !== "loading") {
                    qaCache[fname].polygons.splice(qaSelectedPoly, 1);
                    qaSelectedPoly = -1; drawQA();
                }
            }
        });

        function pointInPolygon(point, vs) {
            let x = point.x, y = point.y, inside = false;
            for (let i = 0, j = vs.length - 1; i < vs.length; j = i++) {
                let xi = vs[i].x, yi = vs[i].y, xj = vs[j].x, yj = vs[j].y;
                let intersect = ((yi > y) != (yj > y)) && (x < (xj - xi) * (y - yi) / (yj - yi) + xi);
                if (intersect) inside = !inside;
            }
            return inside;
        }

        // Start
        loadGallery(currentPath);
    </script>
</body>
</html>
```

---

### Cell 5: Start Server & Generate Public URL
*Note: Make sure to fill out your Colab Secrets before running this.*

```python
import subprocess
import time
import os
from pyngrok import ngrok
from google.colab import userdata

# 1. Authenticate with Ngrok and Hugging Face
ngrok_token = userdata.get('NGROK_TOKEN')
os.environ["HF_TOKEN"] = userdata.get('HF_TOKEN')
ngrok.set_auth_token(ngrok_token.strip())

# 2. Kill old processes to prevent port blocking
os.system("pkill -f -9 'app.py'")
ngrok.kill()
time.sleep(1)

# 3. Start Flask App in the background
print("Starting Engine Backend...")
log_file = open("engine_logs.txt", "w")
flask_process = subprocess.Popen(["python", "-u", "app.py"], stdout=log_file, stderr=subprocess.STDOUT)
time.sleep(5) # Let Flask boot

# 4. Open Ngrok Tunnel
public_url = ngrok.connect(addr="127.0.0.1:5000").public_url

print("="*70)
print(f"🚀 PIPELINE READY! Open Mission Control:")
print(f"👉 {public_url}")
print("="*70)
```