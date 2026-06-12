# SAM 3 YOLO Auto-Labeler Setup Guide (Google Colab)

This guide turns your Google Colab instance into an on-demand SAM 3 inference server and annotation tool. It features a completely revamped, framework-less, lightning-fast WebUI inspired by brutalist web design principles. Google Drive acts as the single source of truth for your data and class ontology.

### Prerequisites:

1. **Hugging Face Token:** You must agree to the SAM 3 license at [https://huggingface.co/facebook/sam3](https://huggingface.co/facebook/sam3) and generate an Access Token.
2. **Ngrok Token:** Get your authtoken from [ngrok.com](https://dashboard.ngrok.com/get-started/your-authtoken).
3. **Colab Secrets:** In the left sidebar of Colab (🔑 icon), add `HF_TOKEN` and `NGROK_TOKEN`. Ensure "Notebook access" is toggled ON for both.

---

### Cell 1: Install Dependencies

Open a new Google Colab notebook, set the runtime to **T4 GPU**, and run this cell.
_(Note: Colab may prompt you to **Restart Session** at the bottom of the output after this runs. Please click it before proceeding)._

```bash
# Clone SAM 3 repository and install
!git clone https://github.com/facebookresearch/sam3.git
%cd sam3
!pip install -e .

# Install Flask, CORS, and Ngrok
!pip install flask flask-cors pyngrok opencv-python-headless
```

---

### Cell 2: Mount Google Drive

This connects your Google Drive so the dataset and configuration are saved permanently.

```python
from google.colab import drive
drive.mount('/content/drive')
print("✅ Google Drive mounted successfully!")
```

---

### Cell 3: Create the Frontend UI (`index.html`)

This generates a brutally fast, semantic HTML/JS frontend using a tabbed architecture. No React, no Vue, just raw browser performance.

```python
import os

os.makedirs("templates", exist_ok=True)

html_code = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SAM 3 Lightning Labeler</title>
    <style>
        :root { --bg: #121212; --panel: #1e1e1e; --text: #e0e0e0; --accent: #007bff; --danger: #dc3545; --success: #28a745; }
        body { margin: 0; font-family: system-ui, -apple-system, sans-serif; background: var(--bg); color: var(--text); display: flex; height: 100vh; overflow: hidden; }

        /* Navigation */
        nav { width: 60px; background: #000; display: flex; flex-direction: column; align-items: center; padding-top: 20px; gap: 30px; border-right: 1px solid #333;}
        .nav-btn { cursor: pointer; background: none; border: none; font-size: 24px; opacity: 0.4; transition: 0.2s; padding: 10px; border-radius: 8px;}
        .nav-btn.active, .nav-btn:hover { opacity: 1; background: #333; }

        /* Layouts */
        .view-panel { flex-grow: 1; display: none; overflow-y: auto; box-sizing: border-box; }
        .view-panel.active { display: flex; flex-direction: column; }
        .padded-view { padding: 30px; max-width: 1200px; margin: 0 auto; width: 100%; }

        /* Studio specific */
        #view-studio { flex-direction: row; padding: 0; }
        .canvas-container { flex-grow: 1; position: relative; background: #050505; display: flex; align-items: center; justify-content: center; overflow: hidden;}
        canvas { max-width: 100%; max-height: 100%; cursor: crosshair; }
        aside { width: 320px; background: var(--panel); padding: 20px; display: flex; flex-direction: column; gap: 15px; border-left: 1px solid #333; overflow-y: auto;}

        /* UI Elements */
        h2 { margin-top: 0; border-bottom: 2px solid #333; padding-bottom: 10px; }
        button { padding: 10px 15px; border: none; border-radius: 4px; font-weight: bold; cursor: pointer; background: #444; color: white; transition: 0.2s; }
        button:hover { filter: brightness(1.2); }
        button:disabled { opacity: 0.5; cursor: not-allowed; }
        .btn-accent { background: var(--accent); }
        .btn-success { background: var(--success); }
        .btn-danger { background: var(--danger); }
        input[type="text"], input[type="color"] { padding: 8px; background: #2a2a2a; border: 1px solid #444; color: white; border-radius: 4px; width: 100%; box-sizing: border-box;}

        /* Gallery */
        .gallery { display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 15px; margin-top: 20px;}
        .gallery-item { background: #2a2a2a; padding: 10px; border-radius: 6px; text-align: center; border: 1px solid #444; display: flex; flex-direction: column; gap: 10px;}
        .gallery-item span { font-size: 12px; word-break: break-all; }
        .badge { font-size: 10px; padding: 3px 6px; border-radius: 12px; font-weight: bold; }
        .badge.approved { background: #1e4620; color: #5cb85c; }
        .badge.pending { background: #463c1e; color: #f0ad4e; }

        /* Ontology Table */
        table { width: 100%; border-collapse: collapse; margin-top: 20px; }
        th, td { padding: 12px; text-align: left; border-bottom: 1px solid #333; }
        th { background: #222; }

        /* Loader & Toast */
        .loader-overlay { position: absolute; inset: 0; background: rgba(0,0,0,0.8); display: none; flex-direction: column; align-items: center; justify-content: center; z-index: 10; }
        .spinner { border: 4px solid #333; border-top: 4px solid var(--accent); border-radius: 50%; width: 40px; height: 40px; animation: spin 1s linear infinite; margin-bottom: 15px;}
        @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
        #toast { position: fixed; bottom: 20px; right: 20px; background: var(--panel); padding: 15px 20px; border-radius: 4px; box-shadow: 0 5px 15px rgba(0,0,0,0.5); transform: translateY(150%); transition: transform 0.3s cubic-bezier(0.175, 0.885, 0.32, 1.275); z-index: 1000; border-left: 5px solid var(--success); font-weight: bold;}
        #toast.show { transform: translateY(0); }

        /* Class selection buttons */
        .class-btn { display: flex; align-items: center; justify-content: space-between; text-align: left; background: #333; border: 1px solid #444; width: 100%; }
        .class-btn.selected { border: 2px solid white; background: #444; }
        .color-box { width: 16px; height: 16px; border-radius: 3px; display: inline-block; border: 1px solid rgba(255,255,255,0.2);}
    </style>
</head>
<body>

    <!-- Navigation -->
    <nav>
        <button class="nav-btn active" onclick="switchView('studio')" title="Annotation Studio">🎨</button>
        <button class="nav-btn" onclick="switchView('data')" title="Data Management">📁</button>
        <button class="nav-btn" onclick="switchView('ontology')" title="Classes & Prompts">🏷️</button>
    </nav>

    <!-- 1. Studio View -->
    <main id="view-studio" class="view-panel active">
        <div class="canvas-container">
            <canvas id="editorCanvas"></canvas>
            <div id="samLoader" class="loader-overlay">
                <div class="spinner"></div>
                <div style="font-weight:bold; letter-spacing: 1px;">SAM 3 INFERENCING...</div>
            </div>
        </div>
        <aside>
            <div style="background: #111; padding: 10px; border-radius: 6px; border: 1px solid #333;">
                <div style="font-size: 11px; color: #888; text-transform: uppercase;">Current File</div>
                <div id="currentFilename" style="font-weight: bold; word-break: break-all; margin-top: 5px;">No image loaded</div>
            </div>

            <div style="flex-grow: 1; overflow-y: auto;">
                <div style="font-size: 11px; color: #888; text-transform: uppercase; margin-bottom: 10px;">Ontology Classes</div>
                <div id="activeClassesList" style="display: flex; flex-direction: column; gap: 8px;"></div>
            </div>

            <div style="font-size: 12px; color: #aaa; background: #222; padding: 10px; border-radius: 4px;">
                🖱️ Click polygon to select<br>
                ⌨️ Backspace/Delete to remove
            </div>

            <hr style="border-color: #333; margin: 0;">
            <button onclick="runSAM3()" id="btnSam" class="btn-accent" style="padding: 15px;">🤖 Auto-Label (SAM 3)</button>
            <button onclick="clearPolygons()" class="btn-danger">🗑️ Clear Polygons</button>
            <button onclick="saveAndNext()" id="btnSave" class="btn-success" style="padding: 15px; margin-top: 10px;">💾 Save & Next</button>
        </aside>
    </main>

    <!-- 2. Data View -->
    <main id="view-data" class="view-panel">
        <div class="padded-view">
            <h2>Data Management</h2>
            <div style="display: flex; gap: 10px; background: var(--panel); padding: 15px; border-radius: 6px; border: 1px solid #333;">
                <div style="flex-grow: 1;">
                    <label style="font-size: 12px; color:#888;">Input Directory (Google Drive)</label>
                    <input type="text" id="inputDir" readonly style="opacity: 0.7;">
                </div>
                <div style="flex-grow: 1;">
                    <label style="font-size: 12px; color:#888;">Output Directory (Google Drive)</label>
                    <input type="text" id="outputDir" readonly style="opacity: 0.7;">
                </div>
            </div>

            <div style="display: flex; gap: 10px; margin-top: 20px;">
                <input type="file" id="fileUpload" multiple accept="image/*" style="display: none;" onchange="handleUpload(event)">
                <button onclick="document.getElementById('fileUpload').click()" class="btn-accent">📤 Upload Local Images</button>
                <button onclick="refreshGallery()">🔄 Refresh Directory</button>
            </div>

            <div class="gallery" id="galleryContainer"></div>
        </div>
    </main>

    <!-- 3. Ontology View -->
    <main id="view-ontology" class="view-panel">
        <div class="padded-view">
            <h2>Class Ontology & Prompts</h2>
            <p style="color: #aaa; font-size: 14px;">Define YOLO class IDs, display names, and the descriptive text prompts fed into SAM 3.</p>

            <div style="background: var(--panel); border: 1px solid #333; border-radius: 6px; overflow: hidden;">
                <table>
                    <thead>
                        <tr>
                            <th style="width: 60px;">ID</th>
                            <th>Display Name</th>
                            <th>SAM 3 Text Prompt</th>
                            <th style="width: 80px;">Color</th>
                            <th style="width: 80px;">Action</th>
                        </tr>
                    </thead>
                    <tbody id="ontologyTableBody"></tbody>
                </table>
            </div>
            <div style="display: flex; justify-content: space-between; margin-top: 20px;">
                <button onclick="addNewClass()">+ Add New Class</button>
                <button onclick="saveConfig()" class="btn-success" style="padding: 10px 30px;">💾 Save Configuration</button>
            </div>
        </div>
    </main>

    <div id="toast"></div>

    <script>
        // --- State Management ---
        const AppState = {
            config: { input_dir: '', output_dir: '', classes: [] },
            gallery: [],
            currentFilename: null,
            imageObj: null,
            polygons: [],
            selectedPolyIndex: -1
        };

        const HEADERS = { 'ngrok-skip-browser-warning': 'true' };
        const HEADERS_JSON = { 'Content-Type': 'application/json', ...HEADERS };

        // --- Snappy UI Utilities ---
        function showToast(msg, type = "success") {
            const toast = document.getElementById('toast');
            toast.textContent = msg;
            toast.style.borderLeftColor = type === 'error' ? 'var(--danger)' : 'var(--success)';
            toast.classList.add('show');
            setTimeout(() => toast.classList.remove('show'), 3000);
        }

        function switchView(viewName) {
            document.querySelectorAll('.view-panel').forEach(p => p.classList.remove('active'));
            document.querySelectorAll('.nav-btn').forEach(b => b.classList.remove('active'));
            document.getElementById(`view-${viewName}`).classList.add('active');
            event.currentTarget.classList.add('active');

            if(viewName === 'data') refreshGallery();
            if(viewName === 'ontology') renderOntology();
            if(viewName === 'studio') renderStudioClasses();
        }

        // --- Initialization ---
        async function initApp() {
            try {
                const res = await fetch('/api/config', { headers: HEADERS });
                AppState.config = await res.json();

                document.getElementById('inputDir').value = AppState.config.input_dir;
                document.getElementById('outputDir').value = AppState.config.output_dir;

                renderStudioClasses();
                await refreshGallery(false); // Silent refresh
                loadNextPendingImage();
            } catch (e) {
                showToast("Failed to connect to backend. Is Flask running?", "error");
            }
        }

        // --- Data & Gallery Logic ---
        async function refreshGallery(showMsg = true) {
            try {
                const res = await fetch('/api/gallery', { headers: HEADERS });
                const data = await res.json();
                AppState.gallery = data.images;

                const container = document.getElementById('galleryContainer');
                container.innerHTML = '';

                data.images.forEach(img => {
                    const isApproved = img.status === 'approved';
                    const div = document.createElement('div');
                    div.className = 'gallery-item';
                    div.id = `card-${img.filename}`;
                    div.innerHTML = `
                        <div style="display:flex; justify-content:space-between; align-items:flex-start;">
                            <span class="badge ${isApproved ? 'approved' : 'pending'}">${img.status.toUpperCase()}</span>
                            <button onclick="deleteImage('${img.filename}')" style="padding:4px 8px; background:transparent; color:#ff4444; border:1px solid #ff4444;" title="Delete Image">🗑️</button>
                        </div>
                        <span title="${img.filename}">${img.filename.length > 20 ? img.filename.substring(0,17)+'...' : img.filename}</span>
                        <button onclick="loadImageIntoStudio('${img.filename}')" style="background:#333;">Open in Studio</button>
                    `;
                    container.appendChild(div);
                });
                if(showMsg) showToast("Gallery refreshed");
            } catch (e) {
                showToast("Failed to load gallery", "error");
            }
        }

        async function handleUpload(e) {
            const files = e.target.files;
            if(files.length === 0) return;

            const formData = new FormData();
            for(let i=0; i<files.length; i++) formData.append('images', files[i]);

            showToast(`Uploading ${files.length} images...`);
            try {
                const res = await fetch('/api/upload', { method: 'POST', headers: HEADERS, body: formData });
                const data = await res.json();
                if(data.success) {
                    showToast(`Successfully uploaded ${data.uploaded} images!`);
                    refreshGallery(false);
                } else throw new Error(data.error);
            } catch(err) {
                showToast(err.message, "error");
            }
            e.target.value = ""; // Reset
        }

        async function deleteImage(filename) {
            // Optimistic UI update
            const card = document.getElementById(`card-${filename}`);
            if(card) card.style.display = 'none';

            try {
                const res = await fetch(`/api/image/${filename}`, { method: 'DELETE', headers: HEADERS });
                if (!res.ok) throw new Error("Delete failed");
                showToast(`Deleted ${filename}`);
                // Remove from local state
                AppState.gallery = AppState.gallery.filter(i => i.filename !== filename);
                if(AppState.currentFilename === filename) loadNextPendingImage();
            } catch (e) {
                if(card) card.style.display = 'flex'; // Revert
                showToast(e.message, "error");
            }
        }

        // --- Ontology Logic ---
        function renderOntology() {
            const tbody = document.getElementById('ontologyTableBody');
            tbody.innerHTML = '';
            AppState.config.classes.forEach((cls, index) => {
                const tr = document.createElement('tr');
                tr.innerHTML = `
                    <td><input type="text" id="cls-id-${index}" value="${cls.id}" style="width:50px; text-align:center;"></td>
                    <td><input type="text" id="cls-name-${index}" value="${cls.name}"></td>
                    <td><input type="text" id="cls-prompt-${index}" value="${cls.prompt}"></td>
                    <td><input type="color" id="cls-color-${index}" value="${cls.color}" style="height:35px; padding:2px;"></td>
                    <td><button onclick="removeOntologyRow(${index})" class="btn-danger" style="padding:8px;">🗑️</button></td>
                `;
                tbody.appendChild(tr);
            });
        }

        function addNewClass() {
            const nextId = AppState.config.classes.length > 0 ? Math.max(...AppState.config.classes.map(c => c.id)) + 1 : 0;
            AppState.config.classes.push({ id: nextId, name: "New Class", prompt: "describe it here", color: "#ffffff" });
            renderOntology();
        }

        function removeOntologyRow(index) {
            AppState.config.classes.splice(index, 1);
            renderOntology();
        }

        async function saveConfig() {
            const newClasses = [];
            for(let i=0; i<AppState.config.classes.length; i++) {
                newClasses.push({
                    id: parseInt(document.getElementById(`cls-id-${i}`).value),
                    name: document.getElementById(`cls-name-${i}`).value,
                    prompt: document.getElementById(`cls-prompt-${i}`).value,
                    color: document.getElementById(`cls-color-${i}`).value
                });
            }
            AppState.config.classes = newClasses;

            try {
                await fetch('/api/config', { method: 'POST', headers: HEADERS_JSON, body: JSON.stringify(AppState.config) });
                showToast("Configuration saved!");
                renderStudioClasses();
            } catch(e) {
                showToast("Failed to save config", "error");
            }
        }

        // --- Studio & Canvas Logic ---
        const canvas = document.getElementById('editorCanvas');
        const ctx = canvas.getContext('2d');

        function renderStudioClasses() {
            const container = document.getElementById('activeClassesList');
            container.innerHTML = '';
            AppState.config.classes.forEach(cls => {
                const btn = document.createElement('button');
                btn.className = 'class-btn';
                btn.innerHTML = `<span style="display:flex; align-items:center; gap:10px;"><div class="color-box" style="background-color: ${cls.color}"></div> ${cls.name}</span> <span style="opacity:0.5; font-size:10px;">ID:${cls.id}</span>`;
                btn.onclick = () => {
                    if (AppState.selectedPolyIndex !== -1) {
                        AppState.polygons[AppState.selectedPolyIndex].classId = cls.id;
                        drawCanvas();
                    }
                };
                container.appendChild(btn);
            });
        }

        async function loadImageIntoStudio(filename) {
            switchView('studio');
            document.getElementById('currentFilename').innerText = "Loading...";

            try {
                const res = await fetch(`/api/image/${filename}/data`, { headers: HEADERS });
                const data = await res.json();

                if (!data.success) throw new Error(data.error);

                AppState.currentFilename = filename;
                document.getElementById('currentFilename').innerText = filename;

                AppState.imageObj = new Image();
                AppState.imageObj.onload = () => {
                    canvas.width = AppState.imageObj.width;
                    canvas.height = AppState.imageObj.height;
                    AppState.polygons = data.annotations || [];
                    AppState.selectedPolyIndex = -1;
                    drawCanvas();
                };
                AppState.imageObj.src = "data:image/jpeg;base64," + data.image_b64;
            } catch(e) {
                showToast(e.message, "error");
                document.getElementById('currentFilename').innerText = "Error loading image";
            }
        }

        function loadNextPendingImage() {
            const pending = AppState.gallery.find(img => img.status === 'pending');
            if (pending) {
                loadImageIntoStudio(pending.filename);
            } else {
                document.getElementById('currentFilename').innerText = "🎉 All caught up!";
                AppState.currentFilename = null;
                ctx.clearRect(0, 0, canvas.width, canvas.height);
            }
        }

        async function runSAM3() {
            if (!AppState.currentFilename || AppState.config.classes.length === 0) return;

            document.getElementById('btnSam').disabled = true;
            document.getElementById('btnSave').disabled = true;
            document.getElementById('samLoader').style.display = 'flex';

            try {
                const res = await fetch('/api/auto_label', {
                    method: 'POST', headers: HEADERS_JSON,
                    body: JSON.stringify({ filename: AppState.currentFilename, prompts: AppState.config.classes })
                });
                const data = await res.json();
                if (data.success) {
                    AppState.polygons.push(...data.polygons);
                    drawCanvas();
                    showToast("SAM 3 labeling complete!");
                } else throw new Error(data.error);
            } catch(e) {
                showToast(e.message, "error");
            } finally {
                document.getElementById('btnSam').disabled = false;
                document.getElementById('btnSave').disabled = false;
                document.getElementById('samLoader').style.display = 'none';
            }
        }

        async function saveAndNext() {
            if (!AppState.currentFilename) return;

            try {
                const res = await fetch('/api/save', {
                    method: 'POST', headers: HEADERS_JSON,
                    body: JSON.stringify({
                        filename: AppState.currentFilename,
                        annotations: AppState.polygons
                    })
                });
                const data = await res.json();
                if (data.success) {
                    showToast("Saved successfully!");
                    // Update local gallery state
                    const img = AppState.gallery.find(i => i.filename === AppState.currentFilename);
                    if(img) img.status = 'approved';

                    loadNextPendingImage();
                } else throw new Error(data.error);
            } catch(e) {
                showToast(e.message, "error");
            }
        }

        function clearPolygons() {
            AppState.polygons = [];
            AppState.selectedPolyIndex = -1;
            drawCanvas();
        }

        // Canvas Interaction
        canvas.addEventListener('mousedown', (e) => {
            const rect = canvas.getBoundingClientRect();
            // Scale mouse coordinates to actual canvas resolution
            const scaleX = canvas.width / rect.width;
            const scaleY = canvas.height / rect.height;
            const x = (e.clientX - rect.left) * scaleX / canvas.width;
            const y = (e.clientY - rect.top) * scaleY / canvas.height;

            AppState.selectedPolyIndex = -1;
            for(let i = AppState.polygons.length - 1; i >= 0; i--) {
                if(pointInPolygon({x, y}, AppState.polygons[i].points)) {
                    AppState.selectedPolyIndex = i;
                    break;
                }
            }
            drawCanvas();
        });

        window.addEventListener('keydown', (e) => {
            if ((e.key === 'Delete' || e.key === 'Backspace') && AppState.selectedPolyIndex !== -1) {
                AppState.polygons.splice(AppState.selectedPolyIndex, 1);
                AppState.selectedPolyIndex = -1;
                drawCanvas();
            }
        });

        function pointInPolygon(point, vs) {
            let x = point.x, y = point.y;
            let inside = false;
            for (let i = 0, j = vs.length - 1; i < vs.length; j = i++) {
                let xi = vs[i].x, yi = vs[i].y;
                let xj = vs[j].x, yj = vs[j].y;
                let intersect = ((yi > y) != (yj > y)) && (x < (xj - xi) * (y - yi) / (yj - yi) + xi);
                if (intersect) inside = !inside;
            }
            return inside;
        }

        function drawCanvas() {
            if (!AppState.imageObj) return;
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            ctx.drawImage(AppState.imageObj, 0, 0, canvas.width, canvas.height);

            AppState.polygons.forEach((poly, idx) => {
                ctx.beginPath();
                poly.points.forEach((p, i) => {
                    const px = p.x * canvas.width;
                    const py = p.y * canvas.height;
                    if (i === 0) ctx.moveTo(px, py);
                    else ctx.lineTo(px, py);
                });
                ctx.closePath();

                const cls = AppState.config.classes.find(c => c.id === poly.classId);
                const color = cls ? cls.color : '#ffffff';

                ctx.fillStyle = color + '66'; // 40% opacity hex
                ctx.fill();

                ctx.lineWidth = idx === AppState.selectedPolyIndex ? 6 : 2;
                ctx.strokeStyle = idx === AppState.selectedPolyIndex ? '#ffffff' : color;
                ctx.stroke();
            });
        }

        window.onload = initApp;
    </script>
</body>
</html>
"""

with open("templates/index.html", "w") as f:
    f.write(html_code)
print("✅ Semantic Frontend template generated!")
```

---

### Cell 4: Create the Flask Backend (`app.py`)

This backend eliminates messy state tracking. Google Drive folders dictates what is pending vs approved, and the `config.json` dictates the YOLO classes.

```python
import os
import json
import glob
import shutil
import base64
import torch
import cv2
import numpy as np
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from PIL import Image

app = Flask(__name__)
CORS(app)

print("--> Loading SAM 3 Model (This takes a minute)...", flush=True)

# 1. Load model with EXPLICIT bpe_path to avoid pkg_resources Colab bug
BPE_PATH = "/content/sam3/sam3/assets/bpe_simple_vocab_16e6.txt.gz"
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

model = build_sam3_image_model(bpe_path=BPE_PATH).cuda().eval()

# Patch BFloat16 to Float32 for Colab T4 hardware
def make_fp32_safe(m):
    for name, param in m.named_parameters(recurse=False):
        if param.dtype in [torch.bfloat16, torch.float16]:
            param.data = param.data.to(torch.float32)
    for name, buf in m.named_buffers(recurse=False):
        if buf.dtype in [torch.bfloat16, torch.float16]:
            buf.data = buf.data.to(torch.float32)

for m in model.modules(): make_fp32_safe(m)

processor = Sam3Processor(model)
if hasattr(processor, 'autocast'):
    processor.autocast = torch.autocast(device_type="cuda", dtype=torch.float32)

print("--> Model loaded successfully!", flush=True)

# --- Configuration & State ---
CONFIG_FILE = "/content/drive/MyDrive/SAM3_YOLO/config.json"
DEFAULT_CONFIG = {
    "input_dir": "/content/drive/MyDrive/SAM3_YOLO/raw_images",
    "output_dir": "/content/drive/MyDrive/SAM3_YOLO/dataset",
    "classes": [
        {"id": 0, "name": "Crack", "prompt": "crack on asphalt road", "color": "#ff0000"}
    ]
}

def load_config():
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, 'r') as f:
            return json.load(f)
    return DEFAULT_CONFIG

def save_config(cfg):
    os.makedirs(os.path.dirname(CONFIG_FILE), exist_ok=True)
    with open(CONFIG_FILE, 'w') as f:
        json.dump(cfg, f, indent=4)

def setup_dirs(cfg):
    os.makedirs(cfg['input_dir'], exist_ok=True)
    os.makedirs(os.path.join(cfg['output_dir'], "images"), exist_ok=True)
    os.makedirs(os.path.join(cfg['output_dir'], "labels"), exist_ok=True)

def mask_to_yolo_polygons(binary_mask):
    binary_mask = np.squeeze(binary_mask)
    if binary_mask.ndim != 2: return []
    h, w = binary_mask.shape
    mask_uint8 = np.ascontiguousarray((binary_mask * 255).astype(np.uint8))
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    polygons = []
    for contour in contours:
        epsilon = 0.002 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        if len(approx) < 3: continue
        points = [{"x": float(pt[0][0]) / w, "y": float(pt[0][1]) / h} for pt in approx]
        polygons.append(points)
    return polygons

def parse_yolo_txt(txt_path):
    polygons = []
    if os.path.exists(txt_path):
        with open(txt_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 7: # class_id + at least 3 points (6 coords)
                    cls_id = int(parts[0])
                    pts = [{"x": float(parts[i]), "y": float(parts[i+1])} for i in range(1, len(parts), 2)]
                    polygons.append({"classId": cls_id, "points": pts})
    return polygons

# --- API Routes ---
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/config', methods=['GET', 'POST'])
def api_config():
    if request.method == 'GET':
        cfg = load_config()
        setup_dirs(cfg)
        return jsonify(cfg)
    else:
        new_cfg = request.json
        save_config(new_cfg)
        setup_dirs(new_cfg)
        return jsonify({"success": True})

@app.route('/api/gallery', methods=['GET'])
def api_gallery():
    cfg = load_config()
    images = []
    for f in sorted(glob.glob(os.path.join(cfg['input_dir'], "*.*"))):
        if f.lower().endswith(('.png', '.jpg', '.jpeg')):
            fname = os.path.basename(f)
            txt_path = os.path.join(cfg['output_dir'], "labels", os.path.splitext(fname)[0] + ".txt")
            status = "approved" if os.path.exists(txt_path) else "pending"
            images.append({"filename": fname, "status": status})
    return jsonify({"success": True, "images": images})

@app.route('/api/upload', methods=['POST'])
def api_upload():
    cfg = load_config()
    uploaded_files = request.files.getlist('images')
    count = 0
    for file in uploaded_files:
        if file.filename == '': continue
        file.save(os.path.join(cfg['input_dir'], file.filename))
        count += 1
    return jsonify({"success": True, "uploaded": count})

@app.route('/api/image/<filename>', methods=['DELETE'])
def api_delete(filename):
    cfg = load_config()
    img_path = os.path.join(cfg['input_dir'], filename)
    txt_path = os.path.join(cfg['output_dir'], "labels", os.path.splitext(filename)[0] + ".txt")
    out_img_path = os.path.join(cfg['output_dir'], "images", filename)

    if os.path.exists(img_path): os.remove(img_path)
    if os.path.exists(txt_path): os.remove(txt_path)
    if os.path.exists(out_img_path): os.remove(out_img_path)

    return jsonify({"success": True})

@app.route('/api/image/<filename>/data', methods=['GET'])
def api_image_data(filename):
    cfg = load_config()
    img_path = os.path.join(cfg['input_dir'], filename)
    txt_path = os.path.join(cfg['output_dir'], "labels", os.path.splitext(filename)[0] + ".txt")

    if not os.path.exists(img_path):
        return jsonify({"success": False, "error": "File not found"})

    with open(img_path, "rb") as img_file:
        b64_string = base64.b64encode(img_file.read()).decode('utf-8')

    annotations = parse_yolo_txt(txt_path)
    return jsonify({"success": True, "image_b64": b64_string, "annotations": annotations})

@app.route('/api/auto_label', methods=['POST'])
def api_auto_label():
    data = request.json
    fname = data['filename']
    prompts = data['prompts']

    cfg = load_config()
    img_path = os.path.join(cfg['input_dir'], fname)

    try:
        image = Image.open(img_path).convert("RGB")
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float32):
            inference_state = processor.set_image(image)
            results = []

            for cls in prompts:
                output = processor.set_text_prompt(state=inference_state, prompt=cls['prompt'])
                masks = output["masks"].cpu().numpy()
                scores = output["scores"].cpu().numpy()

                for i, mask in enumerate(masks):
                    if scores[i] < 0.50: continue
                    polys = mask_to_yolo_polygons(mask)
                    for p in polys:
                        results.append({"classId": cls['id'], "points": p})

        torch.cuda.empty_cache()
        return jsonify({"success": True, "polygons": results})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route('/api/save', methods=['POST'])
def api_save():
    data = request.json
    fname = data['filename']
    annotations = data['annotations']

    cfg = load_config()
    src_img = os.path.join(cfg['input_dir'], fname)
    dst_img = os.path.join(cfg['output_dir'], "images", fname)
    txt_path = os.path.join(cfg['output_dir'], "labels", os.path.splitext(fname)[0] + ".txt")

    try:
        # Copy image to YOLO directory
        if not os.path.exists(dst_img) and os.path.exists(src_img):
            shutil.copy(src_img, dst_img)

        # Write YOLO segmentation format (.txt)
        with open(txt_path, 'w') as f:
            for ann in annotations:
                class_id = ann['classId']
                points_str = " ".join([f"{pt['x']:.6f} {pt['y']:.6f}" for pt in ann['points']])
                f.write(f"{class_id} {points_str}\\n")

        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
"""

with open("app.py", "w") as f:
    f.write(app_code)
print("✅ Robust Backend server generated!")
```

---

### Cell 5: Start the Server & Open the UI

This cell authenticates Ngrok, kills old stray processes, starts Flask in the background, and provides your secure public URL.

```python
import subprocess
import time
import getpass
import os
from pyngrok import ngrok
from google.colab import userdata

# 1. Authenticate with Ngrok and Hugging Face
try:
    ngrok_token = userdata.get('NGROK_TOKEN')
    os.environ["HF_TOKEN"] = userdata.get('HF_TOKEN')
except Exception:
    print("Colab secrets not found. Please paste manually:")
    ngrok_token = getpass.getpass("Ngrok Authtoken: ")
    os.environ["HF_TOKEN"] = getpass.getpass("Hugging Face Token: ")

ngrok.set_auth_token(ngrok_token.strip())

# 2. Kill old processes to prevent port blocking
os.system("pkill -f -9 'app.py'")
ngrok.kill()
time.sleep(1)

# 3. Start Flask App in the background
print("Starting Flask Server... (Logs piping to flask_logs.txt)")
log_file = open("flask_logs.txt", "w")
flask_process = subprocess.Popen(["python", "-u", "app.py"], stdout=log_file, stderr=subprocess.STDOUT)

# 4. Wait for server to boot (SAM 3 takes about 30 seconds to load into VRAM)
print("Loading SAM 3 model weights... Please wait.")
time.sleep(20) # Conservative wait

# 5. Open Ngrok Tunnel
public_url = ngrok.connect(addr="127.0.0.1:5000").public_url

print("="*60)
print(f"✅ READY! Click the link below to open the Annotation UI:")
print(f"👉 {public_url}")
print("="*60)
print("Note: If the page doesn't load immediately, SAM 3 is still booting. Wait 15 seconds and refresh.")
```
