# SAM 3 Colab API Setup Instructions

This guide turns your Google Colab instance into an on-demand SAM 3 inference server accessible via a secure ngrok URL. It supports **both** Image Promptable Concept Segmentation (PCS) and Interactive Video Tracking.

### Prerequisites: Hugging Face Access (CRITICAL)

SAM 3 is a **gated model**. Before starting, you must agree to Meta's license:

1. Go to [https://huggingface.co/facebook/sam3](https://huggingface.co/facebook/sam3)
2. Log in and click the button to agree to the terms and request access (approval is usually instant).
3. Create an Access Token in your Hugging Face settings.

---

### Part 1: The Google Colab Setup (Server)

1. Open a new Google Colab notebook.
2. Go to **Runtime -> Change runtime type** and select **T4 GPU** (or L4/A100).
3. Open the **🔑 Secrets** tab on the left sidebar and add **two** secrets:
   - `HF_TOKEN`: Your Hugging Face access token.
   - `NGROK_TOKEN`: Your authtoken from [ngrok.com](https://dashboard.ngrok.com/get-started/your-authtoken).
   - _(Make sure "Notebook access" is toggled ON for both)._

#### Cell 1: Install Dependencies

Run this cell to clone the repository and install SAM 3 and Ngrok.

```bash
# 1. Clone SAM 3 repository and install
!git clone https://github.com/facebookresearch/sam3.git
%cd sam3
!pip install .

# 2. Install Flask API and Ngrok dependencies
!pip install flask flask-cors pillow pyngrok
```

⚠️ **IMPORTANT:** When Cell 1 finishes, scroll to the bottom of the output. Colab will likely show a button saying **"Restart Session"** (to apply required package downgrades). **Click "Restart Session" before proceeding to Cell 2.**

---

#### Cell 2: Create the Flask API App

_(Run this cell AFTER restarting the session)._

```python
import os
from google.colab import userdata

# Retrieve token from Colab Secrets
hf_token = userdata.get('HF_TOKEN')

app_code = f"""
import io
import os
import torch
import base64
import numpy as np
import traceback
import tempfile
import shutil
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image

# Inject Hugging Face Token for SAM 3 checkpoint downloads
os.environ["HF_TOKEN"] = "{hf_token}"

from sam3.model_builder import build_sam3_image_model, build_sam3_predictor
from sam3.model.sam3_image_processor import Sam3Processor

app = Flask(__name__)
CORS(app)

print("--> Loading SAM 3 Models (Image PCS & Video Tracker)...", flush=True)
device = "cuda" if torch.cuda.is_available() else "cpu"

# 1. Load the models WITHOUT passing dtype to prevent ruining complex tensors
image_model = build_sam3_image_model().to(device=device)
video_predictor = build_sam3_predictor(version="sam3.1").to(device=device)

# Patch the video predictor's hardcoded bfloat16 context
if hasattr(video_predictor, 'bf16_context'):
    try:
        video_predictor.bf16_context.__exit__(None, None, None)
    except Exception:
        pass
    video_predictor.bf16_context = torch.autocast(device_type="cuda", dtype=torch.float32)
    video_predictor.bf16_context.__enter__()

# 2. SURGICAL CASTING: Convert bfloat16/float16 to float32 safely
# This avoids BFloat16 hardware crashes on T4 GPUs while preserving complex RoPE tensors.
def make_fp32_safe(m):
    for name, param in m.named_parameters(recurse=False):
        if param.dtype in [torch.bfloat16, torch.float16]:
            param.data = param.data.to(torch.float32)
    for name, buf in m.named_buffers(recurse=False):
        if buf.dtype in [torch.bfloat16, torch.float16]:
            buf.data = buf.data.to(torch.float32)
    for k, v in m.__dict__.items():
        if isinstance(v, torch.Tensor) and v.dtype in [torch.bfloat16, torch.float16]:
            setattr(m, k, v.to(torch.float32))

for module in image_model.modules(): make_fp32_safe(module)
for module in video_predictor.model.modules(): make_fp32_safe(module)

image_processor = Sam3Processor(image_model)
if hasattr(image_processor, 'autocast'):
    image_processor.autocast = torch.autocast(device_type="cuda", dtype=torch.float32)

print("--> Models surgically purified to Float32! Complex RoPE tensors preserved.", flush=True)

# Global dict to track active video sessions and their temp directories
ACTIVE_SESSIONS = {{}}

def encode_masks(binary_masks):
    encoded = []
    for m in binary_masks:
        if m.ndim == 3 and m.shape[0] == 1:
            m = m[0]
        mask_img = Image.fromarray((m * 255).astype(np.uint8), mode='L')
        buf = io.BytesIO()
        mask_img.save(buf, format='PNG')
        encoded.append(base64.b64encode(buf.getvalue()).decode('utf-8'))
    return encoded

def format_video_outputs(outputs):
    boxes = outputs.get("out_boxes_xywh", [])
    if isinstance(boxes, np.ndarray): boxes = boxes.tolist()
    scores = outputs.get("out_probs", [])
    if isinstance(scores, np.ndarray): scores = scores.tolist()
    obj_ids = outputs.get("out_obj_ids", [])
    if isinstance(obj_ids, np.ndarray): obj_ids = obj_ids.tolist()
    masks = outputs.get("out_binary_masks", [])
    return {{"boxes": boxes, "scores": scores, "obj_ids": obj_ids, "masks_base64": encode_masks(masks)}}

# ==========================================
# IMAGE PCS API
# ==========================================
@app.route('/image/segment', methods=['POST'])
def segment_image():
    if 'image' not in request.files or 'prompt' not in request.form:
        return jsonify({{"error": "Missing 'image' or 'prompt' parameter"}}), 400
    try:
        image = Image.open(io.BytesIO(request.files['image'].read())).convert("RGB")
        prompt = request.form['prompt']

        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float32):
            inference_state = image_processor.set_image(image)
            output = image_processor.set_text_prompt(state=inference_state, prompt=prompt)

            masks = output["masks"].cpu().numpy()
            boxes = output["boxes"].tolist()
            scores = output["scores"].tolist()

        return jsonify({{"success": True, "boxes": boxes, "scores": scores, "masks_base64": encode_masks(masks)}})
    except Exception as e:
        print(f"ERROR: {{traceback.format_exc()}}", flush=True)
        return jsonify({{"success": False, "error": str(e)}}), 500

# ==========================================
# VIDEO TRACKING API
# ==========================================
@app.route('/video/start_session', methods=['POST'])
def start_session():
    if 'video' not in request.files: return jsonify({{"error": "Missing 'video' file"}}), 400
    try:
        video_file = request.files['video']
        temp_dir = tempfile.mkdtemp()
        video_path = os.path.join(temp_dir, video_file.filename)
        video_file.save(video_path)

        req = {{"type": "start_session", "resource_path": video_path}}
        resp = video_predictor.handle_request(req)

        session_id = resp["session_id"]
        ACTIVE_SESSIONS[session_id] = temp_dir
        return jsonify({{"success": True, "session_id": session_id}})
    except Exception as e:
        return jsonify({{"success": False, "error": str(e)}}), 500

@app.route('/video/add_prompt', methods=['POST'])
def add_prompt():
    data = request.get_json()
    if not data or 'session_id' not in data or 'frame_index' not in data:
        return jsonify({{"error": "Missing session_id or frame_index"}}), 400
    try:
        req = {{"type": "add_prompt", "session_id": data["session_id"], "frame_index": int(data["frame_index"])}}
        for key in ["text", "points", "point_labels", "bounding_boxes", "bounding_box_labels", "obj_id"]:
            if key in data: req[key] = data[key]

        resp = video_predictor.handle_request(req)
        formatted_out = format_video_outputs(resp["outputs"])
        return jsonify({{"success": True, "frame_index": resp["frame_index"], "outputs": formatted_out}})
    except Exception as e:
        return jsonify({{"success": False, "error": str(e)}}), 500

@app.route('/video/propagate', methods=['POST'])
def propagate():
    data = request.get_json()
    if not data or 'session_id' not in data:
        return jsonify({{"error": "Missing session_id"}}), 400
    try:
        req = {{"type": "propagate_in_video", "session_id": data["session_id"]}}
        results = []
        for out in video_predictor.handle_stream_request(req):
            results.append({{
                "frame_index": out["frame_index"],
                "outputs": format_video_outputs(out["outputs"])
            }})
        return jsonify({{"success": True, "results": results}})
    except Exception as e:
        return jsonify({{"success": False, "error": str(e)}}), 500

@app.route('/video/close_session', methods=['POST'])
def close_session():
    data = request.get_json()
    if not data or 'session_id' not in data: return jsonify({{"error": "Missing session_id"}}), 400
    try:
        sid = data["session_id"]
        resp = video_predictor.handle_request({{"type": "close_session", "session_id": sid}})
        if sid in ACTIVE_SESSIONS:
            shutil.rmtree(ACTIVE_SESSIONS[sid], ignore_errors=True)
            del ACTIVE_SESSIONS[sid]
        return jsonify({{"success": True, "gpu_mem": resp.get("gpu_mem")}})
    except Exception as e:
        return jsonify({{"success": False, "error": str(e)}}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
"""

with open("/content/app.py", "w") as f:
    f.write(app_code)
print("app.py written successfully!")
```

#### Cell 3: Start Server & Ngrok Tunnel

Run this cell to start your Flask API and expose it using your authenticated Ngrok tunnel.

```python
import subprocess
import time
import urllib.request
import urllib.error
import os
from pyngrok import ngrok
from google.colab import userdata

# 0. Kill any zombie Flask processes to free the port and clear old code from memory
os.system("pkill -f 'python -u /content/app.py'")
os.system("pkill -f 'python /content/app.py'")
time.sleep(1)

# 1. Start Flask in the background (-u makes it unbuffered for instant logging)
print("Starting Flask Server... (Logs piping to /content/flask_logs.txt)")
log_file = open("/content/flask_logs.txt", "w")
flask_process = subprocess.Popen(["python", "-u", "/content/app.py"], stdout=log_file, stderr=subprocess.STDOUT)

print("Waiting for SAM 3 model weights to load and Flask to boot (this may take up to a minute)...")

# 2. Smart loop to wait until Flask is actually responsive
flask_ready = False
for _ in range(300): # Wait up to 10 minutes
    if flask_process.poll() is not None:
        break
    try:
        urllib.request.urlopen("http://127.0.0.1:5000/image/segment")
        flask_ready = True
        break
    except urllib.error.HTTPError: # 405 Method Not Allowed is a good sign (it's alive!)
        flask_ready = True
        break
    except urllib.error.URLError:
        time.sleep(2)

# 3. Handle result
if not flask_ready:
    print("\n❌ ERROR: Flask server crashed or failed to start!")
    print("--- FLASK LOGS ---")
    with open("/content/flask_logs.txt", "r") as f:
        print(f.read())
else:
    print("\n✅ Flask server is fully online!")
    print("=== STARTING TUNNEL ===")

    ngrok_token = userdata.get('NGROK_TOKEN')
    ngrok.set_auth_token(ngrok_token)

    ngrok.kill()
    public_url = ngrok.connect(addr="127.0.0.1:5000").public_url
    print(f"\n🚀 YOUR API URL IS: {public_url} \n")
```

---

### Cleaning up and Debugging

Because the Flask server runs continuously in the background on Colab, you may occasionally need to check its logs or forcibly terminate it if you wish to rewrite the code. You can run these commands in any blank Colab cell:

**To read the server logs and see any errors:**

```python
with open("/content/flask_logs.txt", "r") as f:
    print(f.read())
```

**To manually kill the background Flask server (zombie processes):**

```bash
!pkill -f 'python -u /content/app.py'
!pkill -f 'python /content/app.py'
```
