import os
import time
import json
import base64
import cv2
import threading
from pathlib import Path
from flask import Flask, request, jsonify, render_template, Response, send_file
from werkzeug.utils import secure_filename
from ultralytics import YOLO

from constants import ALLOWED_IMAGE_EXT
from utils import safe_float
from parser_exif import extract_full_photo_metadata
from pipeline_image import generate_grid_preview, recalculate_view
from exports import create_raw_zip, create_flat_zip
from task_manager import start_processing_job, active_tasks, cancel_flags

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent

app = Flask(__name__, static_folder='../static', template_folder='../templates')
app.config['UPLOAD_FOLDER'] = os.path.join(PROJECT_ROOT, 'static', 'uploads')
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

global_model = None
model_lock = threading.Lock()

def handle_model_upload(request_obj):
    global global_model
    if 'model' in request_obj.files and request_obj.files['model'].filename != '':
        model_file = request_obj.files['model']
        model_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(model_file.filename))
        model_file.save(model_path)
        global_model = YOLO(model_path)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process():
    handle_model_upload(request)
    if global_model is None: return jsonify({"error": "No ML model loaded into memory"}), 400
    img_files = request.files.getlist('images')
    if not img_files or img_files[0].filename == '': return jsonify({"error": "No media selected"}), 400

    options = {
        "cam_height": safe_float(request.form.get('cam_height'), 1.6),
        "interval_m": safe_float(request.form.get('interval_m'), 2.0),
        "is_360": request.form.get('is_360') == 'true',
        "draw_grid": request.form.get('draw_grid') == 'true',
        "comp_roll": request.form.get('comp_roll') == 'true',
        "comp_pitch": request.form.get('comp_pitch') == 'true',
        "undistort": request.form.get('undistort') == 'true',
        "ego_mask": request.form.get('ego_mask') == 'true',
        "conf_thresh": safe_float(request.form.get('conf_thresh'), 0.25)
    }

    last_lat, last_lon = safe_float(request.form.get('last_lat'), None), safe_float(request.form.get('last_lon'), None)
    loc_id = int(request.form.get('last_loc_id', 1))

    image_data = []
    for f in img_files:
        ext = os.path.splitext(f.filename)[1].lower()
        filename = f"{int(time.time()*100)}_{secure_filename(f.filename)}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        f.save(filepath)
        file_meta = {"filename": filename, "original_name": f.filename, "path": filepath, "ext": ext, "lat": None, "lon": None, "pitch": None, "roll": None, "klns": None, "fov": None}
        if ext in ALLOWED_IMAGE_EXT:
            lat, lon, dynamic_pitch, dynamic_roll, klns, fov_meta, full_meta = extract_full_photo_metadata(filepath)
            file_meta.update({"lat": lat, "lon": lon, "pitch": dynamic_pitch, "roll": dynamic_roll, "klns": klns, "fov": fov_meta})
            with open(os.path.join(app.config['UPLOAD_FOLDER'], f"meta_{filename}.json"), 'w') as mf: json.dump(full_meta, mf, indent=2)
        image_data.append(file_meta)
    
    res = start_processing_job(image_data, options, last_lat, last_lon, loc_id, app.config['UPLOAD_FOLDER'], global_model, model_lock)
    return jsonify(res)

@app.route('/stream/<task_id>')
def stream(task_id):
    def event_stream():
        q = active_tasks.get(task_id)
        if not q: yield f"data: {json.dumps({'type': 'error', 'message': 'Invalid Task ID'})}\n\n"; return
        while True:
            msg = q.get()
            yield f"data: {json.dumps(msg)}\n\n"
            if msg['type'] in ['complete', 'error', 'cancelled']:
                cancel_flags.pop(task_id, None)
                active_tasks.pop(task_id, None)
                break
    return Response(event_stream(), mimetype="text/event-stream")

@app.route('/cancel/<task_id>', methods=['POST'])
def cancel_task(task_id):
    if task_id in cancel_flags:
        cancel_flags[task_id] = True
        return jsonify({"success": True})
    return jsonify({"error": "Task not found"}), 404

@app.route('/preview_grid', methods=['POST'])
def preview_grid():
    data = request.json
    filename = data['filename']
    view_name = data['view']
    pitch_offset = float(data['pitch_offset'])
    
    meta_path = os.path.join(app.config['UPLOAD_FOLDER'], f"process_meta_{filename}.json")
    if not os.path.exists(meta_path): return jsonify({"error": "Process metadata not found"}), 404
    
    with open(meta_path, 'r') as f: process_meta = json.load(f)
    raw_rect_filename = f"raw_rect_{view_name}_{filename}"
    raw_rect_path = os.path.join(app.config['UPLOAD_FOLDER'], raw_rect_filename)
    
    if not os.path.exists(raw_rect_path): return jsonify({"error": "Raw view image missing"}), 404
    
    preview_bgr = generate_grid_preview(raw_rect_path, process_meta, view_name, pitch_offset)
    _, buffer = cv2.imencode('.jpg', preview_bgr)
    preview_b64 = base64.b64encode(buffer).decode('utf-8')
    return jsonify({"success": True, "image": f"data:image/jpeg;base64,{preview_b64}"})

@app.route('/recalculate_bev', methods=['POST'])
def recalculate_bev():
    data = request.json
    pitch_offset = float(data['pitch_offset'])
    results = data['results']
    
    new_results = []
    for r in results:
        filename = r['filename']
        meta_path = os.path.join(app.config['UPLOAD_FOLDER'], f"process_meta_{filename}.json")
        if not os.path.exists(meta_path): continue
        
        with open(meta_path, 'r') as f: process_meta = json.load(f)
        
        updated_r = r.copy()
        updated_r['geojson'] = []
        
        for view_name in r['views'].keys():
            raw_rect_path = os.path.join(app.config['UPLOAD_FOLDER'], r['views'][view_name]['raw_filename'])
            defects, geo_feats, footprints = recalculate_view(
                raw_rect_path, process_meta['view_meta'][view_name],
                process_meta['telemetry'], process_meta['options'],
                view_name, pitch_offset, r['original_name'], app.config['UPLOAD_FOLDER'], filename
            )
            updated_r['views'][view_name]['defects'] = defects
            updated_r['views'][view_name]['footprint'] = footprints
            updated_r['geojson'].extend(geo_feats)
            
        new_results.append(updated_r)
    return jsonify({"success": True, "results": new_results})

@app.route('/export-zip', methods=['POST'])
def export_zip():
    project_data = request.json.get('results', [])
    if not project_data: return jsonify({"error": "No data provided"}), 400
    mem_file = create_raw_zip(project_data, app.config['UPLOAD_FOLDER'])
    return send_file(mem_file, download_name="DCPM_Export.zip", as_attachment=True)

@app.route('/export-flat-zip', methods=['POST'])
def export_flat_zip():
    project_data = request.json.get('results', [])
    if not project_data: return jsonify({"error": "No data provided"}), 400
    mem_file = create_flat_zip(project_data, app.config['UPLOAD_FOLDER'])
    return send_file(mem_file, download_name="DCPM_Flattened_Export.zip", as_attachment=True)

if __name__ == '__main__': 
    app.run(debug=True, port=5000)
    