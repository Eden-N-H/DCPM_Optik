import os
import time
import json
import uuid
import threading
import queue
import zipfile
import cv2  
from io import BytesIO
from pathlib import Path
from flask import Flask, request, jsonify, render_template, Response, send_file
from werkzeug.utils import secure_filename
from ultralytics import YOLO

from core_math import (
    process_single_image, extract_full_photo_metadata,
    haversine_distance, calculate_bearing, process_video_frames_async,
    get_video_frame_metadata
)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join('static', 'uploads')
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

ALLOWED_IMAGE_EXT = {'.jpg', '.jpeg', '.png'}
ALLOWED_VIDEO_EXT = {'.mp4', '.mov', '.avi'}

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
PIPELINE_OUTPUT_ROOTS = [
    PROJECT_ROOT / "Data_pipelinine" / "output",
    PROJECT_ROOT / "Data_pipeline" / "output"
]

global_model = None
model_lock = threading.Lock()
active_tasks = {}

def safe_float(value, default=None):
    try:
        if value is None or str(value).strip() == "": return default
        return float(value)
    except (TypeError, ValueError): return default

def handle_model_upload(request_obj):
    global global_model
    if 'model' in request_obj.files and request_obj.files['model'].filename != '':
        model_file = request_obj.files['model']
        model_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(model_file.filename))
        model_file.save(model_path)
        global_model = YOLO(model_path)

def start_processing_job(image_data, cam_height, gps_snap, is_360, last_lat, last_lon, loc_id, interval_m, draw_grid):
    image_data = sorted(image_data, key=lambda x: x['filename'])
    trail_coordinates = []
    
    initial_ui_state = []
    has_video = any(a['ext'] in ALLOWED_VIDEO_EXT for a in image_data)

    for i in range(len(image_data)):
        if image_data[i]['ext'] in ALLOWED_IMAGE_EXT:
            lat, lon = image_data[i]['lat'], image_data[i]['lon']
            
            if i < len(image_data) - 1 and image_data[i+1]['lat'] is not None and lat is not None:
                heading = calculate_bearing(lat, lon, image_data[i+1]['lat'], image_data[i+1]['lon'])
            else:
                heading = image_data[i-1].get('heading', 0.0) if i > 0 else 0.0
            image_data[i]['heading'] = heading
            
            if lat is not None and lon is not None:
                trail_coordinates.append([lon, lat])
                if last_lat is not None and last_lon is not None:
                    dist = haversine_distance(last_lat, last_lon, lat, lon)
                    if dist > 50.0:
                        loc_id += 1
                last_lat, last_lon = lat, lon
                
            initial_ui_state.append(image_data[i])
            
        elif image_data[i]['ext'] in ALLOWED_VIDEO_EXT:
            video_frames = get_video_frame_metadata(image_data[i]['path'], interval_m, image_data[i]['original_name'], gps_snap)
            for vf in video_frames:
                initial_ui_state.append(vf)
                if vf.get('lat') is not None and vf.get('lon') is not None:
                    trail_coordinates.append([vf['lon'], vf['lat']])
                    last_lat, last_lon = vf['lat'], vf['lon']

        image_data[i]['location'] = f"Location {loc_id}"

    task_id = str(uuid.uuid4())
    active_tasks[task_id] = queue.Queue()
    total_est_frames = len(initial_ui_state)

    def process_worker(assets, t_id, height, snap, _is_360, _interval_m, _draw_grid):
        try:
            for asset in assets:
                def on_frame_processed(payload):
                    if "error" in payload:
                        active_tasks[t_id].put({"type": "item_error", "original_name": payload.get("original_name", asset['original_name']), "message": payload["error"], "is_video": payload.get("is_video", False)})
                    elif payload.get("type") == "health_report":
                        active_tasks[t_id].put({"type": "health_report", "original_name": payload.get("original_name"), "data": payload["data"]})
                    else:
                        active_tasks[t_id].put({"type": "update", "data": payload})
                    
                if asset['ext'] in ALLOWED_VIDEO_EXT:
                    process_video_frames_async(
                        asset['path'], global_model, app.config['UPLOAD_FOLDER'], height,
                        asset['filename'], asset['original_name'], snap, _interval_m, model_lock, _is_360, asset['location'], on_frame_processed, _draw_grid
                    )
                else:
                    try:
                        defects, geo_feats, base_filename, footprints = process_single_image(
                            asset['path'], global_model, asset['filename'], app.config['UPLOAD_FOLDER'], 
                            asset['lat'], asset['lon'], asset['heading'], height, asset['pitch'], asset['roll'], asset['klns'], asset['fov'], model_lock, _is_360, asset['original_name'], _draw_grid
                        )
                        
                        result_payload = {
                            "original_name": asset['original_name'],
                            "filename": asset['filename'],
                            "lat": round(asset['lat'], 6),
                            "lon": round(asset['lon'], 6),
                            "pitch": round(asset.get('pitch'), 2) if asset.get('pitch') is not None else None,
                            "roll": round(asset.get('roll'), 2) if asset.get('roll') is not None else None,
                            "location": asset['location'],
                            "geojson": geo_feats,
                            "views": {}
                        }
                        
                        for view in (['front', 'rear'] if _is_360 else ['front']):
                            result_payload["views"][view] = {
                                "raw_filename": f"raw_rect_{view}_{base_filename}",
                                "raw_bev_filename": f"raw_bev_{view}_{base_filename}",
                                "raw_bev_url": f"/static/uploads/raw_bev_{view}_{base_filename}",
                                "rect_url": f"/static/uploads/rect_{view}_{base_filename}",
                                "bev_url": f"/static/uploads/bev_{view}_{base_filename}",
                                "defects": defects[view],
                                "footprint": footprints[view]
                            }
                        
                        active_tasks[t_id].put({"type": "update", "data": result_payload})
                    except Exception as e:
                        active_tasks[t_id].put({"type": "item_error", "original_name": asset['original_name'], "message": str(e), "is_video": False})
            
            active_tasks[t_id].put({"type": "complete"})
        except Exception as e:
            active_tasks[t_id].put({"type": "error", "message": str(e)})

    threading.Thread(target=process_worker, args=(image_data, task_id, cam_height, gps_snap, is_360, interval_m, draw_grid)).start()

    initial_geojson = []
    if len(trail_coordinates) > 1:
        initial_geojson.append({
            "type": "Feature",
            "properties": {"type": "trail"},
            "geometry": {"type": "LineString", "coordinates": trail_coordinates}
        })

    return jsonify({
        "success": True,
        "task_id": task_id,
        "total_images": total_est_frames,
        "has_video": has_video,
        "initial_state": initial_ui_state,
        "initial_trail": {"type": "FeatureCollection", "features": initial_geojson},
        "last_lat": last_lat,
        "last_lon": last_lon,
        "last_loc_id": loc_id
    })

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process():
    handle_model_upload(request)
    if global_model is None: return jsonify({"error": "No ML model loaded into memory"}), 400

    img_files = request.files.getlist('images')
    if not img_files or img_files[0].filename == '': return jsonify({"error": "No media selected"}), 400

    cam_height = safe_float(request.form.get('cam_height'), 1.6)
    gps_snap = request.form.get('gps_snap') == 'true'
    is_360 = request.form.get('is_360') == 'true'
    draw_grid = request.form.get('draw_grid') == 'true'
    interval_m = safe_float(request.form.get('interval_m'), 2.0)
    
    last_lat = safe_float(request.form.get('last_lat'), None)
    last_lon = safe_float(request.form.get('last_lon'), None)
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
            
            # Cache the full metadata JSON
            meta_path = os.path.join(app.config['UPLOAD_FOLDER'], f"meta_{filename}.json")
            with open(meta_path, 'w') as mf:
                json.dump(full_meta, mf, indent=2)
            
        image_data.append(file_meta)
    
    return start_processing_job(image_data, cam_height, gps_snap, is_360, last_lat, last_lon, loc_id, interval_m, draw_grid)

@app.route('/process_pipeline_folder', methods=['POST'])
def process_pipeline_folder():
    handle_model_upload(request)
    if global_model is None: return jsonify({"error": "No ML model loaded into memory"}), 400

    cam_height = safe_float(request.form.get('cam_height'), 1.6)
    gps_snap = request.form.get('gps_snap') == 'true'
    is_360 = request.form.get('is_360') == 'true'
    draw_grid = request.form.get('draw_grid') == 'true'
    interval_m = safe_float(request.form.get('interval_m'), 2.0)

    last_lat = safe_float(request.form.get('last_lat'), None)
    last_lon = safe_float(request.form.get('last_lon'), None)
    loc_id = int(request.form.get('last_loc_id', 1))

    found_files = []
    for root_dir in PIPELINE_OUTPUT_ROOTS:
        frames_dir = root_dir / "frames"
        if frames_dir.exists():
            for ext in ALLOWED_IMAGE_EXT.union(ALLOWED_VIDEO_EXT):
                found_files.extend(list(frames_dir.rglob(f"*{ext}")))
                found_files.extend(list(frames_dir.rglob(f"*{ext.upper()}")))
            if found_files: break

    if not found_files: return jsonify({"error": "No media files found in the 'Data_pipeline/output/frames' directory."}), 404

    image_data = []
    for filepath_obj in found_files:
        filepath = str(filepath_obj)
        ext = filepath_obj.suffix.lower()
        filename = secure_filename(filepath_obj.name)
        
        file_meta = {"filename": filename, "original_name": filepath_obj.name, "path": filepath, "ext": ext, "lat": None, "lon": None, "pitch": None, "roll": None, "klns": None, "fov": None}

        if ext in ALLOWED_IMAGE_EXT:
            lat, lon, dynamic_pitch, dynamic_roll, klns, fov_meta, full_meta = extract_full_photo_metadata(filepath)
            file_meta.update({"lat": lat, "lon": lon, "pitch": dynamic_pitch, "roll": dynamic_roll, "klns": klns, "fov": fov_meta})
            
            meta_path = os.path.join(app.config['UPLOAD_FOLDER'], f"meta_{filename}.json")
            with open(meta_path, 'w') as mf:
                json.dump(full_meta, mf, indent=2)
            
        image_data.append(file_meta)
        
    return start_processing_job(image_data, cam_height, gps_snap, is_360, last_lat, last_lon, loc_id, interval_m, draw_grid)

@app.route('/stream/<task_id>')
def stream(task_id):
    def event_stream():
        q = active_tasks.get(task_id)
        if not q: yield f"data: {json.dumps({'type': 'error', 'message': 'Invalid Task ID'})}\n\n"; return
        while True:
            msg = q.get()
            yield f"data: {json.dumps(msg)}\n\n"
            if msg['type'] in ['complete', 'error']:
                del active_tasks[task_id]; break
    return Response(event_stream(), mimetype="text/event-stream")

@app.route('/export-zip', methods=['POST'])
def export_zip():
    project_data = request.json.get('results', [])
    if not project_data: return jsonify({"error": "No data provided"}), 400
    memory_file = BytesIO()
    
    with zipfile.ZipFile(memory_file, 'w', zipfile.ZIP_DEFLATED) as zf:
        for r in project_data:
            loc = r.get('location', 'Unknown Location')
            
            # Format output naming to ensure standard image extensions
            safe_orig = secure_filename(r['original_name'])
            if not safe_orig.lower().endswith(tuple(ALLOWED_IMAGE_EXT)):
                safe_orig += ".jpg"
            base_orig = os.path.splitext(safe_orig)[0]
            
            for view in r['views'].keys():
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], r['views'][view]['raw_filename'])
                if os.path.exists(file_path): 
                    zf.write(file_path, f"{loc}/{view}/RAW_{safe_orig}")
                    
                meta_path = os.path.join(app.config['UPLOAD_FOLDER'], f"meta_{r['filename']}.json")
                if os.path.exists(meta_path):
                    zf.write(meta_path, f"{loc}/{view}/RAW_{base_orig}.json")
                    
    memory_file.seek(0)
    return send_file(memory_file, download_name="DCPM_Export.zip", as_attachment=True)

@app.route('/export-flat-zip', methods=['POST'])
def export_flat_zip():
    project_data = request.json.get('results', [])
    if not project_data: return jsonify({"error": "No data provided"}), 400
    memory_file = BytesIO()
    
    with zipfile.ZipFile(memory_file, 'w', zipfile.ZIP_DEFLATED) as zf:
        for r in project_data:
            loc = r.get('location', 'Unknown Location')
            
            safe_orig = secure_filename(r['original_name'])
            if not safe_orig.lower().endswith(tuple(ALLOWED_IMAGE_EXT)):
                safe_orig += ".jpg"
            base_orig = os.path.splitext(safe_orig)[0]
            
            for view in r['views'].keys():
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], r['views'][view].get('raw_bev_filename', ''))
                if os.path.exists(file_path): 
                    zf.write(file_path, f"{loc}/{view}/FLAT_{safe_orig}")
                
                meta_path = os.path.join(app.config['UPLOAD_FOLDER'], f"meta_{r['filename']}.json")
                if os.path.exists(meta_path):
                    zf.write(meta_path, f"{loc}/{view}/FLAT_{base_orig}.json")
                    
    memory_file.seek(0)
    return send_file(memory_file, download_name="DCPM_Flattened_Export.zip", as_attachment=True)

if __name__ == '__main__': app.run(debug=True, port=5000)