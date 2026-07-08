import os
import sys
import time
import json
import base64
import cv2
import numpy as np
import threading
import traceback
from pathlib import Path
from flask import Flask, request, jsonify, render_template, Response, send_file
from werkzeug.utils import secure_filename
from ultralytics import YOLO

from constants import ALLOWED_IMAGE_EXT
from utils import safe_float
from parser_exif import extract_full_photo_metadata
from pipeline_image import generate_grid_preview, recalculate_view, get_projected_image, render_view_from_detections, _run_sam2_on_points
from cv_vp import find_vanishing_point_hough, calculate_pitch_yaw_deltas
from exports import create_raw_zip, create_flat_zip
from task_manager import start_processing_job, active_tasks, cancel_flags
from sam2_integration import load_sam2, get_predictor

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
os.chdir(PROJECT_ROOT)

app = Flask(__name__, static_folder='../static', template_folder='../templates')
app.config['UPLOAD_FOLDER'] = os.path.join(PROJECT_ROOT, 'static', 'uploads')
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

global_model = None
model_lock = threading.Lock()
sam2_predictor = None

FALLBACK_CLASSES = ["Defect", "Pothole", "Cracking", "Rutting", "Patching", "Edge Break", "Line Marking", "Other"]

# Load default YOLO model at startup
DEFAULT_MODEL_PATH = os.path.join(PROJECT_ROOT, 'models', 'RMCC_8_classes.pt')
if os.path.exists(DEFAULT_MODEL_PATH):
    global_model = YOLO(DEFAULT_MODEL_PATH)
    print(f"✓ Default YOLO model loaded: RMCC_8_classes.pt")
else:
    print("⚠ Default model not found at models/RMCC_8_classes.pt — upload one via the UI")

# Load SAM2 at startup
try:
    sam2_predictor = load_sam2()
    print("✓ SAM2 model loaded")
except Exception as e:
    print(f"⚠ SAM2 failed to load: {e}. Running without segmentation.")
    sam2_predictor = None

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

@app.route('/classes', methods=['GET'])
def get_classes():
    if global_model:
        return jsonify(list(global_model.names.values()))
    # Fallback classes if the user bypassed AI without a model but still wants to do manual QC
    return jsonify(FALLBACK_CLASSES)

@app.route('/process', methods=['POST'])
def process():
    handle_model_upload(request)
    
    skip_ai = request.form.get('skip_ai') == 'true'
    
    if not skip_ai and global_model is None: 
        return jsonify({"error": "No ML model loaded into memory. Upload a model or check 'Bypass AI'."}), 400
        
    img_files = request.files.getlist('images')
    if not img_files or img_files[0].filename == '': return jsonify({"error": "No media selected"}), 400

    options = {
        "media_type": request.form.get('media_type', 'standard-photos'),
        "has_telemetry": request.form.get('has_telemetry') == 'true',
        "cam_height": safe_float(request.form.get('cam_height'), 1.6),
        "interval_m": safe_float(request.form.get('interval_m'), 2.0),
        "is_360": request.form.get('is_360') == 'true',
        "draw_grid": request.form.get('draw_grid') == 'true',
        "comp_roll": request.form.get('comp_roll') == 'true',
        "comp_pitch": request.form.get('comp_pitch') == 'true',
        "undistort": request.form.get('undistort') == 'true',
        "ego_mask": request.form.get('ego_mask') == 'true',
        "skip_ai": skip_ai,
        "conf_thresh": safe_float(request.form.get('conf_thresh'), 0.25)
    }

    last_lat = safe_float(request.form.get('last_lat'), None)
    last_lon = safe_float(request.form.get('last_lon'), None)
    loc_id = int(request.form.get('last_loc_id', 1))

    image_data = []
    for f in img_files:
        ext = os.path.splitext(f.filename)[1].lower()
        filename = f"{int(time.time()*100)}_{secure_filename(f.filename)}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        f.save(filepath)
        
        file_meta = {
            "filename": filename, "original_name": f.filename, "path": filepath, "ext": ext, 
            "lat": None, "lon": None, "grav_vec": None, "klns": None, 
            "xfov": None, "yfov": None, "pitch": None, "roll": None
        }
        if ext in ALLOWED_IMAGE_EXT and options.get('has_telemetry', False):
            lat, lon, grav_vec, klns, xfov, yfov, pitch_ui, roll_ui, full_meta = extract_full_photo_metadata(filepath)
            file_meta.update({
                "lat": lat, "lon": lon, "grav_vec": grav_vec, 
                "klns": klns, "xfov": xfov, "yfov": yfov,
                "pitch": pitch_ui, "roll": roll_ui
            })
            with open(os.path.join(app.config['UPLOAD_FOLDER'], f"meta_{filename}.json"), 'w') as mf:
                json.dump(full_meta, mf, indent=2)
        image_data.append(file_meta)
    
    res = start_processing_job(image_data, options, last_lat, last_lon, loc_id, app.config['UPLOAD_FOLDER'], global_model, model_lock, sam2_predictor=sam2_predictor)
    return jsonify(res)

@app.route('/stream/<task_id>')
def stream(task_id):
    def event_stream():
        q = active_tasks.get(task_id)
        if not q:
            yield f"data: {json.dumps({'type': 'error', 'message': 'Invalid Task ID'})}\n\n"
            return
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
    calib = data['calibration']
    
    meta_path = os.path.join(app.config['UPLOAD_FOLDER'], f"process_meta_{filename}.json")
    if not os.path.exists(meta_path): return jsonify({"error": "Process metadata not found"}), 404
    
    with open(meta_path, 'r') as f: process_meta = json.load(f)
    
    source_path = os.path.join(app.config['UPLOAD_FOLDER'], f"source_{filename}")
    if not os.path.exists(source_path): return jsonify({"error": "Source original image missing. Cannot preview."}), 404
    
    preview_bgr = generate_grid_preview(source_path, process_meta, view_name, calib)
    _, buffer = cv2.imencode('.jpg', preview_bgr)
    preview_b64 = base64.b64encode(buffer).decode('utf-8')
    return jsonify({"success": True, "image": f"data:image/jpeg;base64,{preview_b64}"})

@app.route('/auto_vp', methods=['POST'])
def auto_vp():
    data = request.json
    filename = data['filename']
    view_name = data['view']
    calib = data['calibration']
    
    meta_path = os.path.join(app.config['UPLOAD_FOLDER'], f"process_meta_{filename}.json")
    with open(meta_path, 'r') as f: process_meta = json.load(f)
    source_path = os.path.join(app.config['UPLOAD_FOLDER'], f"source_{filename}")
    is_360 = process_meta['options'].get('is_360', True)
    
    rect_img, K, grav_vec, eff_pitch, eff_roll, eff_yaw = get_projected_image(source_path, process_meta['telemetry'], process_meta['options'], view_name, calib)
    
    vp = find_vanishing_point_hough(rect_img)
    if not vp:
        return jsonify({"success": False, "error": "AI could not detect strong lane lines or road geometry to determine the Vanishing Point."})
        
    u, v = vp
    h, w = rect_img.shape[:2]
    dp, dy = calculate_pitch_yaw_deltas(u, v, w, h, float(calib.get('fov') or 100.0), is_360)
    
    calib['pitch_offset'] = round(float(calib.get('pitch_offset') or 0.0) + dp, 1)
    calib['yaw_offset'] = round(float(calib.get('yaw_offset') or 0.0) + dy, 1)
    
    return jsonify({"success": True, "calibration": calib})

@app.route('/click_vp', methods=['POST'])
def click_vp():
    data = request.json
    filename = data['filename']
    view_name = data['view']
    calib = data['calibration']
    px = data['px']
    py = data['py']
    
    meta_path = os.path.join(app.config['UPLOAD_FOLDER'], f"process_meta_{filename}.json")
    with open(meta_path, 'r') as f: process_meta = json.load(f)
    source_path = os.path.join(app.config['UPLOAD_FOLDER'], f"source_{filename}")
    is_360 = process_meta['options'].get('is_360', True)
    
    rect_img, K, grav_vec, eff_pitch, eff_roll, eff_yaw = get_projected_image(source_path, process_meta['telemetry'], process_meta['options'], view_name, calib)
    h, w = rect_img.shape[:2]
    
    u, v = px * w, py * h
    dp, dy = calculate_pitch_yaw_deltas(u, v, w, h, float(calib.get('fov') or 100.0), is_360)
    
    calib['pitch_offset'] = round(float(calib.get('pitch_offset') or 0.0) + dp, 1)
    calib['yaw_offset'] = round(float(calib.get('yaw_offset') or 0.0) + dy, 1)
    
    return jsonify({"success": True, "calibration": calib})

@app.route('/preview_sam2', methods=['POST'])
def preview_sam2():
    try:
        data = request.json
        filename = data['filename']
        view_name = data['view']
        points = data.get('points')
        calib = data['calibration']
        
        meta_path = os.path.join(app.config['UPLOAD_FOLDER'], f"process_meta_{filename}.json")
        if not os.path.exists(meta_path): return jsonify({"error": "Process metadata not found"}), 404
        with open(meta_path, 'r') as f: process_meta = json.load(f)
        
        source_path = os.path.join(app.config['UPLOAD_FOLDER'], f"source_{filename}")
        media_type = process_meta.get('options', {}).get('media_type', '360-video')
        is_simple_frame = (media_type == 'orthographic')
        
        predictor = get_predictor()
        if not predictor:
            return jsonify({"error": "SAM2 not loaded"}), 400

        normalized_pts = []
        if is_simple_frame:
            rect_img = cv2.imread(source_path)
            h, w = rect_img.shape[:2]
            rect_points = [[int(np.clip(px * w, 0, w - 1)), int(np.clip(py * h, 0, h - 1))] for px, py in points]
            
            pts = _run_sam2_on_points(rect_img, rect_points, predictor)
            if pts is None or len(pts) < 3:
                return jsonify({"error": "SAM2 could not generate a valid mask."}), 400
                
            for pt in pts:
                normalized_pts.append([float(pt[0]/w), float(pt[1]/h)])
        else:
            rect_img, K, grav_vec, eff_pitch, eff_roll, eff_yaw = get_projected_image(source_path, process_meta['telemetry'], process_meta['options'], view_name, calib)
            
            cam_h = float(calib.get('cam_height') or 1.6)
            y_min = float(calib.get('z_near') or 1.5)
            y_max = float(calib.get('z_far') or 10.0)
            road_width = float(calib.get('lane_width') or 8.0)

            from cv_bev import get_bev_homography
            H_mat, bev_w, bev_h, PPM, _, _, _ = get_bev_homography(
                K, cam_h, grav_vec, eff_pitch, eff_roll, eff_yaw, y_min, y_max, road_width
            )
            
            bev_points = []
            for px, py in points:
                bev_x = np.clip(px * bev_w, 0, bev_w - 1)
                bev_y = np.clip(py * bev_h, 0, bev_h - 1)
                bev_points.append([bev_x, bev_y])
                
            raw_bev_bgr = cv2.warpPerspective(rect_img, H_mat, (bev_w, bev_h))
            bev_pts = _run_sam2_on_points(raw_bev_bgr, bev_points, predictor)
            
            if bev_pts is None or len(bev_pts) < 3:
                return jsonify({"error": "SAM2 could not generate a valid mask."}), 400
                
            for pt in bev_pts:
                normalized_pts.append([float(pt[0]/bev_w), float(pt[1]/bev_h)])
                
        return jsonify({"success": True, "points": normalized_pts})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/recalculate_bev', methods=['POST'])
def recalculate_bev():
    data = request.json
    calib = data['calibration']
    results = data['results']
    
    new_results = []
    for r in results:
        filename = r['filename']
        meta_path = os.path.join(app.config['UPLOAD_FOLDER'], f"process_meta_{filename}.json")
        if not os.path.exists(meta_path): continue
        with open(meta_path, 'r') as f: process_meta = json.load(f)
        
        source_path = os.path.join(app.config['UPLOAD_FOLDER'], f"source_{filename}")
        if not os.path.exists(source_path): continue

        updated_r = r.copy()
        updated_r['geojson'] = []
        
        for view_name in r['views'].keys():
            defects, geo_feats, footprints, view_meta_detections = recalculate_view(
                source_path, process_meta['telemetry'], process_meta['options'],
                view_name, calib, r['original_name'], app.config['UPLOAD_FOLDER'], filename,
                global_model, model_lock, sam2_predictor=sam2_predictor
            )
            updated_r['views'][view_name]['calibration'] = calib.copy()
            updated_r['views'][view_name]['defects'] = defects
            updated_r['views'][view_name]['footprint'] = footprints
            updated_r['geojson'].extend(geo_feats)
            
            process_meta['view_meta'][view_name]['detections'] = view_meta_detections
            
        with open(meta_path, 'w') as f: json.dump(process_meta, f)
            
        new_results.append(updated_r)
    return jsonify({"success": True, "results": new_results})

@app.route('/modify_defects', methods=['POST'])
def modify_defects():
    try:
        data = request.json
        filename = data['filename']
        view_name = data['view']
        action = data['action']
        idx = data.get('index')
        points = data.get('points')
        class_name = data.get('class_name')
        calib = data['calibration']
        use_sam2 = data.get('use_sam2', False)
        
        meta_path = os.path.join(app.config['UPLOAD_FOLDER'], f"process_meta_{filename}.json")
        if not os.path.exists(meta_path): return jsonify({"error": "Process metadata not found"}), 404
        with open(meta_path, 'r') as f: process_meta = json.load(f)
        
        if 'view_meta' not in process_meta:
            process_meta['view_meta'] = {}
        if view_name not in process_meta['view_meta']:
            process_meta['view_meta'][view_name] = {"detections": []}
            
        detections = process_meta['view_meta'][view_name].get('detections', [])
        
        if action in ['add', 're-outline'] and points:
            source_path = os.path.join(app.config['UPLOAD_FOLDER'], f"source_{filename}")
            media_type = process_meta.get('options', {}).get('media_type', '360-video')
            is_simple_frame = (media_type == 'orthographic')
            
            if is_simple_frame:
                rect_img = cv2.imread(source_path)
                h, w = rect_img.shape[:2]
                rect_points = [[int(np.clip(px * w, 0, w - 1)), int(np.clip(py * h, 0, h - 1))] for px, py in points]
                
                pts = None
                if use_sam2:
                    predictor = get_predictor()
                    if predictor:
                        pts = _run_sam2_on_points(rect_img, rect_points, predictor)
                        
                if pts is None or len(pts) < 3:
                    pts = np.array(rect_points, dtype=np.float32)
                    
                polygon_list = pts.tolist()
            else:
                rect_img, K, grav_vec, eff_pitch, eff_roll, eff_yaw = get_projected_image(source_path, process_meta['telemetry'], process_meta['options'], view_name, calib)
                h, w = rect_img.shape[:2]
                
                cam_h = float(calib.get('cam_height') or 1.6)
                y_min = float(calib.get('z_near') or 1.5)
                y_max = float(calib.get('z_far') or 10.0)
                road_width = float(calib.get('lane_width') or 8.0)

                from cv_bev import get_bev_homography
                H_mat, bev_w, bev_h, PPM, _, _, _ = get_bev_homography(
                    K, cam_h, grav_vec, eff_pitch, eff_roll, eff_yaw, y_min, y_max, road_width
                )
                
                try:
                    H_inv = np.linalg.inv(H_mat)
                except np.linalg.LinAlgError:
                    H_inv = np.eye(3)
                    
                bev_points = []
                for px, py in points:
                    bev_x = np.clip(px * bev_w, 0, bev_w - 1)
                    bev_y = np.clip(py * bev_h, 0, bev_h - 1)
                    bev_points.append([bev_x, bev_y])
                    
                bev_pts = None
                if use_sam2:
                    raw_bev_bgr = cv2.warpPerspective(rect_img, H_mat, (bev_w, bev_h))
                    predictor = get_predictor()
                    if predictor:
                        bev_pts = _run_sam2_on_points(raw_bev_bgr, bev_points, predictor)
                        
                if bev_pts is None or len(bev_pts) < 3:
                    bev_pts = np.array(bev_points, dtype=np.float32)
                    
                rect_contour_points = []
                for pt in bev_pts:
                    bev_x, bev_y = pt[0], pt[1]
                    vec = np.array([bev_x, bev_y, 1.0])
                    rect_pt = H_inv @ vec
                    
                    if rect_pt[2] != 0:
                        rect_x = rect_pt[0] / rect_pt[2]
                        rect_y = rect_pt[1] / rect_pt[2]
                    else:
                        rect_x, rect_y = rect_pt[0], rect_pt[1]
                        
                    rect_x = np.clip(rect_x, 0, w - 1)
                    rect_y = np.clip(rect_y, 0, h - 1)
                    rect_contour_points.append([float(rect_x), float(rect_y)])
                    
                polygon_list = rect_contour_points
            
            if action == 'add':
                cls_idx = 0
                if global_model is not None and class_name in global_model.names.values():
                    cls_idx = list(global_model.names.values()).index(class_name)
                elif class_name in FALLBACK_CLASSES:
                    cls_idx = FALLBACK_CLASSES.index(class_name)
                    
                from ultralytics.utils.plotting import colors
                color_bgr = colors(cls_idx, bgr=True)
                hex_color = f"#{int(color_bgr[2]):02x}{int(color_bgr[1]):02x}{int(color_bgr[0]):02x}"
                
                detections.append({
                    "class_name": class_name,
                    "conf": 1.0, 
                    "color_bgr": [int(c) for c in color_bgr],
                    "hex_color": hex_color,
                    "polygon": polygon_list
                })
            elif action == 're-outline':
                detections[idx]['polygon'] = polygon_list
                
        elif action == 'update':
            cls_idx = 0
            if global_model is not None and class_name in global_model.names.values():
                cls_idx = list(global_model.names.values()).index(class_name)
            elif class_name in FALLBACK_CLASSES:
                cls_idx = FALLBACK_CLASSES.index(class_name)
                
            from ultralytics.utils.plotting import colors
            color_bgr = colors(cls_idx, bgr=True)
            hex_color = f"#{int(color_bgr[2]):02x}{int(color_bgr[1]):02x}{int(color_bgr[0]):02x}"
            
            detections[idx]['class_name'] = class_name
            detections[idx]['color_bgr'] = [int(c) for c in color_bgr]
            detections[idx]['hex_color'] = hex_color
            
        elif action == 'delete':
            if 0 <= idx < len(detections):
                detections.pop(idx)
            
        process_meta['view_meta'][view_name]['detections'] = detections
        with open(meta_path, 'w') as f: json.dump(process_meta, f)
        
        defects, geojson_features = render_view_from_detections(
            process_meta, view_name, calib, filename, app.config['UPLOAD_FOLDER']
        )
        
        return jsonify({
            "success": True, 
            "defects": defects, 
            "geojson": geojson_features
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

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
    app.run(debug=False, port=5001)