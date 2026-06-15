import os
import time
import json
import uuid
import threading
import queue
import zipfile
from io import BytesIO
from flask import Flask, request, jsonify, render_template, Response, send_file
from werkzeug.utils import secure_filename
from ultralytics import YOLO
from core_math import process_single_image, get_exif_gps, calculate_bearing, extract_gpmf_pitch, haversine_distance

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join('static', 'uploads')
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Global variables to retain model state and manage background tasks
global_model = None
model_lock = threading.Lock()
active_tasks = {}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process():
    global global_model

    if 'images' not in request.files:
        return jsonify({"error": "Missing image files"}), 400

    # Load Model (only required if not previously loaded)
    if 'model' in request.files and request.files['model'].filename != '':
        model_file = request.files['model']
        model_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(model_file.filename))
        model_file.save(model_path)
        global_model = YOLO(model_path)

    if global_model is None:
        return jsonify({"error": "No ML model loaded into memory"}), 400

    img_files = request.files.getlist('images')
    if not img_files or img_files[0].filename == '':
        return jsonify({"error": "No files selected"}), 400

    # Form parameters for incremental adding
    cam_height = float(request.form.get('cam_height', 1.6))
    last_lat = float(request.form.get('last_lat', 0.0))
    last_lon = float(request.form.get('last_lon', 0.0))
    loc_id = int(request.form.get('last_loc_id', 1))

    image_data = []
    trail_coordinates = []

    # Fast initial extraction loop (No ML, just GPS and Telemetry)
    for f in img_files:
        filename = f"{int(time.time()*100)}_{secure_filename(f.filename)}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        f.save(filepath)
        
        lat, lon = get_exif_gps(filepath)
        dynamic_pitch = extract_gpmf_pitch(filepath)
        
        image_data.append({
            "filename": filename, 
            "original_name": f.filename,
            "path": filepath, 
            "lat": lat, 
            "lon": lon,
            "pitch": dynamic_pitch
        })
    
    image_data = sorted(image_data, key=lambda x: x['filename'])

    for i in range(len(image_data)):
        if i < len(image_data) - 1:
            heading = calculate_bearing(image_data[i]['lat'], image_data[i]['lon'], image_data[i+1]['lat'], image_data[i+1]['lon'])
        else:
            heading = image_data[i-1]['heading'] if i > 0 else 0.0
        image_data[i]['heading'] = heading
        
        lat, lon = image_data[i]['lat'], image_data[i]['lon']
        if lat != 0.0 and lon != 0.0:
            trail_coordinates.append([lon, lat])
            
            if last_lat != 0.0 and last_lon != 0.0:
                dist = haversine_distance(last_lat, last_lon, lat, lon)
                if dist > 50.0:
                    loc_id += 1
            last_lat, last_lon = lat, lon
            
        image_data[i]['location'] = f"Location {loc_id}"

    # Initialize Background Task Tracking
    task_id = str(uuid.uuid4())
    active_tasks[task_id] = queue.Queue()

    def process_worker(images, t_id, height):
        try:
            for img in images:
                defects, geo_feats, base_filename = process_single_image(
                    img['path'], global_model, img['filename'], app.config['UPLOAD_FOLDER'], 
                    img['lat'], img['lon'], img['heading'], height, img['pitch'], model_lock
                )
                
                # Removed Flask url_for() here to prevent Application Context errors in the background thread.
                # Constructing the static URL manually instead.
                result_payload = {
                    "original_name": img['original_name'],
                    "filename": img['filename'],
                    "lat": round(img['lat'], 6),
                    "lon": round(img['lon'], 6),
                    "pitch": round(img['pitch'], 2),
                    "location": img['location'],
                    "geojson": geo_feats,
                    "views": {
                        "front": {
                            "raw_filename": f"raw_rect_front_{base_filename}",
                            "raw_bev_filename": f"raw_bev_front_{base_filename}",
                            "rect_url": f"/static/uploads/rect_front_{base_filename}",
                            "bev_url": f"/static/uploads/bev_front_{base_filename}",
                            "defects": defects['front']
                        },
                        "rear": {
                            "raw_filename": f"raw_rect_rear_{base_filename}",
                            "raw_bev_filename": f"raw_bev_rear_{base_filename}",
                            "rect_url": f"/static/uploads/rect_rear_{base_filename}",
                            "bev_url": f"/static/uploads/bev_rear_{base_filename}",
                            "defects": defects['rear']
                        }
                    }
                }
                active_tasks[t_id].put({"type": "update", "data": result_payload})
            
            active_tasks[t_id].put({"type": "complete"})
        except Exception as e:
            active_tasks[t_id].put({"type": "error", "message": str(e)})

    # Start the background thread
    threading.Thread(target=process_worker, args=(image_data, task_id, cam_height)).start()

    # Send back the immediate trajectory payload so UI can render pending grey markers
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
        "total_images": len(image_data),
        "initial_state": image_data,
        "initial_trail": {"type": "FeatureCollection", "features": initial_geojson},
        "last_lat": last_lat,
        "last_lon": last_lon,
        "last_loc_id": loc_id
    })

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
            if msg['type'] in ['complete', 'error']:
                # Cleanup memory
                del active_tasks[task_id]
                break

    return Response(event_stream(), mimetype="text/event-stream")

@app.route('/export-zip', methods=['POST'])
def export_zip():
    project_data = request.json.get('results', [])
    if not project_data:
        return jsonify({"error": "No data provided"}), 400

    memory_file = BytesIO()
    with zipfile.ZipFile(memory_file, 'w', zipfile.ZIP_DEFLATED) as zf:
        for r in project_data:
            loc = r.get('location', 'Unknown Location')
            for view in ['front', 'rear']:
                raw_filename = r['views'][view]['raw_filename']
                original_name = r['original_name']
                # Create a clean target path inside the ZIP using the user's original filename
                target_filename = f"{loc}/{view}/RAW_{original_name}"
                
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], raw_filename)
                if os.path.exists(file_path):
                    zf.write(file_path, target_filename)

    memory_file.seek(0)
    return send_file(memory_file, download_name="DCPM_Export.zip", as_attachment=True)

@app.route('/export-flat-zip', methods=['POST'])
def export_flat_zip():
    project_data = request.json.get('results', [])
    if not project_data:
        return jsonify({"error": "No data provided"}), 400

    memory_file = BytesIO()
    with zipfile.ZipFile(memory_file, 'w', zipfile.ZIP_DEFLATED) as zf:
        for r in project_data:
            loc = r.get('location', 'Unknown Location')
            for view in ['front', 'rear']:
                raw_bev_filename = r['views'][view].get('raw_bev_filename') 
                if not raw_bev_filename:
                    continue
                
                original_name = r['original_name']
                # Create a clean target path inside the ZIP using the user's original filename
                target_filename = f"{loc}/{view}/FLAT_{original_name}"
                
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], raw_bev_filename)
                if os.path.exists(file_path):
                    zf.write(file_path, target_filename)

    memory_file.seek(0)
    return send_file(memory_file, download_name="DCPM_Flattened_Export.zip", as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True, port=5000)