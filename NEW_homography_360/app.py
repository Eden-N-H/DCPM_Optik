import os
import time
from flask import Flask, request, jsonify, render_template, url_for
from werkzeug.utils import secure_filename
from ultralytics import YOLO
import cv2

from core_math import (
    process_single_image,
    get_exif_gps,
    calculate_bearing,
    extract_gpmf_pitch,
    extract_video_gpmf_pitch_track,
    process_video_frames,
    haversine_distance
)

# Create the flask app and sets the upload directory to static/uploads
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join('static', 'uploads')
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Define two sets of allowed file extensions.
ALLOWED_IMAGE_EXT = {'.jpg', '.jpeg', '.png'}
ALLOWED_VIDEO_EXT = {'.mp4', '.mov', '.avi'}

# Root route renders the HTML page
@app.route('/')
def index():
    return render_template('index.html')

# Main route which accepts a POST HTML request with files + form data. Runs the full pipeline and returns a JSON.
@app.route('/process', methods=['POST'])
def process():
    # Prints the file fields + form fields
    print("FILES RECEIVED:", list(request.files.keys()))
    print("FORM DATA:", list(request.form.keys()))

    # 400 error if the files/images are not present or a model was not included
    if 'files' not in request.files and 'images' not in request.files or 'model' not in request.files:
        return jsonify({"error": "Missing input assets or model file"}), 400

    # assigns the list of images/files and the model files to a local variable
    uploaded_files = request.files.getlist('images') if 'images' in request.files else request.files.getlist('files')
    model_file = request.files['model']

    # Validation to ensure that assets and files were uploaded
    if len(uploaded_files) == 0 or model_file.filename == '':
        return jsonify({"error": "No processing targets selected"}), 400

    # Reads parameters and prints the snap mode.
    cam_height = float(request.form.get('cam_height', 1.6))
    gps_snap = request.form.get('gps_snap') == 'true'
    print(f"GPS snap mode: {gps_snap}")

    # Save and setup the ML Model
    model_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(model_file.filename))
    model_file.save(model_path)
    model = YOLO(model_path)

    # initialize accumulator lists.
    all_geojson_features = []
    processed_results = []
    trail_coordinates = []

    # 1. Step: Save files and build metadata staging queue
    # Loops through each asset in the processing queue.
    processing_queue = []
    for f in uploaded_files:
        # Skips empty file slots.
        if not f.filename:
            continue

        # Extracts the extension, builds a filename with prefix Unix timestamp and saves the file to disk.
        ext = os.path.splitext(f.filename)[1].lower()
        filename = f"{int(time.time())}_{secure_filename(f.filename)}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        f.save(filepath)

        # builds a metadata dict with fallback defaults.
        file_meta = {
            "filename": filename,
            "original_name": f.filename,
            "path": filepath,
            "ext": ext,
            "lat": 0.0,
            "lon": 0.0,
            "pitch": -15.0  # fallback defaults
        }

        # Extracts GPS and pitch data for images.
        if ext in ALLOWED_IMAGE_EXT:
            # Extracts the GPS coordinates from EXIF data.
            lat, lon = get_exif_gps(filepath)
            # Extracts pitch from embedded GPMF metadata (if possible)
            dynamic_pitch = extract_gpmf_pitch(filepath)
            file_meta.update({"lat": lat, "lon": lon, "pitch": dynamic_pitch})

        # Adds metadata to the queue.
        processing_queue.append(file_meta)

    # 2. Step: Chronological sorting (keeps assets tracking forward correctly)
    processing_queue = sorted(processing_queue, key=lambda x: x['filename'])

    # 3. Step: Heading Assignment & Spatial Location Clustering
    loc_id = 1
    last_valid_lat, last_valid_lon = None, None

    # Processes image entries (videos are excluded and handled elsewhere).
    for i in range(len(processing_queue)):
        if processing_queue[i]['ext'] in ALLOWED_IMAGE_EXT:
            # Computes the cameras direction towards the next asset's GPS position.
            if i < len(processing_queue) - 1 and processing_queue[i + 1]['lat'] != 0.0:
                heading = calculate_bearing(
                    processing_queue[i]['lat'], processing_queue[i]['lon'],
                    processing_queue[i + 1]['lat'], processing_queue[i + 1]['lon']
                )
            # If there is no next point, it inherits the previous item's heading.
            else:
                heading = processing_queue[i - 1].get('heading', 0.0) if i > 0 else 0.0
            processing_queue[i]['heading'] = heading

            # Calculate spatial grouping clusters (>50m gap forks into a new Location cluster ID)
            lat, lon = processing_queue[i]['lat'], processing_queue[i]['lon']
            if lat != 0.0 and lon != 0.0:
                if last_valid_lat is not None:
                    dist = haversine_distance(last_valid_lat, last_valid_lon, lat, lon)
                    if dist > 50.0:
                        loc_id += 1
                last_valid_lat, last_valid_lon = lat, lon

            processing_queue[i]['location'] = f"Location {loc_id}"

    # 4. Step: Deep Asset Engine Pipeline execution
    for asset in processing_queue:
        if asset['ext'] in ALLOWED_VIDEO_EXT:
            pitch_interpolator = extract_video_gpmf_pitch_track(asset['path'])

            video_defects, video_geojson, video_trail = process_video_frames(
                video_path=asset['path'],
                model=model,
                upload_dir=app.config['UPLOAD_FOLDER'],
                cam_height=cam_height,
                pitch_interp=pitch_interpolator,
                base_filename=asset['filename'],
                gps_snap=gps_snap
            )

            processed_results.extend(video_defects)
            all_geojson_features.extend(video_geojson)
            trail_coordinates.extend(video_trail)

        elif asset['ext'] in ALLOWED_IMAGE_EXT:
            defects, geo_feats = process_single_image(
                asset['path'], model, asset['filename'], app.config['UPLOAD_FOLDER'],
                asset['lat'], asset['lon'], asset['heading'], cam_height, asset['pitch']
            )

            all_geojson_features.extend(geo_feats)

            if asset['lat'] != 0.0:
                trail_coordinates.append([asset['lon'], asset['lat']])
                all_geojson_features.append({
                    "type": "Feature",
                    "properties": {
                        "type": "camera",
                        "filename": asset['original_name'],
                        "location": asset['location']
                    },
                    "geometry": {"type": "Point", "coordinates": [asset['lon'], asset['lat']]}
                })

            processed_results.append({
                "original_name": asset['original_name'],
                "lat": round(asset['lat'], 6),
                "lon": round(asset['lon'], 6),
                "pitch": round(asset['pitch'], 2),
                "location": asset['location'],
                "views": {
                    "front": {
                        "rect_url": url_for('static', filename=f"uploads/rect_front_{asset['filename']}"),
                        "bev_url": url_for('static', filename=f"uploads/bev_front_{asset['filename']}"),
                        "defects": defects['front']
                    },
                    "rear": {
                        "rect_url": url_for('static', filename=f"uploads/rect_rear_{asset['filename']}"),
                        "bev_url": url_for('static', filename=f"uploads/bev_rear_{asset['filename']}"),
                        "defects": defects['rear']
                    }
                }
            })

    # Prepend trail LineString layer if spatial tracking data points exist
    if len(trail_coordinates) > 1:
        all_geojson_features.insert(0, {
            "type": "Feature",
            "properties": {"type": "trail"},
            "geometry": {"type": "LineString", "coordinates": trail_coordinates}
        })

    return jsonify({
        "success": True,
        "results": processed_results,
        "geojson": {"type": "FeatureCollection", "features": all_geojson_features}
    })


if __name__ == '__main__':
    app.run(debug=True, port=5000)