import os
import time
from flask import Flask, request, jsonify, render_template, url_for
from werkzeug.utils import secure_filename
from ultralytics import YOLO
import cv2

# Import updated core math dependencies
from core_math import (
    process_single_image,
    get_exif_gps,
    calculate_bearing,
    extract_video_gpmf_pitch_track,
    process_video_frames
)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join('static', 'uploads')
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

ALLOWED_IMAGE_EXT = {'.jpg', '.jpeg', '.png'}
ALLOWED_VIDEO_EXT = {'.mp4', '.mov', '.avi'}


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/process', methods=['POST'])
def process():
    if 'files' not in request.files and 'images' not in request.files or 'model' not in request.files:
        return jsonify({"error": "Missing input assets or model file"}), 400

    # Combine or fallback files across common frontend field names
    uploaded_files = request.files.getlist('images') if 'images' in request.files else request.files.getlist('files')
    model_file = request.files['model']

    if len(uploaded_files) == 0 or model_file.filename == '':
        return jsonify({"error": "No processing targets selected"}), 400

    cam_height = float(request.form.get('cam_height', 1.6))

    # Save and boot the YOLO validation engine
    model_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(model_file.filename))
    model_file.save(model_path)
    model = YOLO(model_path)

    all_geojson_features = []
    processed_results = []
    trail_coordinates = []

    for f in uploaded_files:
        if not f.filename: continue

        ext = os.path.splitext(f.filename)[1].lower()
        filename = f"{int(time.time())}_{secure_filename(f.filename)}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        f.save(filepath)

        # ──────────────────────────────────────────────────────────────────────
        # PATH A: VIDEO PROCESSING PIPELINE
        # ──────────────────────────────────────────────────────────────────────
        if ext in ALLOWED_VIDEO_EXT:
            # Extract historical pitch telemetry curve mapped to real elapsed seconds
            pitch_interpolator = extract_video_gpmf_pitch_track(filepath)

            # Extract metadata and process frames sequentially
            video_defects, video_geojson, video_trail = process_video_frames(
                video_path=filepath,
                model=model,
                upload_dir=app.config['UPLOAD_FOLDER'],
                cam_height=cam_height,
                pitch_interp=pitch_interpolator,
                base_filename=filename
            )

            processed_results.extend(video_defects)
            all_geojson_features.extend(video_geojson)
            trail_coordinates.extend(video_trail)

        # ──────────────────────────────────────────────────────────────────────
        # PATH B: LEGACY STANDALONE IMAGE PIPELINE
        # ──────────────────────────────────────────────────────────────────────
        elif ext in ALLOWED_IMAGE_EXT:
            from core_math import extract_gpmf_pitch  # For images
            lat, lon = get_exif_gps(filepath)
            dynamic_pitch = extract_gpmf_pitch(filepath)

            # Placeholder dynamic bearing calculation (requires sequential images)
            heading = 0.0
            if len(processed_results) > 0 and lat != 0.0:
                heading = calculate_bearing(processed_results[-1]['lat'], processed_results[-1]['lon'], lat, lon)

            rect_name = f"rect_{filename}"
            bev_name = f"bev_{filename}"
            rect_path = os.path.join(app.config['UPLOAD_FOLDER'], rect_name)
            bev_path = os.path.join(app.config['UPLOAD_FOLDER'], bev_name)

            defects, geo_feats = process_single_image(
                filepath, model, rect_path, bev_path,
                lat, lon, heading, cam_height, dynamic_pitch
            )

            all_geojson_features.extend(geo_feats)
            if lat != 0.0:
                trail_coordinates.append([lon, lat])
                all_geojson_features.append({
                    "type": "Feature",
                    "properties": {"type": "camera", "filename": f.filename},
                    "geometry": {"type": "Point", "coordinates": [lon, lat]}
                })

            processed_results.append({
                "original_name": f.filename,
                "lat": round(lat, 6),
                "lon": round(lon, 6),
                "pitch": round(dynamic_pitch, 2),
                "rect_url": url_for('static', filename=f'uploads/{rect_name}'),
                "bev_url": url_for('static', filename=f'uploads/{bev_name}'),
                "defects": defects
            })

    # Append driving track path feature string
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