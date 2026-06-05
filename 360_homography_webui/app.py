import os
import time
from flask import Flask, request, jsonify, render_template, url_for
from werkzeug.utils import secure_filename
from ultralytics import YOLO
from core_math import process_single_image, get_exif_gps, calculate_bearing, extract_gpmf_pitch

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join('static', 'uploads')
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process():
    if 'images' not in request.files or 'model' not in request.files:
        return jsonify({"error": "Missing inputs"}), 400
        
    img_files = request.files.getlist('images')
    model_file = request.files['model']
    
    if len(img_files) == 0 or model_file.filename == '':
        return jsonify({"error": "No files selected"}), 400

    cam_height = float(request.form.get('cam_height', 1.6))
    # We no longer strictly rely on manual pitch, we will extract it per image.

    model_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(model_file.filename))
    model_file.save(model_path)
    model = YOLO(model_path)

    image_data = []
    trail_coordinates = []

    for f in img_files:
        filename = f"{int(time.time())}_{secure_filename(f.filename)}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        f.save(filepath)
        
        lat, lon = get_exif_gps(filepath)
        dynamic_pitch = extract_gpmf_pitch(filepath) # Auto-extract pitch!
        
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
        
        if image_data[i]['lat'] != 0.0:
            trail_coordinates.append([image_data[i]['lon'], image_data[i]['lat']])

    all_geojson_features = []
    processed_results = []

    # Add the Driving Trail (LineString) to the map
    if len(trail_coordinates) > 1:
        all_geojson_features.append({
            "type": "Feature",
            "properties": {"type": "trail"},
            "geometry": {"type": "LineString", "coordinates": trail_coordinates}
        })

    for img in image_data:
        rect_name = f"rect_{img['filename']}"
        bev_name = f"bev_{img['filename']}"
        rect_path = os.path.join(app.config['UPLOAD_FOLDER'], rect_name)
        bev_path = os.path.join(app.config['UPLOAD_FOLDER'], bev_name)

        defects, geo_feats = process_single_image(
            img['path'], model, rect_path, bev_path, 
            img['lat'], img['lon'], img['heading'], cam_height, img['pitch']
        )
        
        all_geojson_features.extend(geo_feats)

        if img['lat'] != 0.0:
            all_geojson_features.append({
                "type": "Feature",
                "properties": {"type": "camera", "filename": img['original_name']},
                "geometry": {"type": "Point", "coordinates": [img['lon'], img['lat']]}
            })

        processed_results.append({
            "original_name": img['original_name'],
            "lat": round(img['lat'], 6),
            "lon": round(img['lon'], 6),
            "pitch": round(img['pitch'], 2),
            "rect_url": url_for('static', filename=f'uploads/{rect_name}'),
            "bev_url": url_for('static', filename=f'uploads/{bev_name}'),
            "defects": defects
        })

    return jsonify({
        "success": True,
        "results": processed_results,
        "geojson": {"type": "FeatureCollection", "features": all_geojson_features}
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)