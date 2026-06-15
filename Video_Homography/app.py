import os
import sys
import time
import csv
from pathlib import Path

from flask import Flask, request, jsonify, render_template, url_for
from werkzeug.utils import secure_filename
from ultralytics import YOLO
import cv2

# ----------------------------------------------------------------------
# Project paths
# ----------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent

# Allows Video_Homography/core_math.py to import GPMF_Extraction/extract_gpmf.py
GPMF_DIR = PROJECT_ROOT / "GPMF_Extraction"
if GPMF_DIR.exists():
    sys.path.insert(0, str(GPMF_DIR))

# Your folder is currently named Data_pipelinine.
# This also supports Data_pipeline in case you rename it later.
PIPELINE_OUTPUT_ROOTS = [
    PROJECT_ROOT / "Data_pipelinine" / "output",
    PROJECT_ROOT / "Data_pipeline" / "output",
]

from core_math import (
    process_single_image,
    get_exif_gps,
    calculate_bearing,
    extract_video_gpmf_pitch_track,
    process_video_frames,
    extract_gpmf_pitch,
)

# ----------------------------------------------------------------------
# Flask app setup
# ----------------------------------------------------------------------

app = Flask(__name__)

# Absolute upload path so the app works whether you run it from
# DCPM_Optik or from inside Video_Homography.
app.config["UPLOAD_FOLDER"] = str(BASE_DIR / "static" / "uploads")
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

ALLOWED_IMAGE_EXT = {".jpg", ".jpeg", ".png"}
ALLOWED_VIDEO_EXT = {".mp4", ".mov", ".avi"}


# ----------------------------------------------------------------------
# Helper functions for manifest processing
# ----------------------------------------------------------------------

def safe_float(value, default=0.0):
    """Safely convert CSV/form values into float."""
    try:
        if value is None or str(value).strip() == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def has_valid_gps(lat, lon):
    """Return True if latitude and longitude look usable."""
    return lat not in (0.0, None) and lon not in (0.0, None)


def normalise_manifest_path(path_value):
    """
    Normalise image paths from Windows/CSV format.

    Example CSV values:
    frames/hero5/hero5_frame_000001.jpg
    frames\\hero5\\hero5_frame_000001.jpg
    """
    return str(path_value).strip().replace("\\", os.sep).replace("/", os.sep)


def resolve_manifest_image_path(image_path_value):
    """
    Resolve image path from homography_input_manifest.csv.

    Your manifest usually stores image paths relative to:
    Data_pipelinine/output

    Example manifest value:
    frames/hero5/hero5_frame_000001.jpg

    Actual file location:
    Data_pipelinine/output/frames/hero5/hero5_frame_000001.jpg
    """
    if not image_path_value:
        return None

    cleaned_path = normalise_manifest_path(image_path_value)
    image_path = Path(cleaned_path)

    # 1. Absolute path in CSV
    if image_path.is_absolute() and image_path.exists():
        return image_path

    # 2. Relative to Data_pipelinine/output or Data_pipeline/output
    for output_root in PIPELINE_OUTPUT_ROOTS:
        candidate = output_root / image_path
        if candidate.exists():
            return candidate

    # 3. Relative to main project folder
    candidate = PROJECT_ROOT / image_path
    if candidate.exists():
        return candidate

    # 4. Relative to Video_Homography folder
    candidate = BASE_DIR / image_path
    if candidate.exists():
        return candidate

    # Return likely location for useful error messages
    return PIPELINE_OUTPUT_ROOTS[0] / image_path


def save_uploaded_file(file_storage, prefix=""):
    """Save uploaded file to static/uploads and return its saved path."""
    filename = secure_filename(file_storage.filename)

    if prefix:
        filename = f"{prefix}_{filename}"

    save_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file_storage.save(save_path)

    return save_path


# ----------------------------------------------------------------------
# Routes
# ----------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/process", methods=["POST"])
def process():
    """
    Existing website route.

    This processes manually uploaded images/videos plus a YOLO model.
    This route is kept so the original localhost website still works.
    """
    print("FILES RECEIVED:", list(request.files.keys()))
    print("FORM DATA:", list(request.form.keys()))

    missing_targets = "files" not in request.files and "images" not in request.files
    missing_model = "model" not in request.files

    if missing_targets or missing_model:
        return jsonify({"error": "Missing input assets or model file"}), 400

    uploaded_files = (
        request.files.getlist("images")
        if "images" in request.files
        else request.files.getlist("files")
    )

    model_file = request.files["model"]

    if len(uploaded_files) == 0 or model_file.filename == "":
        return jsonify({"error": "No processing targets selected"}), 400

    cam_height = safe_float(request.form.get("cam_height"), 1.6)

    gps_snap = request.form.get("gps_snap") == "true"
    print(f"GPS snap mode: {gps_snap}")

    model_path = save_uploaded_file(model_file, prefix=str(int(time.time())))
    model = YOLO(model_path)

    all_geojson_features = []
    processed_results = []
    trail_coordinates = []

    for f in uploaded_files:
        if not f.filename:
            continue

        ext = os.path.splitext(f.filename)[1].lower()
        filename = f"{int(time.time())}_{secure_filename(f.filename)}"
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        f.save(filepath)

        if ext in ALLOWED_VIDEO_EXT:
            pitch_interpolator = extract_video_gpmf_pitch_track(filepath)

            video_defects, video_geojson, video_trail = process_video_frames(
                video_path=filepath,
                model=model,
                upload_dir=app.config["UPLOAD_FOLDER"],
                cam_height=cam_height,
                pitch_interp=pitch_interpolator,
                base_filename=filename,
                gps_snap=gps_snap,
            )

            processed_results.extend(video_defects)
            all_geojson_features.extend(video_geojson)
            trail_coordinates.extend(video_trail)

        elif ext in ALLOWED_IMAGE_EXT:
            lat, lon = get_exif_gps(filepath)
            dynamic_pitch = extract_gpmf_pitch(filepath)

            heading = 0.0

            if len(processed_results) > 0 and lat != 0.0:
                heading = calculate_bearing(
                    processed_results[-1]["lat"],
                    processed_results[-1]["lon"],
                    lat,
                    lon,
                )

            rect_name = f"rect_{filename}"
            bev_name = f"bev_{filename}"

            rect_path = os.path.join(app.config["UPLOAD_FOLDER"], rect_name)
            bev_path = os.path.join(app.config["UPLOAD_FOLDER"], bev_name)

            defects, geo_feats = process_single_image(
                filepath,
                model,
                rect_path,
                bev_path,
                lat,
                lon,
                heading,
                cam_height,
                dynamic_pitch,
            )

            all_geojson_features.extend(geo_feats)

            if lat != 0.0:
                trail_coordinates.append([lon, lat])

                all_geojson_features.append({
                    "type": "Feature",
                    "properties": {
                        "type": "camera",
                        "filename": f.filename,
                    },
                    "geometry": {
                        "type": "Point",
                        "coordinates": [lon, lat],
                    },
                })

            processed_results.append({
                "original_name": f.filename,
                "lat": round(lat, 6),
                "lon": round(lon, 6),
                "pitch": round(dynamic_pitch, 2),
                "rect_url": url_for("static", filename=f"uploads/{rect_name}"),
                "bev_url": url_for("static", filename=f"uploads/{bev_name}"),
                "defects": defects,
            })

        else:
            print(f"[Skipped] Unsupported file type: {f.filename}")

    if len(trail_coordinates) > 1:
        all_geojson_features.insert(0, {
            "type": "Feature",
            "properties": {
                "type": "trail",
            },
            "geometry": {
                "type": "LineString",
                "coordinates": trail_coordinates,
            },
        })

    return jsonify({
        "success": True,
        "results": processed_results,
        "geojson": {
            "type": "FeatureCollection",
            "features": all_geojson_features,
        },
    })


@app.route("/process_manifest", methods=["POST"])
def process_manifest():
    """
    New manifest route.

    This allows the localhost website to process frames created by your
    data pipeline using homography_input_manifest.csv.

    Expected form data:
    - model: YOLO .pt model file
    - manifest: homography_input_manifest.csv
    - cam_height: fallback camera height, e.g. 1.6
    """
    print("MANIFEST PROCESSING STARTED")
    print("FILES RECEIVED:", list(request.files.keys()))
    print("FORM DATA:", list(request.form.keys()))

    if "manifest" not in request.files or "model" not in request.files:
        return jsonify({"error": "Missing manifest CSV or YOLO model file"}), 400

    manifest_file = request.files["manifest"]
    model_file = request.files["model"]

    if manifest_file.filename == "" or model_file.filename == "":
        return jsonify({"error": "Manifest file or model file was not selected"}), 400

    cam_height_default = safe_float(request.form.get("cam_height"), 1.6)

    timestamp_prefix = str(int(time.time()))

    # Save and load YOLO model
    model_path = save_uploaded_file(model_file, prefix=timestamp_prefix)
    model = YOLO(model_path)

    # Save uploaded manifest CSV
    manifest_path = save_uploaded_file(manifest_file, prefix=timestamp_prefix)

    # Read manifest rows
    try:
        with open(manifest_path, "r", newline="", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            rows = list(reader)
    except Exception as exc:
        return jsonify({"error": f"Could not read manifest CSV: {exc}"}), 400

    if not rows:
        return jsonify({"error": "Manifest CSV is empty"}), 400

    all_geojson_features = []
    processed_results = []
    trail_coordinates = []
    skipped_frames = []

    previous_lat = None
    previous_lon = None
    previous_heading = 0.0

    for index, row in enumerate(rows):
        frame_id = row.get("frame_id", f"frame_{index + 1}").strip()
        image_path_value = row.get("image_path", "").strip()
        homography_status = row.get("homography_status", "").strip()

        # If you accidentally upload validation_manifest.csv,
        # this skips invalid rows.
        if homography_status.startswith("invalid"):
            skipped_frames.append({
                "frame_id": frame_id,
                "reason": homography_status,
            })

            print(f"[Skipped] {frame_id}: {homography_status}")
            continue

        frame_path = resolve_manifest_image_path(image_path_value)

        if frame_path is None or not frame_path.exists():
            skipped_frames.append({
                "frame_id": frame_id,
                "reason": f"image_not_found: {frame_path}",
            })

            print(f"[Skipped] Image not found for {frame_id}: {frame_path}")
            continue

        # Read manifest metadata
        lat = safe_float(row.get("latitude"), 0.0)
        lon = safe_float(row.get("longitude"), 0.0)
        cam_height = safe_float(row.get("camera_height_m"), cam_height_default)

        # Your current manifest may have blank pitch/roll/yaw.
        # This fallback lets the prototype run.
        # Later, replace this with real GoPro IMU pitch data.
        pitch = safe_float(row.get("pitch"), -15.0)

        valid_gps = has_valid_gps(lat, lon)

        # Calculate heading from previous GPS point
        if previous_lat is not None and previous_lon is not None and valid_gps:
            heading = calculate_bearing(previous_lat, previous_lon, lat, lon)
            previous_heading = heading
        else:
            heading = previous_heading

        if valid_gps:
            previous_lat = lat
            previous_lon = lon
            trail_coordinates.append([lon, lat])

        safe_frame_id = secure_filename(frame_id)

        rect_name = f"rect_{safe_frame_id}.jpg"
        bev_name = f"bev_{safe_frame_id}.jpg"

        rect_path = os.path.join(app.config["UPLOAD_FOLDER"], rect_name)
        bev_path = os.path.join(app.config["UPLOAD_FOLDER"], bev_name)

        try:
            defects, geo_feats = process_single_image(
                str(frame_path),
                model,
                rect_path,
                bev_path,
                lat,
                lon,
                heading,
                cam_height,
                pitch,
            )

            # Avoid displaying fake polygons at [0, 0] if GPS is missing.
            if valid_gps:
                all_geojson_features.extend(geo_feats)

                all_geojson_features.append({
                    "type": "Feature",
                    "properties": {
                        "type": "camera",
                        "filename": frame_id,
                    },
                    "geometry": {
                        "type": "Point",
                        "coordinates": [lon, lat],
                    },
                })

            processed_results.append({
                "original_name": frame_id,
                "lat": round(lat, 6),
                "lon": round(lon, 6),
                "pitch": round(pitch, 2),
                "rect_url": url_for("static", filename=f"uploads/{rect_name}"),
                "bev_url": url_for("static", filename=f"uploads/{bev_name}"),
                "defects": defects,
            })

            print(f"[Processed] {frame_id}")

        except Exception as exc:
            skipped_frames.append({
                "frame_id": frame_id,
                "reason": str(exc),
            })

            print(f"[Error] Failed processing {frame_id}: {exc}")

    if len(trail_coordinates) > 1:
        all_geojson_features.insert(0, {
            "type": "Feature",
            "properties": {
                "type": "trail",
            },
            "geometry": {
                "type": "LineString",
                "coordinates": trail_coordinates,
            },
        })

    return jsonify({
        "success": True,
        "processed_count": len(processed_results),
        "skipped_count": len(skipped_frames),
        "skipped_frames": skipped_frames,
        "results": processed_results,
        "geojson": {
            "type": "FeatureCollection",
            "features": all_geojson_features,
        },
    })


if __name__ == "__main__":
    app.run(debug=True, port=5000)