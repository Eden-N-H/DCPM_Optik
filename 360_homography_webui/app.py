import os
import time
import csv
from pathlib import Path
from threading import Lock

from flask import Flask, request, jsonify, render_template, url_for
from werkzeug.utils import secure_filename
from ultralytics import YOLO

from core_math import (
    process_single_image,
    get_exif_gps,
    calculate_bearing,
    extract_gpmf_pitch,
    haversine_distance,
)

# ---------------------------------------------------------------------
# Flask app setup
# ---------------------------------------------------------------------

app = Flask(__name__)

# Required by 360_homography_webui/core_math.py
model_lock = Lock()

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent

app.config["UPLOAD_FOLDER"] = str(BASE_DIR / "static" / "uploads")
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

ALLOWED_IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

PIPELINE_OUTPUT_ROOTS = [
    PROJECT_ROOT / "Data_pipelinine" / "output",
    PROJECT_ROOT / "Data_pipeline" / "output",
]


# ---------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------

def safe_float(value, default=0.0):
    """Safely convert values from CSV/form input into float."""
    try:
        if value is None or str(value).strip() == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def has_valid_gps(lat, lon):
    """Check whether GPS values are usable."""
    return lat not in (None, 0.0) and lon not in (None, 0.0)


def normalise_manifest_path(path_value):
    """
    Normalise Windows/Linux paths from CSV.

    Example:
    frames\\hero5\\frame.jpg
    frames/hero5/frame.jpg
    """
    return str(path_value).strip().replace("\\", os.sep).replace("/", os.sep)


def resolve_manifest_image_path(image_path_value):
    """
    Resolve image_path from homography_input_manifest.csv.

    Manifest usually stores:
    frames/hero5/hero5_frame_000001.jpg

    Real file is usually:
    Data_pipelinine/output/frames/hero5/hero5_frame_000001.jpg
    """
    if not image_path_value:
        return None

    cleaned_path = normalise_manifest_path(image_path_value)
    image_path = Path(cleaned_path)

    if image_path.is_absolute() and image_path.exists():
        return image_path

    for output_root in PIPELINE_OUTPUT_ROOTS:
        candidate = output_root / image_path
        if candidate.exists():
            return candidate

    candidate = PROJECT_ROOT / image_path
    if candidate.exists():
        return candidate

    candidate = BASE_DIR / image_path
    if candidate.exists():
        return candidate

    return PIPELINE_OUTPUT_ROOTS[0] / image_path


def save_uploaded_file(file_storage, prefix=""):
    """Save uploaded file to static/uploads and return saved path."""
    filename = secure_filename(file_storage.filename)

    if prefix:
        filename = f"{prefix}_{filename}"

    save_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file_storage.save(save_path)

    return save_path


def run_process_single_image(
    image_path,
    model,
    filename,
    upload_folder,
    lat,
    lon,
    heading,
    cam_height,
    pitch,
):
    """
    Runs 360 core_math.process_single_image safely.

    Some versions return:
    - defects, geo_feats

    Other versions return:
    - defects, geo_feats, extra_value

    This wrapper accepts both and uses the first two values.
    """
    result = process_single_image(
        image_path,
        model,
        filename,
        upload_folder,
        lat,
        lon,
        heading,
        cam_height,
        pitch,
        model_lock,
    )

    if isinstance(result, tuple) and len(result) >= 2:
        defects = result[0]
        geo_feats = result[1]
        return defects, geo_feats

    raise RuntimeError(
        "process_single_image did not return defects and geojson features correctly"
    )


def assign_headings_and_locations(image_data):
    """
    Assign heading and location grouping to image records.

    A new location group starts when distance between GPS points is over 50 m.
    """
    trail_coordinates = []

    loc_id = 1
    last_valid_lat = None
    last_valid_lon = None
    previous_heading = 0.0

    for i in range(len(image_data)):
        current = image_data[i]

        lat = current["lat"]
        lon = current["lon"]

        heading = previous_heading

        if has_valid_gps(lat, lon):
            for j in range(i + 1, len(image_data)):
                next_lat = image_data[j]["lat"]
                next_lon = image_data[j]["lon"]

                if has_valid_gps(next_lat, next_lon):
                    heading = calculate_bearing(lat, lon, next_lat, next_lon)
                    previous_heading = heading
                    break

        current["heading"] = heading

        if has_valid_gps(lat, lon):
            trail_coordinates.append([lon, lat])

            if last_valid_lat is not None and last_valid_lon is not None:
                distance = haversine_distance(last_valid_lat, last_valid_lon, lat, lon)

                if distance > 50.0:
                    loc_id += 1

            last_valid_lat = lat
            last_valid_lon = lon

        current["location"] = f"Location {loc_id}"

    return image_data, trail_coordinates


def build_result_object(img, defects):
    """Build frontend-compatible result object."""
    return {
        "original_name": img["original_name"],
        "lat": round(img["lat"], 6),
        "lon": round(img["lon"], 6),
        "pitch": round(img["pitch"], 2),
        "location": img["location"],
        "views": {
            "front": {
                "rect_url": url_for("static", filename=f"uploads/rect_front_{img['filename']}"),
                "bev_url": url_for("static", filename=f"uploads/bev_front_{img['filename']}"),
                "defects": defects.get("front", []),
            },
            "rear": {
                "rect_url": url_for("static", filename=f"uploads/rect_rear_{img['filename']}"),
                "bev_url": url_for("static", filename=f"uploads/bev_rear_{img['filename']}"),
                "defects": defects.get("rear", []),
            },
        },
    }


# ---------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/process", methods=["POST"])
def process():
    """
    Manual upload route.

    Input:
    - model: YOLO .pt file
    - images: uploaded 360 images
    - cam_height
    """
    print("MANUAL IMAGE PROCESSING STARTED")
    print("FILES RECEIVED:", list(request.files.keys()))
    print("FORM DATA:", list(request.form.keys()))

    if "images" not in request.files or "model" not in request.files:
        return jsonify({"error": "Missing image files or YOLO model"}), 400

    img_files = request.files.getlist("images")
    model_file = request.files["model"]

    if len(img_files) == 0 or model_file.filename == "":
        return jsonify({"error": "No files selected"}), 400

    cam_height = safe_float(request.form.get("cam_height"), 1.6)

    timestamp_prefix = str(int(time.time()))

    model_path = save_uploaded_file(model_file, prefix=timestamp_prefix)
    model = YOLO(model_path)

    image_data = []

    for f in img_files:
        if not f.filename:
            continue

        ext = os.path.splitext(f.filename)[1].lower()

        if ext not in ALLOWED_IMAGE_EXT:
            print(f"[Skipped] Unsupported image file: {f.filename}")
            continue

        filename = f"{int(time.time())}_{secure_filename(f.filename)}"
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        f.save(filepath)

        lat, lon = get_exif_gps(filepath)
        dynamic_pitch = extract_gpmf_pitch(filepath)

        image_data.append({
            "filename": filename,
            "original_name": f.filename,
            "path": filepath,
            "lat": lat,
            "lon": lon,
            "pitch": dynamic_pitch,
            "cam_height": cam_height,
        })

    if not image_data:
        return jsonify({"error": "No valid images were provided"}), 400

    image_data = sorted(image_data, key=lambda x: x["filename"])
    image_data, trail_coordinates = assign_headings_and_locations(image_data)

    all_geojson_features = []
    processed_results = []
    skipped_frames = []

    if len(trail_coordinates) > 1:
        all_geojson_features.append({
            "type": "Feature",
            "properties": {
                "type": "trail",
            },
            "geometry": {
                "type": "LineString",
                "coordinates": trail_coordinates,
            },
        })

    for img in image_data:
        try:
            defects, geo_feats = run_process_single_image(
                img["path"],
                model,
                img["filename"],
                app.config["UPLOAD_FOLDER"],
                img["lat"],
                img["lon"],
                img["heading"],
                img["cam_height"],
                img["pitch"],
            )

            if has_valid_gps(img["lat"], img["lon"]):
                all_geojson_features.extend(geo_feats)

                all_geojson_features.append({
                    "type": "Feature",
                    "properties": {
                        "type": "camera",
                        "filename": img["original_name"],
                        "location": img["location"],
                    },
                    "geometry": {
                        "type": "Point",
                        "coordinates": [img["lon"], img["lat"]],
                    },
                })

            processed_results.append(build_result_object(img, defects))

            print(f"[Processed] {img['original_name']}")

        except Exception as exc:
            skipped_frames.append({
                "frame_id": img["original_name"],
                "reason": str(exc),
            })

            print(f"[Error] Failed processing {img['original_name']}: {exc}")

    if len(processed_results) == 0:
        return jsonify({
            "success": False,
            "error": "All manually uploaded images failed during 360 homography processing.",
            "processed_count": 0,
            "skipped_count": len(skipped_frames),
            "skipped_frames": skipped_frames,
        }), 500

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


@app.route("/process_manifest", methods=["POST"])
def process_manifest():
    """
    Manifest-based route.

    Input:
    - model: YOLO .pt file
    - manifest: homography_input_manifest.csv
    - cam_height: fallback camera height
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

    model_path = save_uploaded_file(model_file, prefix=timestamp_prefix)
    model = YOLO(model_path)

    manifest_path = save_uploaded_file(manifest_file, prefix=timestamp_prefix)

    try:
        with open(manifest_path, "r", newline="", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            rows = list(reader)
    except Exception as exc:
        return jsonify({"error": f"Could not read manifest CSV: {exc}"}), 400

    if not rows:
        return jsonify({"error": "Manifest CSV is empty"}), 400

    image_data = []
    skipped_frames = []

    for index, row in enumerate(rows):
        frame_id = row.get("frame_id", f"frame_{index + 1}").strip()
        image_path_value = row.get("image_path", "").strip()
        homography_status = row.get("homography_status", "").strip()

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

        lat = safe_float(row.get("latitude"), 0.0)
        lon = safe_float(row.get("longitude"), 0.0)

        cam_height = safe_float(
            row.get("camera_height_m"),
            cam_height_default,
        )

        # Temporary fallback because current manifest has blank pitch.
        pitch = safe_float(row.get("pitch"), -15.0)

        safe_frame_id = secure_filename(frame_id)

        # This filename is used to generate rect_front_*, bev_front_*, etc.
        base_filename = f"{safe_frame_id}.jpg"

        image_data.append({
            "filename": base_filename,
            "original_name": frame_id,
            "path": str(frame_path),
            "lat": lat,
            "lon": lon,
            "pitch": pitch,
            "cam_height": cam_height,
        })

    if not image_data:
        return jsonify({
            "success": False,
            "error": "No valid manifest rows were available for processing.",
            "processed_count": 0,
            "skipped_count": len(skipped_frames),
            "skipped_frames": skipped_frames,
        }), 400

    image_data, trail_coordinates = assign_headings_and_locations(image_data)

    all_geojson_features = []
    processed_results = []

    if len(trail_coordinates) > 1:
        all_geojson_features.append({
            "type": "Feature",
            "properties": {
                "type": "trail",
            },
            "geometry": {
                "type": "LineString",
                "coordinates": trail_coordinates,
            },
        })

    for img in image_data:
        try:
            defects, geo_feats = run_process_single_image(
                img["path"],
                model,
                img["filename"],
                app.config["UPLOAD_FOLDER"],
                img["lat"],
                img["lon"],
                img["heading"],
                img["cam_height"],
                img["pitch"],
            )

            if has_valid_gps(img["lat"], img["lon"]):
                all_geojson_features.extend(geo_feats)

                all_geojson_features.append({
                    "type": "Feature",
                    "properties": {
                        "type": "camera",
                        "filename": img["original_name"],
                        "location": img["location"],
                    },
                    "geometry": {
                        "type": "Point",
                        "coordinates": [img["lon"], img["lat"]],
                    },
                })

            processed_results.append(build_result_object(img, defects))

            print(f"[Processed] {img['original_name']}")

        except Exception as exc:
            skipped_frames.append({
                "frame_id": img["original_name"],
                "reason": str(exc),
            })

            print(f"[Error] Failed processing {img['original_name']}: {exc}")

    if len(processed_results) == 0:
        return jsonify({
            "success": False,
            "error": "All manifest frames failed during 360 homography processing.",
            "processed_count": 0,
            "skipped_count": len(skipped_frames),
            "skipped_frames": skipped_frames,
        }), 500

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