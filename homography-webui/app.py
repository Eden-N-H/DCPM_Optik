import os
import time
import json
import uuid
import threading
import queue
import zipfile
from io import BytesIO
from pathlib import Path

from flask import Flask, request, jsonify, render_template, Response, send_file
from werkzeug.utils import secure_filename
from ultralytics import YOLO

from core_math import (
    process_single_image,
    get_exif_gps,
    calculate_bearing,
    extract_gpmf_pitch,
    haversine_distance,
    extract_video_gpmf_pitch_track,
    process_video_frames_async,
    get_video_frame_metadata
)

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent

UPLOAD_FOLDER = BASE_DIR / "static" / "uploads"
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)

app.config["UPLOAD_FOLDER"] = str(UPLOAD_FOLDER)

ALLOWED_IMAGE_EXT = {".jpg", ".jpeg", ".png"}
ALLOWED_VIDEO_EXT = {".mp4", ".mov", ".avi"}

PIPELINE_OUTPUT_ROOTS = [
    PROJECT_ROOT / "Data_pipelinine" / "output",
    PROJECT_ROOT / "Data_pipeline" / "output",
    BASE_DIR / "Data_pipelinine" / "output",
    BASE_DIR / "Data_pipeline" / "output",
]

global_model = None
model_lock = threading.Lock()
active_tasks = {}


def safe_float(value, default=0.0):
    try:
        if value is None or str(value).strip() == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def safe_int(value, default=0):
    try:
        if value is None or str(value).strip() == "":
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def form_bool(name, default=False):
    value = request.form.get(name)

    if value is None:
        return default

    return str(value).lower() in {"true", "1", "yes", "on"}


def handle_model_upload(request_obj):
    """Save uploaded YOLO .pt model and load it globally."""
    global global_model

    model_file = request_obj.files.get("model")

    if model_file and model_file.filename:
        model_filename = secure_filename(model_file.filename)
        model_path = UPLOAD_FOLDER / model_filename
        model_file.save(model_path)

        with model_lock:
            global_model = YOLO(str(model_path))

        print(f"[Model] Loaded model: {model_path}")


def collect_uploaded_media(request_obj):
    """
    Accept multiple possible frontend field names:
    - images
    - media
    - files
    """
    files = []

    for key in ["images", "media", "files"]:
        files.extend(request_obj.files.getlist(key))

    valid_files = [f for f in files if f and f.filename]

    return valid_files


def build_file_metadata(filepath, filename, original_name, ext):
    file_meta = {
        "filename": filename,
        "original_name": original_name,
        "path": str(filepath),
        "ext": ext,
        "lat": 0.0,
        "lon": 0.0,
        "pitch": -15.0,
        "heading": 0.0,
        "location": "Location 1"
    }

    if ext in ALLOWED_IMAGE_EXT:
        lat, lon = get_exif_gps(str(filepath))
        dynamic_pitch = extract_gpmf_pitch(str(filepath))

        file_meta.update({
            "lat": lat,
            "lon": lon,
            "pitch": dynamic_pitch
        })

    return file_meta


def merge_geojson_features(full_geojson, new_geojson):
    if not new_geojson:
        return full_geojson

    if isinstance(new_geojson, list):
        full_geojson["features"].extend(new_geojson)
        return full_geojson

    if isinstance(new_geojson, dict):
        if new_geojson.get("type") == "FeatureCollection":
            full_geojson["features"].extend(new_geojson.get("features", []))
        elif new_geojson.get("type") == "Feature":
            full_geojson["features"].append(new_geojson)

    return full_geojson


def start_processing_job(
    image_data,
    cam_height,
    gps_snap,
    is_360,
    last_lat,
    last_lon,
    loc_id,
    frame_skip
):
    image_data = sorted(image_data, key=lambda x: x["filename"])

    trail_coordinates = []
    initial_ui_state = []

    for i in range(len(image_data)):
        asset = image_data[i]

        if asset["ext"] in ALLOWED_IMAGE_EXT:
            if i < len(image_data) - 1 and image_data[i + 1].get("lat", 0.0) != 0.0:
                heading = calculate_bearing(
                    asset["lat"],
                    asset["lon"],
                    image_data[i + 1]["lat"],
                    image_data[i + 1]["lon"]
                )
            else:
                heading = image_data[i - 1].get("heading", 0.0) if i > 0 else 0.0

            asset["heading"] = heading

            lat = asset["lat"]
            lon = asset["lon"]

            if lat != 0.0 and lon != 0.0:
                trail_coordinates.append([lon, lat])

                if last_lat != 0.0 and last_lon != 0.0:
                    dist = haversine_distance(last_lat, last_lon, lat, lon)

                    if dist > 50.0:
                        loc_id += 1

                last_lat, last_lon = lat, lon

            asset["location"] = f"Location {loc_id}"
            initial_ui_state.append(asset)

        elif asset["ext"] in ALLOWED_VIDEO_EXT:
            asset["location"] = f"Location {loc_id}"

            video_frames = get_video_frame_metadata(
                asset["path"],
                frame_skip,
                asset["original_name"],
                gps_snap
            )

            for vf in video_frames:
                vf["location"] = asset["location"]
                initial_ui_state.append(vf)

                if vf.get("lat", 0.0) != 0.0 and vf.get("lon", 0.0) != 0.0:
                    trail_coordinates.append([vf["lon"], vf["lat"]])
                    last_lat, last_lon = vf["lat"], vf["lon"]

    task_id = str(uuid.uuid4())
    active_tasks[task_id] = queue.Queue()

    total_est_frames = len(initial_ui_state)

    def process_worker(assets, t_id, height, snap, _is_360, f_skip):
        try:
            if global_model is None:
                active_tasks[t_id].put({
                    "type": "error",
                    "message": "No YOLO model is loaded."
                })
                return

            for asset in assets:
                def on_frame_processed(payload):
                    active_tasks[t_id].put({
                        "type": "update",
                        "data": payload
                    })

                if asset["ext"] in ALLOWED_VIDEO_EXT:
                    pitch_interp = extract_video_gpmf_pitch_track(asset["path"])

                    process_video_frames_async(
                        asset["path"],
                        global_model,
                        app.config["UPLOAD_FOLDER"],
                        height,
                        pitch_interp,
                        asset["filename"],
                        snap,
                        f_skip,
                        model_lock,
                        _is_360,
                        asset["location"],
                        on_frame_processed
                    )

                elif asset["ext"] in ALLOWED_IMAGE_EXT:
                    defects, geo_feats, base_filename = process_single_image(
                        asset["path"],
                        global_model,
                        asset["filename"],
                        app.config["UPLOAD_FOLDER"],
                        asset["lat"],
                        asset["lon"],
                        asset["heading"],
                        height,
                        asset["pitch"],
                        model_lock,
                        _is_360,
                        asset["original_name"]
                    )

                    result_payload = {
                        "original_name": asset["original_name"],
                        "filename": asset["filename"],
                        "lat": round(asset["lat"], 6),
                        "lon": round(asset["lon"], 6),
                        "pitch": round(asset["pitch"], 2),
                        "location": asset["location"],
                        "geojson": geo_feats,
                        "views": {}
                    }

                    views_list = ["front", "rear"] if _is_360 else ["front"]

                    for view in views_list:
                        result_payload["views"][view] = {
                            "raw_filename": f"raw_rect_{view}_{base_filename}",
                            "raw_bev_filename": f"raw_bev_{view}_{base_filename}",
                            "rect_url": f"/static/uploads/rect_{view}_{base_filename}",
                            "bev_url": f"/static/uploads/bev_{view}_{base_filename}",
                            "defects": defects.get(view, [])
                        }

                    if "rear" not in result_payload["views"]:
                        result_payload["views"]["rear"] = {
                            "raw_filename": "",
                            "raw_bev_filename": "",
                            "rect_url": "",
                            "bev_url": "",
                            "defects": []
                        }

                    active_tasks[t_id].put({
                        "type": "update",
                        "data": result_payload
                    })

            active_tasks[t_id].put({"type": "complete"})

        except Exception as e:
            print(f"[Processing Error] {e}")
            active_tasks[t_id].put({
                "type": "error",
                "message": str(e)
            })

    threading.Thread(
        target=process_worker,
        args=(image_data, task_id, cam_height, gps_snap, is_360, frame_skip),
        daemon=True
    ).start()

    initial_geojson = []

    if len(trail_coordinates) > 1:
        initial_geojson.append({
            "type": "Feature",
            "properties": {"type": "trail"},
            "geometry": {
                "type": "LineString",
                "coordinates": trail_coordinates
            }
        })

    return jsonify({
        "success": True,
        "task_id": task_id,
        "total_images": total_est_frames,
        "initial_state": initial_ui_state,
        "initial_trail": {
            "type": "FeatureCollection",
            "features": initial_geojson
        },
        "last_lat": last_lat,
        "last_lon": last_lon,
        "last_loc_id": loc_id
    })


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/process", methods=["POST"])
@app.route("/process_uploads", methods=["POST"])
def process():
    print("\nPROCESS UPLOADS STARTED")

    handle_model_upload(request)

    if global_model is None:
        return jsonify({"error": "No ML model loaded into memory"}), 400

    media_files = collect_uploaded_media(request)

    if not media_files:
        return jsonify({"error": "No images or videos selected"}), 400

    cam_height = safe_float(request.form.get("cam_height"), 1.6)
    gps_snap = form_bool("gps_snap", False)
    is_360 = form_bool("is_360", True)
    frame_skip = safe_int(request.form.get("frame_skip"), 30)

    last_lat = safe_float(request.form.get("last_lat"), 0.0)
    last_lon = safe_float(request.form.get("last_lon"), 0.0)
    loc_id = safe_int(request.form.get("last_loc_id"), 1)

    print("FILES RECEIVED:", [f.filename for f in media_files])
    print("FORM DATA:", dict(request.form))

    image_data = []

    for f in media_files:
        ext = Path(f.filename).suffix.lower()

        if ext not in ALLOWED_IMAGE_EXT and ext not in ALLOWED_VIDEO_EXT:
            print(f"[Skip] Unsupported file type: {f.filename}")
            continue

        filename = f"{int(time.time() * 1000)}_{secure_filename(f.filename)}"
        filepath = UPLOAD_FOLDER / filename
        f.save(filepath)

        file_meta = build_file_metadata(
            filepath=filepath,
            filename=filename,
            original_name=f.filename,
            ext=ext
        )

        image_data.append(file_meta)

    if not image_data:
        return jsonify({"error": "No supported image or video files were uploaded"}), 400

    return start_processing_job(
        image_data,
        cam_height,
        gps_snap,
        is_360,
        last_lat,
        last_lon,
        loc_id,
        frame_skip
    )


@app.route("/process_pipeline_folder", methods=["POST"])
@app.route("/scan_pipeline", methods=["POST"])
def process_pipeline_folder():
    print("\nSCAN PIPELINE STARTED")

    handle_model_upload(request)

    if global_model is None:
        return jsonify({"error": "No ML model loaded into memory"}), 400

    cam_height = safe_float(request.form.get("cam_height"), 1.6)
    gps_snap = form_bool("gps_snap", False)
    is_360 = form_bool("is_360", True)
    frame_skip = safe_int(request.form.get("frame_skip"), 30)

    last_lat = safe_float(request.form.get("last_lat"), 0.0)
    last_lon = safe_float(request.form.get("last_lon"), 0.0)
    loc_id = safe_int(request.form.get("last_loc_id"), 1)

    found_files = []

    for root_dir in PIPELINE_OUTPUT_ROOTS:
        frames_dir = root_dir / "frames"

        if frames_dir.exists():
            for ext in ALLOWED_IMAGE_EXT.union(ALLOWED_VIDEO_EXT):
                found_files.extend(list(frames_dir.rglob(f"*{ext}")))
                found_files.extend(list(frames_dir.rglob(f"*{ext.upper()}")))

            if found_files:
                print(f"[Pipeline] Using frames directory: {frames_dir}")
                break

    if not found_files:
        checked = "\n".join(str(root / "frames") for root in PIPELINE_OUTPUT_ROOTS)

        return jsonify({
            "error": (
                "No media files found in the Data_pipeline output frames directory.\n\n"
                f"Checked:\n{checked}"
            )
        }), 404

    image_data = []

    for filepath_obj in sorted(found_files):
        ext = filepath_obj.suffix.lower()

        if ext not in ALLOWED_IMAGE_EXT and ext not in ALLOWED_VIDEO_EXT:
            continue

        filename = secure_filename(filepath_obj.name)

        file_meta = build_file_metadata(
            filepath=filepath_obj,
            filename=filename,
            original_name=filepath_obj.name,
            ext=ext
        )

        image_data.append(file_meta)

    if not image_data:
        return jsonify({"error": "Pipeline folder exists, but no supported media was found"}), 400

    print(f"[Pipeline] Found {len(image_data)} media file(s)")

    return start_processing_job(
        image_data,
        cam_height,
        gps_snap,
        is_360,
        last_lat,
        last_lon,
        loc_id,
        frame_skip
    )


@app.route("/stream/<task_id>")
def stream(task_id):
    def event_stream():
        q = active_tasks.get(task_id)

        if not q:
            yield f"data: {json.dumps({'type': 'error', 'message': 'Invalid Task ID'})}\n\n"
            return

        while True:
            msg = q.get()

            yield f"data: {json.dumps(msg)}\n\n"

            if msg["type"] in ["complete", "error"]:
                active_tasks.pop(task_id, None)
                break

    return Response(event_stream(), mimetype="text/event-stream")


@app.route("/export-zip", methods=["POST"])
def export_zip():
    project_data = request.json.get("results", [])

    if not project_data:
        return jsonify({"error": "No data provided"}), 400

    memory_file = BytesIO()

    with zipfile.ZipFile(memory_file, "w", zipfile.ZIP_DEFLATED) as zf:
        for r in project_data:
            loc = r.get("location", "Unknown Location")

            for view in r.get("views", {}).keys():
                raw_filename = r["views"][view].get("raw_filename")

                if not raw_filename:
                    continue

                original_name = secure_filename(r.get("original_name", "unknown"))
                target_filename = f"{loc}/{view}/RAW_{original_name}"
                file_path = UPLOAD_FOLDER / raw_filename

                if file_path.exists():
                    zf.write(file_path, target_filename)

    memory_file.seek(0)

    return send_file(
        memory_file,
        download_name="DCPM_Export.zip",
        as_attachment=True
    )


@app.route("/export-flat-zip", methods=["POST"])
def export_flat_zip():
    project_data = request.json.get("results", [])

    if not project_data:
        return jsonify({"error": "No data provided"}), 400

    memory_file = BytesIO()

    with zipfile.ZipFile(memory_file, "w", zipfile.ZIP_DEFLATED) as zf:
        for r in project_data:
            loc = r.get("location", "Unknown Location")

            for view in r.get("views", {}).keys():
                raw_bev_filename = r["views"][view].get("raw_bev_filename")

                if not raw_bev_filename:
                    continue

                original_name = secure_filename(r.get("original_name", "unknown"))
                target_filename = f"{loc}/{view}/FLAT_{original_name}"
                file_path = UPLOAD_FOLDER / raw_bev_filename

                if file_path.exists():
                    zf.write(file_path, target_filename)

    memory_file.seek(0)

    return send_file(
        memory_file,
        download_name="DCPM_Flattened_Export.zip",
        as_attachment=True
    )


if __name__ == "__main__":
    app.run(debug=True, port=5000)