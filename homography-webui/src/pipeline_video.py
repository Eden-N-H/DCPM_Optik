import os
import cv2
import json
import math
from utils import sanitize_meta
from geo_math import calculate_bearing
from parser_gpmf import extract_streams_with_time
from telemetry import evaluate_telemetry_health, get_telemetry_interpolators
from pipeline_image import process_single_image

def process_video_frames_async(video_path, model, upload_dir, file_name, original_name, location_str, options, model_lock, callback, is_cancelled=None, sam2_predictor=None):
    cap = cv2.VideoCapture(video_path)
    base_stem = os.path.splitext(file_name)[0]
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    
    interval_m = options.get('interval_m', 2.0)
    
    try:
        streams, constants = extract_streams_with_time(video_path)
        health_report = evaluate_telemetry_health(streams)
        callback({"type": "health_report", "is_video": True, "original_name": original_name, "data": health_report})
        
        interpolators = get_telemetry_interpolators(streams)
        gps_interp = interpolators.get("gps")
        speed_interp = interpolators.get("speed")
        pitch_interp = interpolators.get("pitch")
        roll_interp = interpolators.get("roll")
        pitch_base_interp = interpolators.get("pitch_base")
        roll_base_interp = interpolators.get("roll_base")
        
        klns = constants.get('KLNS', None)
        
        fov_from_meta = constants.get('XFOV', None)
        if fov_from_meta is None:
            zfov, aruw = constants.get('ZFOV'), constants.get('ARUW')
            if zfov is not None and aruw is not None:
                try: fov_from_meta = math.degrees(2.0 * math.atan(math.tan(math.radians(float(zfov)) / 2.0) * (float(aruw) / math.sqrt(float(aruw)**2 + 1))))
                except Exception: pass
    except Exception as e:
        callback({"error": f"Failed to parse GPMF for video: {str(e)}", "is_video": True, "original_name": original_name})
        cap.release()
        return

    if options.get('is_360', True) and fov_from_meta is None: fov_from_meta = 100.0

    if not all([gps_interp, speed_interp, pitch_interp, roll_interp, fov_from_meta]):
        callback({"error": "Missing required GPMF telemetry streams (GPS, Speed, GRAV, or computed FOV).", "is_video": True, "original_name": original_name})
        cap.release()
        return

    dist_accum, last_time = 0.0, 0.0
    frame_idx = -1
    
    while True:
        if is_cancelled and is_cancelled():
            callback({"type": "cancelled", "is_video": True, "original_name": original_name})
            break

        ret, frame = cap.read()
        if not ret: break
        frame_idx += 1
        elapsed_sec = frame_idx / fps
        
        dist_accum += float(speed_interp(elapsed_sec)) * (elapsed_sec - last_time)
        last_time = elapsed_sec

        if dist_accum >= interval_m or frame_idx == 0:
            dist_accum = 0.0 
            current_pitch = float(pitch_interp(elapsed_sec)) if pitch_interp else 0.0
            current_roll = float(roll_interp(elapsed_sec)) if roll_interp else 0.0
            current_base_pitch = float(pitch_base_interp(elapsed_sec)) if pitch_base_interp else current_pitch
            current_base_roll = float(roll_base_interp(elapsed_sec)) if roll_base_interp else current_roll

            try:
                c_loc = gps_interp(elapsed_sec)
                n_loc = gps_interp(elapsed_sec + 0.5)
                current_lat, current_lon = float(c_loc[0]), float(c_loc[1])
                current_heading = calculate_bearing(current_lat, current_lon, float(n_loc[0]), float(n_loc[1]))
            except Exception: continue

            frame_base_name = f"fr{frame_idx}_{base_stem}.jpg"
            original_frame_name = f"{original_name} (Frame {frame_idx})"
            
            frame_meta = {
                "Video_Global_GPMF": sanitize_meta(constants),
                "Frame_Telemetry": sanitize_meta({
                    "Timestamp_sec": elapsed_sec, "Latitude": current_lat, "Longitude": current_lon,
                    "Heading": current_heading, "Pitch_Inst": current_pitch, "Pitch_Base": current_base_pitch,
                    "Roll_Inst": current_roll, "Roll_Base": current_base_roll,
                    "Speed_ms": float(speed_interp(elapsed_sec)) if speed_interp else None,
                    "FOV": fov_from_meta, "KLNS": klns
                })
            }
            with open(os.path.join(upload_dir, f"meta_{frame_base_name}.json"), 'w') as mf:
                json.dump(frame_meta, mf, indent=2)

            telemetry = {
                "lat": current_lat,
                "lon": current_lon,
                "heading": current_heading,
                "pitch": current_pitch,
                "roll": current_roll,
                "base_pitch": current_base_pitch,
                "base_roll": current_base_roll,
                "klns": klns,
                "fov": fov_from_meta
            }

            try:
                defects, geo_feats, gen_files, footprints, view_meta, calibrations = process_single_image(
                    frame, model, frame_base_name, upload_dir, telemetry, options, model_lock, original_frame_name,
                    sam2_predictor=sam2_predictor
                )
                
                process_meta_data = {
                    "telemetry": telemetry,
                    "options": options,
                    "view_meta": view_meta,
                    "original_name": original_frame_name
                }
                with open(os.path.join(upload_dir, f"process_meta_{frame_base_name}.json"), 'w') as f:
                    json.dump(process_meta_data, f)
                    
            except Exception as e:
                callback({"error": str(e), "is_video": False, "original_name": original_frame_name})
                continue
            
            result_payload = {
                "original_name": original_frame_name,
                "filename": frame_base_name,
                "lat": round(current_lat, 6),
                "lon": round(current_lon, 6),
                "pitch": round(current_pitch, 2),
                "roll": round(current_roll, 2),
                "location": location_str,
                "geojson": geo_feats,
                "views": {}
            }
            
            for view in (['front', 'rear'] if options.get('is_360', True) else ['front']):
                gf = gen_files[view]
                result_payload["views"][view] = {
                    "calibration": calibrations[view],
                    "raw_filename": gf["raw_rect"],
                    "raw_bev_filename": gf["raw_bev"], 
                    "raw_bev_url": f"/static/uploads/{gf['raw_bev']}",
                    "rect_url": f"/static/uploads/{gf['rect']}", 
                    "bev_url": f"/static/uploads/{gf['bev']}",
                    "edit_bev_url": f"/static/uploads/{gf.get('edit_bev', gf['raw_bev'])}",
                    "defects": defects[view],
                    "footprint": footprints[view]
                }
            callback(result_payload)

    cap.release()

def get_video_frame_metadata(video_path, interval_m, original_name):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): return []
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    try:
        streams, _ = extract_streams_with_time(video_path)
        interpolators = get_telemetry_interpolators(streams)
        gps_interp = interpolators.get("gps")
        speed_interp = interpolators.get("speed")
    except Exception: return []
    if not gps_interp or not speed_interp: return []
    
    frames_meta = []
    dist_accum, last_time = 0.0, 0.0
    
    for frame_idx in range(total_frames):
        elapsed_sec = frame_idx / fps
        dist_accum += float(speed_interp(elapsed_sec)) * (elapsed_sec - last_time)
        last_time = elapsed_sec
        if dist_accum >= interval_m or frame_idx == 0:
            dist_accum = 0.0
            try:
                loc = gps_interp(elapsed_sec)
                frames_meta.append({"original_name": f"{original_name} (Frame {frame_idx})", "lat": float(loc[0]), "lon": float(loc[1])})
            except Exception: pass
            
    return frames_meta