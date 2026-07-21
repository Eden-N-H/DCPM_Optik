import os
import cv2
import json
import math
import numpy as np
from utils import sanitize_meta, atomic_write_json
from geo_math import calculate_bearing, apply_camera_offset
from parser_gpmf import extract_streams_with_time
from telemetry import evaluate_telemetry_health, get_telemetry_interpolators
from pipeline_image import process_single_image

def process_video_frames_async(video_path, model, upload_dir, file_name, original_name, location_str, options, model_lock, callback, is_cancelled=None, sam2_predictor=None, sam2_lock=None):
    cap = cv2.VideoCapture(video_path)
    base_stem = os.path.splitext(file_name)[0]
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    
    interval_m = options.get('interval_m', 2.0)
    gps_lag = options.get('gps_lag_sec', 0.8)
    cam_off_fwd = options.get('cam_offset_forward_m', 0.0) or 0.0
    cam_off_right = options.get('cam_offset_right_m', 0.0) or 0.0
    
    try:
        streams, constants = extract_streams_with_time(video_path)
        health_report = evaluate_telemetry_health(streams)
        callback({"type": "health_report", "is_video": True, "original_name": original_name, "data": health_report})
        
        interpolators = get_telemetry_interpolators(streams)
        gps_interp = interpolators.get("gps")
        speed_interp = interpolators.get("speed")
        heading_interp = interpolators.get("heading")
        
        grav_x_interp = interpolators.get("grav_x")
        grav_y_interp = interpolators.get("grav_y")
        grav_z_interp = interpolators.get("grav_z")
        
        klns = constants.get('KLNS', None)
        
        xfov_from_meta = constants.get('XFOV', None)
        yfov_from_meta = constants.get('YFOV', None)
        
        if xfov_from_meta is None:
            zfov, aruw = constants.get('ZFOV'), constants.get('ARUW')
            if zfov is not None and aruw is not None:
                try: xfov_from_meta = math.degrees(2.0 * math.atan(math.tan(math.radians(float(zfov)) / 2.0) * (float(aruw) / math.sqrt(float(aruw)**2 + 1))))
                except Exception: pass
    except Exception as e:
        callback({"error": f"Failed to parse GPMF for video: {str(e)}", "is_video": True, "original_name": original_name})
        cap.release()
        return

    if options.get('is_360', True) and xfov_from_meta is None: xfov_from_meta = 100.0

    if not all([gps_interp, speed_interp, heading_interp, grav_x_interp, grav_y_interp, grav_z_interp, xfov_from_meta]):
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
        
        msec = cap.get(cv2.CAP_PROP_POS_MSEC)
        if msec > 0:
            elapsed_sec = msec / 1000.0
        else:
            elapsed_sec = frame_idx / fps
            
        speed_val = max(0.0, float(speed_interp(elapsed_sec + gps_lag)))
        dist_accum += speed_val * (elapsed_sec - last_time)
        last_time = elapsed_sec

        if dist_accum >= interval_m or frame_idx == 0:
            if frame_idx > 0:
                intervals_consumed = max(1, int(dist_accum // interval_m))
                dist_accum -= interval_m * intervals_consumed
            else:
                dist_accum = 0.0 
            
            sample_time = elapsed_sec + gps_lag
            current_grav = [float(grav_x_interp(sample_time)), float(grav_y_interp(sample_time)), float(grav_z_interp(sample_time))]
            gx, gy, gz = current_grav
            
            norm = math.sqrt(gx*gx + gy*gy + gz*gz)
            if norm > 1e-6:
                gx, gy, gz = gx/norm, gy/norm, gz/norm
                
            current_pitch = math.degrees(math.asin(np.clip(-gz, -1.0, 1.0)))
            current_roll = math.degrees(math.atan2(gx, gy))

            try:
                c_loc = gps_interp(sample_time)
                raw_lat, raw_lon = float(c_loc[0]), float(c_loc[1])
                current_heading = float(heading_interp(sample_time))
            except Exception: continue

            if cam_off_fwd or cam_off_right:
                current_lat, current_lon = apply_camera_offset(raw_lat, raw_lon, current_heading, cam_off_right, cam_off_fwd)
            else:
                current_lat, current_lon = raw_lat, raw_lon

            frame_base_name = f"fr{frame_idx}_{base_stem}.jpg"
            original_frame_name = f"{original_name} (Frame {frame_idx})"
            
            frame_meta = {
                "Video_Global_GPMF": sanitize_meta(constants),
                "Frame_Telemetry": sanitize_meta({
                    "Timestamp_sec": elapsed_sec,
                    "GPS_Sample_Time_sec": sample_time,
                    "Raw_GPS_Latitude": raw_lat,
                    "Raw_GPS_Longitude": raw_lon,
                    "Latitude": current_lat, 
                    "Longitude": current_lon,
                    "Camera_Offset_Forward_m": cam_off_fwd,
                    "Camera_Offset_Right_m": cam_off_right,
                    "Heading": current_heading,
                    "Grav_Vec": current_grav,
                    "Speed_ms": speed_val,
                    "XFOV": xfov_from_meta,
                    "YFOV": yfov_from_meta,
                    "KLNS": klns,
                    "Pitch_UI": current_pitch,
                    "Roll_UI": current_roll
                })
            }
            atomic_write_json(os.path.join(upload_dir, f"meta_{frame_base_name}.json"), frame_meta, indent=2)

            telemetry = {
                "lat": current_lat,
                "lon": current_lon,
                "raw_lat": raw_lat,
                "raw_lon": raw_lon,
                "heading": current_heading,
                "grav_vec": current_grav,
                "klns": klns,
                "xfov": xfov_from_meta,
                "yfov": yfov_from_meta,
                "pitch": current_pitch,
                "roll": current_roll,
                "gps_lag_sec": gps_lag,
                "gps_sample_time_sec": sample_time,
                "frame_timestamp_sec": elapsed_sec,
                "cam_offset_forward_m": cam_off_fwd,
                "cam_offset_right_m": cam_off_right
            }

            try:
                defects, geo_feats, gen_files, footprints, view_meta, calibrations = process_single_image(
                    frame, model, frame_base_name, upload_dir, telemetry, options, model_lock, original_frame_name,
                    sam2_predictor=sam2_predictor, sam2_lock=sam2_lock
                )
                
                process_meta_data = {
                    "telemetry": telemetry,
                    "options": options,
                    "view_meta": view_meta,
                    "original_name": original_frame_name
                }
                atomic_write_json(os.path.join(upload_dir, f"process_meta_{frame_base_name}.json"), process_meta_data)
                    
            except Exception as e:
                callback({"error": str(e), "is_video": False, "original_name": original_frame_name})
                continue
            
            result_payload = {
                "original_name": original_frame_name,
                "filename": frame_base_name,
                "lat": current_lat,
                "lon": current_lon,
                "pitch": current_pitch,
                "roll": current_roll,
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

def get_video_frame_metadata(video_path, options, original_name):
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
        heading_interp = interpolators.get("heading")
    except Exception: return []
    if not gps_interp or not speed_interp: return []
    
    interval_m = options.get('interval_m', 2.0)
    gps_lag = options.get('gps_lag_sec', 0.8)
    cam_off_fwd = options.get('cam_offset_forward_m', 0.0) or 0.0
    cam_off_right = options.get('cam_offset_right_m', 0.0) or 0.0
    
    frames_meta = []
    dist_accum, last_time = 0.0, 0.0
    
    for frame_idx in range(total_frames):
        elapsed_sec = frame_idx / fps
        speed_val = max(0.0, float(speed_interp(elapsed_sec + gps_lag)))
        dist_accum += speed_val * (elapsed_sec - last_time)
        last_time = elapsed_sec
        
        if dist_accum >= interval_m or frame_idx == 0:
            if frame_idx > 0:
                intervals_consumed = max(1, int(dist_accum // interval_m))
                dist_accum -= interval_m * intervals_consumed
            else:
                dist_accum = 0.0
                
            try:
                sample_time = elapsed_sec + gps_lag
                loc = gps_interp(sample_time)
                lat, lon = float(loc[0]), float(loc[1])
                if (cam_off_fwd or cam_off_right) and heading_interp is not None:
                    heading = float(heading_interp(sample_time))
                    lat, lon = apply_camera_offset(lat, lon, heading, cam_off_right, cam_off_fwd)
                frames_meta.append({"original_name": f"{original_name} (Frame {frame_idx})", "lat": lat, "lon": lon})
            except Exception: pass
            
    return frames_meta
