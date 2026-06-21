import cv2
import numpy as np
from ultralytics import YOLO
from ultralytics.utils.plotting import colors 
import math
import exifread
import os

def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371000.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2)**2
    return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1 - a))

def extract_photo_telemetry(filepath):
    try:
        from extract_gpmf import extract_jpeg_metadata_blocks, parse_xmp_gpano, parse_gpmf, extract_all_telemetry, flatten_global_ast
        xmp_raw, gpmf_raw = extract_jpeg_metadata_blocks(filepath)
        
        pitch, roll, fov, klns = None, None, None, None
        
        if gpmf_raw:
            ast = parse_gpmf(gpmf_raw)
            constants, _ = extract_all_telemetry(ast)
            global_constants = flatten_global_ast(ast)
            constants.update(global_constants)
            
            if 'GRAV' in constants:
                x, y, z = constants['GRAV']
                pitch = -math.degrees(math.atan2(z, y))
                roll = math.degrees(math.atan2(x, y))
            
            fov = constants.get('MFOV', None)
            
            if fov is None:
                zfov = constants.get('ZFOV')
                aruw = constants.get('ARUW')
                if zfov is not None and aruw is not None:
                    try:
                        zfov_rad = math.radians(float(zfov))
                        aruw_val = float(aruw)
                        tan_half_dfov = math.tan(zfov_rad / 2.0)
                        ratio = aruw_val / math.sqrt(aruw_val**2 + 1)
                        hfov_rad = 2.0 * math.atan(tan_half_dfov * ratio)
                        fov = math.degrees(hfov_rad)
                    except Exception: pass
            
            klns = constants.get('KLNS', None)

        if xmp_raw:
            gpano = parse_xmp_gpano(xmp_raw)
            if pitch is None and 'PosePitchDegrees' in gpano:
                pitch = float(gpano['PosePitchDegrees'])
            if roll is None and 'PoseRollDegrees' in gpano:
                roll = float(gpano['PoseRollDegrees'])
                
        return pitch, roll, klns, fov
    except Exception:
        pass
    return None, None, None, None

def get_exif_gps(filepath):
    try:
        with open(filepath, 'rb') as f:
            tags = exifread.process_file(f, details=False)
        if 'GPS GPSLatitude' in tags and 'GPS GPSLongitude' in tags:
            def convert_to_degrees(value):
                d, m, s = value.values
                return float(d.num)/d.den + (float(m.num)/m.den)/60.0 + (float(s.num)/s.den)/3600.0
            lat = convert_to_degrees(tags['GPS GPSLatitude'])
            if tags.get('GPS GPSLatitudeRef', None) and tags['GPS GPSLatitudeRef'].printable != 'N': lat = -lat
            lon = convert_to_degrees(tags['GPS GPSLongitude'])
            if tags.get('GPS GPSLongitudeRef', None) and tags['GPS GPSLongitudeRef'].printable != 'E': lon = -lon
            return lat, lon
    except Exception:
        pass
    return None, None

def calculate_bearing(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    x = math.sin(lon2 - lon1) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - (math.sin(lat1) * math.cos(lat2) * math.cos(lon2 - lon1))
    return (math.degrees(math.atan2(x, y)) + 360) % 360

def local_to_global(lat, lon, heading_deg, local_x, local_z):
    R = 6378137.0
    d = math.hypot(local_x, local_z)
    true_heading_rad = math.radians(heading_deg) + math.atan2(local_x, local_z)
    lat_rad, lon_rad = math.radians(lat), math.radians(lon)
    out_lat = math.asin(math.sin(lat_rad)*math.cos(d/R) + math.cos(lat_rad)*math.sin(d/R)*math.cos(true_heading_rad))
    out_lon = lon_rad + math.atan2(math.sin(true_heading_rad)*math.sin(d/R)*math.cos(lat_rad), math.cos(d/R) - math.sin(lat_rad)*math.sin(out_lat))
    return math.degrees(out_lat), math.degrees(out_lon)

def equirectangular_to_rectilinear(equi_img, fov_deg, pitch_deg, roll_deg, yaw_deg, output_width=1280, output_height=720):
    h, w = equi_img.shape[:2]
    f = (output_width / 2.0) / math.tan(math.radians(fov_deg) / 2.0)
    K = np.array([[f, 0, output_width / 2.0], [0, f, output_height / 2.0], [0, 0, 1]], dtype=np.float32)

    pitch, yaw, roll = math.radians(pitch_deg), math.radians(yaw_deg), math.radians(roll_deg)
    
    R_pitch = np.array([[1, 0, 0], [0, math.cos(pitch), -math.sin(pitch)], [0, math.sin(pitch), math.cos(pitch)]])
    R_yaw = np.array([[math.cos(yaw), 0, math.sin(yaw)], [0, 1, 0], [-math.sin(yaw), 0, math.cos(yaw)]])
    R_roll = np.array([[math.cos(roll), -math.sin(roll), 0], [math.sin(roll), math.cos(roll), 0], [0, 0, 1]])
    
    R_combined = R_yaw @ R_pitch @ R_roll 
    R_inv = np.linalg.inv(R_combined)

    x, y = np.meshgrid(np.arange(output_width), np.arange(output_height))
    pixels = np.stack((x, y, np.ones_like(x)), axis=-1).reshape(-1, 3).T
    rays = R_inv @ (np.linalg.inv(K) @ pixels)
    
    theta = np.arctan2(rays[0, :], rays[2, :])
    phi = np.arcsin(np.clip(rays[1, :] / np.linalg.norm(rays, axis=0), -1, 1))
    
    u, v = (theta / (2 * math.pi) + 0.5) * w, (phi / math.pi + 0.5) * h
    map_x, map_y = u.reshape((output_height, output_width)).astype(np.float32), v.reshape((output_height, output_width)).astype(np.float32)
    return cv2.remap(equi_img, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP), K

def get_bev_homography(K, cam_height_m, pitch_deg, roll_deg, gsd=0.01, z_near=2.0, z_far=8.0, x_range=3.0):
    pitch_rad, roll_rad = math.radians(-pitch_deg), math.radians(roll_deg)
    road_pts = np.array([[-x_range, z_near], [x_range, z_near], [x_range, z_far], [-x_range, z_far]], dtype=np.float32)
    
    bev_w, bev_h = int((2 * x_range) / gsd), int((z_far - z_near) / gsd)
    bev_pts = np.array([[0, bev_h], [bev_w, bev_h], [bev_w, 0], [0, 0]], dtype=np.float32)

    Rx = np.array([[1, 0, 0], [0, math.cos(pitch_rad), -math.sin(pitch_rad)], [0, math.sin(pitch_rad), math.cos(pitch_rad)]])
    Rz = np.array([[math.cos(roll_rad), -math.sin(roll_rad), 0], [math.sin(roll_rad), math.cos(roll_rad), 0], [0, 0, 1]])
    R = Rx @ Rz 

    rect_pts = []
    for pt in road_pts:
        xyz = R @ np.array([pt[0], cam_height_m, pt[1]])
        u = (K[0,0] * xyz[0] / xyz[2]) + K[0,2]
        v = (K[1,1] * xyz[1] / xyz[2]) + K[1,2]
        rect_pts.append([u, v])

    return cv2.getPerspectiveTransform(np.array(rect_pts, dtype=np.float32), bev_pts), bev_w, bev_h, gsd, x_range, z_far

def draw_bev_grid(img, K, cam_height_m, pitch_deg, roll_deg, z_near=2.0, z_far=8.0, x_range=3.0):
    pitch_rad, roll_rad = math.radians(-pitch_deg), math.radians(roll_deg)
    Rx = np.array([[1, 0, 0], [0, math.cos(pitch_rad), -math.sin(pitch_rad)], [0, math.sin(pitch_rad), math.cos(pitch_rad)]])
    Rz = np.array([[math.cos(roll_rad), -math.sin(roll_rad), 0], [math.sin(roll_rad), math.cos(roll_rad), 0], [0, 0, 1]])
    R = Rx @ Rz 
    
    for z in np.arange(math.floor(z_near), math.ceil(z_far) + 1, 1.0):
        pts = []
        for x in np.arange(-x_range, x_range + 0.5, 0.5):
            xyz = R @ np.array([x, cam_height_m, z])
            if xyz[2] <= 0: continue
            u = int((K[0,0] * xyz[0] / xyz[2]) + K[0,2])
            v = int((K[1,1] * xyz[1] / xyz[2]) + K[1,2])
            pts.append((u, v))
        if len(pts) > 1:
            for i in range(len(pts)-1):
                cv2.line(img, pts[i], pts[i+1], (0, 255, 255), 2)
                
    for x in np.arange(math.floor(-x_range), math.ceil(x_range) + 1, 1.0):
        pts = []
        for z in np.arange(z_near, z_far + 0.5, 0.5):
            xyz = R @ np.array([x, cam_height_m, z])
            if xyz[2] <= 0: continue
            u = int((K[0,0] * xyz[0] / xyz[2]) + K[0,2])
            v = int((K[1,1] * xyz[1] / xyz[2]) + K[1,2])
            pts.append((u, v))
        if len(pts) > 1:
            for i in range(len(pts)-1):
                cv2.line(img, pts[i], pts[i+1], (0, 255, 255), 2)
                
    return img

def process_single_image(img_input, model, base_filename, output_dir, gps_lat, gps_lon, heading, cam_height, pitch, roll, klns, fov_from_meta, model_lock, is_360=True, original_filename="", draw_grid=False):
    if fov_from_meta is None: raise ValueError("Missing Field of View (MFOV/ZFOV) in metadata.")
    if pitch is None or roll is None: raise ValueError("Missing Camera Pose (GRAV/GPano) in metadata.")
    if gps_lat is None or gps_lon is None: raise ValueError("Missing GPS Coordinates.")

    fov_val = float(fov_from_meta)
    if fov_val <= 0 or fov_val >= 180:
        raise ValueError(f"Invalid calculated FOV value: {fov_val}")

    img_mat = cv2.imread(img_input) if isinstance(img_input, str) else img_input
    all_defects, all_geojson_features, bev_footprints = {}, [], {}
    
    views_to_process = {'front': {'yaw': 0, 'heading_offset': 0}}
    if is_360: views_to_process['rear'] = {'yaw': 180, 'heading_offset': 180}
        
    for view_name, config in views_to_process.items():
        all_defects[view_name] = []
        
        if is_360:
            rect_img, K = equirectangular_to_rectilinear(img_mat, fov_deg=fov_val, pitch_deg=pitch, roll_deg=roll, yaw_deg=config['yaw'])
        else:
            rect_img = img_mat.copy()
            h, w = rect_img.shape[:2]
            f = (w / 2.0) / math.tan(math.radians(fov_val) / 2.0)
            K = np.array([[f, 0, w / 2.0], [0, f, h / 2.0], [0, 0, 1]], dtype=np.float32)
            
            if klns and len(klns) >= 5:
                try:
                    dist_coeffs = np.array(klns[1:6], dtype=np.float32)
                    K_undist = np.array([[klns[0], 0, w/2], [0, klns[0], h/2], [0, 0, 1]], dtype=np.float32)
                    rect_img = cv2.undistort(rect_img, K_undist, dist_coeffs)
                    K = K_undist
                except: pass

        raw_rect_filename = f"raw_rect_{view_name}_{base_filename}"
        cv2.imwrite(os.path.join(output_dir, raw_rect_filename), rect_img)
        
        with model_lock: results = model.predict(source=rect_img, conf=0.25, save=False, verbose=False)
            
        H_mat, bev_w, bev_h, gsd, x_range, z_far = get_bev_homography(K, cam_height, pitch, roll)
        bev_img = cv2.warpPerspective(rect_img, H_mat, (bev_w, bev_h))
        
        cv2.imwrite(os.path.join(output_dir, f"raw_bev_{view_name}_{base_filename}"), bev_img.copy())
        view_heading = (heading + config['heading_offset']) % 360

        z_near = z_far - (bev_h * gsd)
        bev_center_lat, bev_center_lon = local_to_global(gps_lat, gps_lon, view_heading, 0, (z_near + z_far) / 2.0)
        bev_footprints[view_name] = {"lat": bev_center_lat, "lon": bev_center_lon, "heading": view_heading, "width_m": 2 * x_range, "height_m": z_far - z_near}

        annotated_rect = rect_img.copy()
        
        if draw_grid:
            annotated_rect = draw_bev_grid(annotated_rect, K, cam_height, pitch, roll, z_near, z_far, x_range)

        for r in results:
            annotated_rect = r.plot(img=annotated_rect)
            if r.masks is not None:
                for i, mask_pts in enumerate(r.masks.xy):
                    cls_id, conf = int(r.boxes.cls[i]), float(r.boxes.conf[i])
                    class_name = model.names[cls_id]
                    mask_color_bgr = colors(cls_id, bgr=True)
                    hex_color = f"#{int(mask_color_bgr[2]):02x}{int(mask_color_bgr[1]):02x}{int(mask_color_bgr[0]):02x}"
                    
                    mask_canvas = np.zeros(rect_img.shape[:2], dtype=np.uint8)
                    cv2.fillPoly(mask_canvas, [np.array(mask_pts, dtype=np.int32)], 255)
                    contours, _ = cv2.findContours(cv2.warpPerspective(mask_canvas, H_mat, (bev_w, bev_h)), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    
                    for contour in contours:
                        area_sqm = cv2.contourArea(contour) * (gsd ** 2)
                        if area_sqm <= 0.0001: continue
                            
                        cv2.fillPoly(bev_img, [contour], color=mask_color_bgr)
                        cv2.addWeighted(bev_img, conf, bev_img, 1.0 - conf, 0, bev_img)
                        
                        geo_coords = [[local_to_global(gps_lat, gps_lon, view_heading, (pt[0][0]*gsd)-x_range, z_far-(pt[0][1]*gsd))[1], local_to_global(gps_lat, gps_lon, view_heading, (pt[0][0]*gsd)-x_range, z_far-(pt[0][1]*gsd))[0]] for pt in contour]
                        if geo_coords[0] != geo_coords[-1]: geo_coords.append(geo_coords[0])

                        all_defects[view_name].append({"class": class_name, "conf": round(conf, 2), "area_sqm": round(area_sqm, 4), "color": hex_color})
                        all_geojson_features.append({"type": "Feature", "properties": {"class": class_name, "area_sqm": round(area_sqm, 4), "view": view_name, "color": hex_color, "filename": original_filename, "conf": round(conf, 2)}, "geometry": {"type": "Polygon", "coordinates": [geo_coords]}})

        cv2.imwrite(os.path.join(output_dir, f"rect_{view_name}_{base_filename}"), annotated_rect)
        cv2.imwrite(os.path.join(output_dir, f"bev_{view_name}_{base_filename}"), bev_img)
        
    return all_defects, all_geojson_features, base_filename, bev_footprints

def process_video_frames_async(video_path, model, upload_dir, cam_height, file_name, original_name, gps_snap, interval_m, model_lock, is_360, location_str, callback, draw_grid=False):
    cap = cv2.VideoCapture(video_path)
    base_stem = os.path.splitext(file_name)[0]
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    
    try:
        from extract_gpmf import extract_streams_with_time, get_telemetry_interpolators, evaluate_telemetry_health
        streams, constants = extract_streams_with_time(video_path)
        
        # Trigger health assessment on the raw streams
        health_report = evaluate_telemetry_health(streams)
        callback({"type": "health_report", "is_video": True, "original_name": original_name, "data": health_report})
        
        interpolators = get_telemetry_interpolators(streams)
        gps_interp = interpolators.get("gps")
        speed_interp = interpolators.get("speed")
        pitch_interp = interpolators.get("pitch")
        roll_interp = interpolators.get("roll")
        klns = constants.get('KLNS', None)
        
        fov_from_meta = constants.get('MFOV', None)
        if fov_from_meta is None:
            zfov = constants.get('ZFOV')
            aruw = constants.get('ARUW')
            if zfov is not None and aruw is not None:
                try:
                    zfov_rad = math.radians(float(zfov))
                    aruw_val = float(aruw)
                    tan_half_dfov = math.tan(zfov_rad / 2.0)
                    ratio = aruw_val / math.sqrt(aruw_val**2 + 1)
                    hfov_rad = 2.0 * math.atan(tan_half_dfov * ratio)
                    fov_from_meta = math.degrees(hfov_rad)
                except Exception: pass
                
    except Exception as e:
        callback({"error": f"Failed to parse GPMF for video: {str(e)}", "is_video": True, "original_name": original_name})
        cap.release()
        return

    if not all([gps_interp, speed_interp, pitch_interp, roll_interp, fov_from_meta]):
        callback({"error": "Missing required GPMF telemetry streams (GPS, Speed, GRAV, or computed FOV).", "is_video": True, "original_name": original_name})
        cap.release()
        return

    dist_accum, last_time = 0.0, 0.0
    frame_idx = -1
    
    while True:
        ret, frame = cap.read()
        if not ret: break
        frame_idx += 1
        elapsed_sec = frame_idx / fps
        
        dist_accum += float(speed_interp(elapsed_sec)) * (elapsed_sec - last_time)
        last_time = elapsed_sec

        if dist_accum >= interval_m or frame_idx == 0:
            dist_accum = 0.0 
            
            current_pitch = float(pitch_interp(elapsed_sec))
            current_roll = float(roll_interp(elapsed_sec))

            try:
                c_loc = gps_interp(elapsed_sec)
                n_loc = gps_interp(elapsed_sec + 0.5)
                current_lat, current_lon = float(c_loc[0]), float(c_loc[1])
                current_heading = calculate_bearing(current_lat, current_lon, float(n_loc[0]), float(n_loc[1]))
            except Exception: continue

            frame_base_name = f"fr{frame_idx}_{base_stem}.jpg"
            original_frame_name = f"{original_name} (Frame {frame_idx})"

            try:
                defects, geo_feats, _, footprints = process_single_image(
                    frame, model, frame_base_name, upload_dir,
                    current_lat, current_lon, current_heading, cam_height, current_pitch, current_roll, klns, fov_from_meta, model_lock, is_360, original_frame_name, draw_grid
                )
            except Exception as e:
                callback({"error": str(e), "is_video": False, "original_name": original_frame_name})
                continue
            
            result_payload = {"original_name": original_frame_name, "filename": frame_base_name, "lat": round(current_lat, 6), "lon": round(current_lon, 6), "pitch": round(current_pitch, 2), "roll": round(current_roll, 2), "location": location_str, "geojson": geo_feats, "views": {}}
            for view in (['front', 'rear'] if is_360 else ['front']):
                result_payload["views"][view] = {"raw_filename": f"raw_rect_{view}_{frame_base_name}", "raw_bev_filename": f"raw_bev_{view}_{frame_base_name}", "raw_bev_url": f"/static/uploads/raw_bev_{view}_{frame_base_name}", "rect_url": f"/static/uploads/rect_{view}_{frame_base_name}", "bev_url": f"/static/uploads/bev_{view}_{frame_base_name}", "defects": defects[view], "footprint": footprints[view]}
            callback(result_payload)

    cap.release()

def get_video_frame_metadata(video_path, interval_m, original_name, gps_snap):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): return []
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    
    try:
        from extract_gpmf import extract_streams_with_time, get_telemetry_interpolators
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