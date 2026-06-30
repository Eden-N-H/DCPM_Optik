import cv2
import numpy as np
from ultralytics import YOLO
from ultralytics.utils.plotting import colors 
import math
import exifread
import os
import json

def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371000.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2)**2
    return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1 - a))

def sanitize_meta(obj):
    if isinstance(obj, dict):
        return {str(k): sanitize_meta(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sanitize_meta(v) for v in obj]
    elif isinstance(obj, tuple):
        return [sanitize_meta(v) for v in obj]
    elif isinstance(obj, bytes):
        if len(obj) > 1024:
            return f"<binary data: {len(obj)} bytes>"
        try:
            return obj.decode('utf-8', errors='ignore')
        except:
            return f"<binary data: {len(obj)} bytes>"
    elif isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    elif hasattr(obj, 'tolist') and callable(obj.tolist):
        return sanitize_meta(obj.tolist())
    elif hasattr(obj, 'item') and callable(obj.item):
        return sanitize_meta(obj.item())
    elif hasattr(obj, 'printable'):
        return str(obj.printable)
    return obj

def extract_full_photo_metadata(filepath):
    lat, lon = None, None
    exif_dict = {}
    try:
        with open(filepath, 'rb') as f:
            tags = exifread.process_file(f, details=False)
        for tag, val in tags.items():
            if tag.startswith('JPEG') or tag.startswith('Thumbnail') or tag.startswith('EXIF MakerNote'):
                continue
            exif_dict[tag] = str(val.printable) if hasattr(val, 'printable') else str(val)
            
        if 'GPS GPSLatitude' in tags and 'GPS GPSLongitude' in tags:
            def convert_to_degrees(value):
                d, m, s = value.values
                return float(d.num)/d.den + (float(m.num)/m.den)/60.0 + (float(s.num)/s.den)/3600.0
            lat = convert_to_degrees(tags['GPS GPSLatitude'])
            if tags.get('GPS GPSLatitudeRef') and tags['GPS GPSLatitudeRef'].printable != 'N': lat = -lat
            lon = convert_to_degrees(tags['GPS GPSLongitude'])
            if tags.get('GPS GPSLongitudeRef') and tags['GPS GPSLongitudeRef'].printable != 'E': lon = -lon
            exif_dict['Parsed_Latitude'] = lat
            exif_dict['Parsed_Longitude'] = lon
    except Exception:
        pass

    pitch, roll, fov, klns = None, None, None, None
    xmp_dict, gpmf_dict = {}, {}
    try:
        from extract_gpmf import extract_jpeg_metadata_blocks, parse_xmp_gpano, parse_gpmf, extract_all_telemetry, flatten_global_ast
        xmp_raw, gpmf_raw = extract_jpeg_metadata_blocks(filepath)
        
        if xmp_raw:
            xmp_dict = parse_xmp_gpano(xmp_raw)
            if 'PosePitchDegrees' in xmp_dict: pitch = float(xmp_dict['PosePitchDegrees'])
            if 'PoseRollDegrees' in xmp_dict: roll = float(xmp_dict['PoseRollDegrees'])

        if gpmf_raw:
            ast = parse_gpmf(gpmf_raw)
            constants, _ = extract_all_telemetry(ast)
            global_constants = flatten_global_ast(ast)
            constants.update(global_constants)
            gpmf_dict = constants
            
            if 'GRAV' in constants:
                x, y, z = constants['GRAV']
                if pitch is None: pitch = -math.degrees(math.atan2(z, y))
                if roll is None: roll = math.degrees(math.atan2(x, y))
            
            fov = constants.get('MFOV', None)
            if fov is None:
                zfov = constants.get('ZFOV')
                aruw = constants.get('ARUW')
                if zfov is not None and aruw is not None:
                    try:
                        zfov_rad = math.radians(float(zfov))
                        aruw_val = float(aruw)
                        fov = math.degrees(2.0 * math.atan(math.tan(zfov_rad / 2.0) * (aruw_val / math.sqrt(aruw_val**2 + 1))))
                    except Exception: pass
            
            klns = constants.get('KLNS', None)
    except Exception:
        pass

    full_meta = {
        "EXIF": sanitize_meta(exif_dict),
        "XMP_GPano": sanitize_meta(xmp_dict),
        "GPMF": sanitize_meta(gpmf_dict),
        "Computed_Variables": {
            "Latitude": lat,
            "Longitude": lon,
            "Pitch": pitch,
            "Roll": roll,
            "FOV": fov,
            "KLNS": klns
        }
    }
    
    return lat, lon, pitch, roll, klns, fov, full_meta

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

# --- NEW IMPROVED VISUAL & GIMBAL FUNCTIONS ---

def apply_bev_feathering(bev_bgr):
    h, w = bev_bgr.shape[:2]
    rgba = cv2.cvtColor(bev_bgr, cv2.COLOR_BGR2BGRA)
    alpha = np.ones((h, w), dtype=np.float32)
    
    top_fade = int(h * 0.3)
    for y in range(top_fade): alpha[y, :] *= (y / top_fade) ** 2.0
        
    side_fade = int(w * 0.15)
    for x in range(side_fade):
        fade_val = (x / side_fade) ** 1.5
        alpha[:, x] *= fade_val
        alpha[:, w - 1 - x] *= fade_val
        
    rgba[:, :, 3] = (alpha * 255).astype(np.uint8)
    return rgba

def apply_ego_mask(img, mask_pct=0.15):
    h, w = img.shape[:2]
    mask_h = int(h * mask_pct)
    img[h - mask_h:, :] = 0
    return img

def digital_gimbal_warp(img, K, delta_pitch, delta_roll):
    h, w = img.shape[:2]
    f = K[0,0]
    dy = f * math.tan(math.radians(delta_pitch))
    
    M = cv2.getRotationMatrix2D((K[0,2], K[1,2]), -delta_roll, 1.0)
    M[1, 2] += dy
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

def get_bev_homography(K, cam_height_m, pitch_deg, roll_deg, gsd=0.01):
    z_near = max(1.5, cam_height_m * 1.2)
    z_far = min(12.0, z_near + 8.0)
    x_range = 4.0 
    
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
        if xyz[2] <= 1e-5: xyz[2] = 1e-5 
        u = (K[0,0] * xyz[0] / xyz[2]) + K[0,2]
        v = (K[1,1] * xyz[1] / xyz[2]) + K[1,2]
        rect_pts.append([u, v])

    H = cv2.getPerspectiveTransform(np.array(rect_pts, dtype=np.float32), bev_pts)
    return H, bev_w, bev_h, gsd, x_range, z_far, z_near

def draw_bev_grid(img, K, cam_height_m, pitch_deg, roll_deg, z_near, z_far, x_range):
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
            for i in range(len(pts)-1): cv2.line(img, pts[i], pts[i+1], (0, 255, 255), 2)
                
    for x in np.arange(math.floor(-x_range), math.ceil(x_range) + 1, 1.0):
        pts = []
        for z in np.arange(z_near, z_far + 0.5, 0.5):
            xyz = R @ np.array([x, cam_height_m, z])
            if xyz[2] <= 0: continue
            u = int((K[0,0] * xyz[0] / xyz[2]) + K[0,2])
            v = int((K[1,1] * xyz[1] / xyz[2]) + K[1,2])
            pts.append((u, v))
        if len(pts) > 1:
            for i in range(len(pts)-1): cv2.line(img, pts[i], pts[i+1], (0, 255, 255), 2)
                
    return img

def process_single_image(img_input, model, base_filename, output_dir, gps_lat, gps_lon, heading, cam_height, pitch, roll, base_pitch, base_roll, klns, fov_from_meta, model_lock, is_360=True, original_filename="", draw_grid=False):
    if fov_from_meta is None or pitch is None or roll is None or gps_lat is None or gps_lon is None:
        raise ValueError("Missing critical metadata (GPS, FOV, or Pose).")

    fov_val = float(fov_from_meta)
    img_mat = cv2.imread(img_input) if isinstance(img_input, str) else img_input
    base_name_no_ext = os.path.splitext(base_filename)[0]
    
    all_defects, all_geojson_features, generated_files, bev_footprints = {}, [], {}, {}
    views_to_process = {'front': {'yaw': 0, 'heading_offset': 0}}
    if is_360: views_to_process['rear'] = {'yaw': 180, 'heading_offset': 180}
        
    for view_name, config in views_to_process.items():
        all_defects[view_name] = []
        
        # 1. Image Extraction, Gimbaling, & Camera Matrix
        if is_360:
            # For FFmpeg raw stitched files, we pass the IMU pitch/roll into the projector to stabilize it mathematically
            rect_img, K = equirectangular_to_rectilinear(img_mat, fov_deg=fov_val, pitch_deg=pitch, roll_deg=roll, yaw_deg=config['yaw'])
            # Since the image is now stabilized relative to gravity, the BEV projection uses perfectly flat inputs (0,0)
            bev_pitch, bev_roll = 0.0, 0.0
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
            
            rect_img = digital_gimbal_warp(rect_img, K, pitch - base_pitch, roll - base_roll)
            bev_pitch, bev_roll = base_pitch, base_roll

        # 2. Get Dynamic ROI geometry
        H_mat, bev_w, bev_h, gsd, x_range, z_far, z_near = get_bev_homography(K, cam_height, bev_pitch, bev_roll)
        
        raw_rect_filename = f"raw_rect_{view_name}_{base_filename}"
        cv2.imwrite(os.path.join(output_dir, raw_rect_filename), rect_img)
        
        raw_bev_bgr = cv2.warpPerspective(rect_img, H_mat, (bev_w, bev_h))
        raw_bev_rgba = apply_bev_feathering(raw_bev_bgr)
        raw_bev_filename = f"raw_bev_{view_name}_{base_name_no_ext}.png" 
        cv2.imwrite(os.path.join(output_dir, raw_bev_filename), raw_bev_rgba)

        # 3. Geo Footprint tracking (Generating 4 exact corners for MapLibre Stitching)
        view_heading = (heading + config['heading_offset']) % 360
        bev_center_lat, bev_center_lon = local_to_global(gps_lat, gps_lon, view_heading, 0, (z_near + z_far) / 2.0)
        
        def to_lng_lat(x, z):
            lat_out, lon_out = local_to_global(gps_lat, gps_lon, view_heading, x, z)
            return [lon_out, lat_out] # MapLibre format [lng, lat]
            
        maplibre_corners = [
            to_lng_lat(-x_range, z_far), # Top-Left
            to_lng_lat(x_range, z_far),  # Top-Right
            to_lng_lat(x_range, z_near), # Bottom-Right
            to_lng_lat(-x_range, z_near) # Bottom-Left
        ]

        bev_footprints[view_name] = {
            "lat": bev_center_lat, "lon": bev_center_lon, 
            "heading": view_heading, 
            "width_m": 2 * x_range, "height_m": z_far - z_near,
            "corners": maplibre_corners
        }

        annotated_rect = rect_img.copy()
        annotated_bev_bgr = raw_bev_bgr.copy()

        # 4. Draw the clear ROI Trapezoid on annotated_rect
        pitch_rad, roll_rad = math.radians(-bev_pitch), math.radians(bev_roll)
        Rx = np.array([[1, 0, 0], [0, math.cos(pitch_rad), -math.sin(pitch_rad)], [0, math.sin(pitch_rad), math.cos(pitch_rad)]])
        Rz = np.array([[math.cos(roll_rad), -math.sin(roll_rad), 0], [math.sin(roll_rad), math.cos(roll_rad), 0], [0, 0, 1]])
        R = Rx @ Rz
        
        roi_pts_3d = [[-x_range, cam_height, z_near], [x_range, cam_height, z_near], [x_range, cam_height, z_far], [-x_range, cam_height, z_far]]
        roi_pts_2d = []
        for pt in roi_pts_3d:
            xyz = R @ np.array(pt)
            if xyz[2] > 1e-5:
                u = int((K[0,0] * xyz[0] / xyz[2]) + K[0,2])
                v = int((K[1,1] * xyz[1] / xyz[2]) + K[1,2])
                roi_pts_2d.append([u, v])
                
        if len(roi_pts_2d) == 4:
            overlay = annotated_rect.copy()
            pts_array = np.array(roi_pts_2d, np.int32).reshape((-1, 1, 2))
            cv2.fillPoly(overlay, [pts_array], (0, 255, 255))
            cv2.addWeighted(overlay, 0.15, annotated_rect, 0.85, 0, annotated_rect)
            cv2.polylines(annotated_rect, [pts_array], isClosed=True, color=(0, 255, 255), thickness=2)

        if draw_grid:
            annotated_rect = draw_bev_grid(annotated_rect, K, cam_height, bev_pitch, bev_roll, z_near, z_far, x_range)

        # 5. Model Inference (Applying Ego Mask first!)
        inference_img = apply_ego_mask(rect_img.copy(), mask_pct=0.15)
        with model_lock: results = model.predict(source=inference_img, conf=0.25, save=False, verbose=False)

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
                            
                        overlay = annotated_bev_bgr.copy()
                        cv2.fillPoly(overlay, [contour], color=mask_color_bgr)
                        cv2.addWeighted(overlay, 0.4, annotated_bev_bgr, 0.6, 0, annotated_bev_bgr)
                        
                        geo_coords = [[local_to_global(gps_lat, gps_lon, view_heading, (pt[0][0]*gsd)-x_range, z_far-(pt[0][1]*gsd))[1], local_to_global(gps_lat, gps_lon, view_heading, (pt[0][0]*gsd)-x_range, z_far-(pt[0][1]*gsd))[0]] for pt in contour]
                        if geo_coords[0] != geo_coords[-1]: geo_coords.append(geo_coords[0])

                        all_defects[view_name].append({"class": class_name, "conf": round(conf, 2), "area_sqm": round(area_sqm, 4), "color": hex_color})
                        all_geojson_features.append({"type": "Feature", "properties": {"class": class_name, "area_sqm": round(area_sqm, 4), "view": view_name, "color": hex_color, "filename": original_filename, "conf": round(conf, 2)}, "geometry": {"type": "Polygon", "coordinates": [geo_coords]}})

        annotated_bev_rgba = apply_bev_feathering(annotated_bev_bgr)
        rect_filename = f"rect_{view_name}_{base_filename}"
        bev_filename = f"bev_{view_name}_{base_name_no_ext}.png"
        
        cv2.imwrite(os.path.join(output_dir, rect_filename), annotated_rect)
        cv2.imwrite(os.path.join(output_dir, bev_filename), annotated_bev_rgba)
        
        generated_files[view_name] = {
            "raw_rect": raw_rect_filename,
            "raw_bev": raw_bev_filename,
            "rect": rect_filename,
            "bev": bev_filename
        }
        
    return all_defects, all_geojson_features, generated_files, bev_footprints

def process_video_frames_async(video_path, model, upload_dir, cam_height, file_name, original_name, gps_snap, interval_m, model_lock, is_360, location_str, callback, draw_grid=False):
    cap = cv2.VideoCapture(video_path)
    base_stem = os.path.splitext(file_name)[0]
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    
    try:
        from extract_gpmf import extract_streams_with_time, get_telemetry_interpolators, evaluate_telemetry_health
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
        fov_from_meta = constants.get('MFOV', None)
        if fov_from_meta is None:
            zfov = constants.get('ZFOV')
            aruw = constants.get('ARUW')
            if zfov is not None and aruw is not None:
                try:
                    zfov_rad = math.radians(float(zfov))
                    aruw_val = float(aruw)
                    hfov_rad = 2.0 * math.atan(math.tan(zfov_rad / 2.0) * (aruw_val / math.sqrt(aruw_val**2 + 1)))
                    fov_from_meta = math.degrees(hfov_rad)
                except Exception: pass
                
    except Exception as e:
        callback({"error": f"Failed to parse GPMF for video: {str(e)}", "is_video": True, "original_name": original_name})
        cap.release()
        return

    # FOR FFmpeg STITCHED 360: FFmpeg strips the global metadata block. 
    # We fallback to a target 100-degree rectilinear FOV extraction.
    if is_360 and fov_from_meta is None:
        fov_from_meta = 100.0

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
            
            # Using actual pitch/roll for 360 allows the math engine to Horizon-Level raw stitched FFmpeg video!
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
                    "Timestamp_sec": elapsed_sec,
                    "Latitude": current_lat,
                    "Longitude": current_lon,
                    "Heading": current_heading,
                    "Pitch_Inst": current_pitch,
                    "Pitch_Base": current_base_pitch,
                    "Roll_Inst": current_roll,
                    "Roll_Base": current_base_roll,
                    "Speed_ms": float(speed_interp(elapsed_sec)) if speed_interp else None,
                    "FOV": fov_from_meta,
                    "KLNS": klns
                })
            }
            with open(os.path.join(upload_dir, f"meta_{frame_base_name}.json"), 'w') as mf:
                json.dump(frame_meta, mf, indent=2)

            try:
                defects, geo_feats, gen_files, footprints = process_single_image(
                    frame, model, frame_base_name, upload_dir,
                    current_lat, current_lon, current_heading, cam_height, 
                    current_pitch, current_roll, current_base_pitch, current_base_roll, 
                    klns, fov_from_meta, model_lock, is_360, original_frame_name, draw_grid
                )
            except Exception as e:
                callback({"error": str(e), "is_video": False, "original_name": original_frame_name})
                continue
            
            result_payload = {"original_name": original_frame_name, "filename": frame_base_name, "lat": round(current_lat, 6), "lon": round(current_lon, 6), "pitch": round(current_pitch, 2), "roll": round(current_roll, 2), "location": location_str, "geojson": geo_feats, "views": {}}
            for view in (['front', 'rear'] if is_360 else ['front']):
                gf = gen_files[view]
                result_payload["views"][view] = {
                    "raw_filename": gf["raw_rect"], 
                    "raw_bev_filename": gf["raw_bev"], 
                    "raw_bev_url": f"/static/uploads/{gf['raw_bev']}", 
                    "rect_url": f"/static/uploads/{gf['rect']}", 
                    "bev_url": f"/static/uploads/{gf['bev']}", 
                    "defects": defects[view], 
                    "footprint": footprints[view]
                }
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