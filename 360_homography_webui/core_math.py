import cv2
import numpy as np
from ultralytics import YOLO
from ultralytics.utils.plotting import colors 
import math
import struct
import exifread
import os

def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculates the distance in meters between two coordinates."""
    R = 6371000.0  # Earth radius in meters
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2)**2
    return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1 - a))

def extract_gpmf_pitch(filepath, fallback_pitch=-15.0):
    try:
        with open(filepath, 'rb') as f:
            data = f.read()
        idx = data.find(b'GRAV')
        if idx != -1:
            payload = data[idx+8 : idx+8+12]
            if len(payload) == 12:
                x, y, z = struct.unpack('>fff', payload)
                pitch = math.degrees(math.atan2(z, y))
                return -abs(pitch)
    except Exception as e:
        print(f"GPMF Extraction failed: {e}")
    return fallback_pitch

def get_exif_gps(filepath):
    try:
        with open(filepath, 'rb') as f:
            tags = exifread.process_file(f, details=False)
        if 'GPS GPSLatitude' in tags and 'GPS GPSLongitude' in tags:
            def convert_to_degrees(value):
                d, m, s = value.values
                return d.num/d.den + (m.num/m.den)/60.0 + (s.num/s.den)/3600.0

            lat = convert_to_degrees(tags['GPS GPSLatitude'])
            lat_ref = tags.get('GPS GPSLatitudeRef', None)
            if lat_ref and lat_ref.printable != 'N': lat = -lat

            lon = convert_to_degrees(tags['GPS GPSLongitude'])
            lon_ref = tags.get('GPS GPSLongitudeRef', None)
            if lon_ref and lon_ref.printable != 'E': lon = -lon
            
            return lat, lon
    except Exception:
        pass
    return 0.0, 0.0

def calculate_bearing(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1
    x = math.sin(dlon) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - (math.sin(lat1) * math.cos(lat2) * math.cos(dlon))
    return (math.degrees(math.atan2(x, y)) + 360) % 360

def local_to_global(lat, lon, heading_deg, local_x, local_z):
    R = 6378137.0
    d = math.hypot(local_x, local_z)
    angle_rad = math.atan2(local_x, local_z)
    true_heading_rad = math.radians(heading_deg) + angle_rad

    lat_rad, lon_rad = math.radians(lat), math.radians(lon)
    out_lat = math.asin(math.sin(lat_rad)*math.cos(d/R) + math.cos(lat_rad)*math.sin(d/R)*math.cos(true_heading_rad))
    out_lon = lon_rad + math.atan2(math.sin(true_heading_rad)*math.sin(d/R)*math.cos(lat_rad), math.cos(d/R) - math.sin(lat_rad)*math.sin(out_lat))
    return math.degrees(out_lat), math.degrees(out_lon)

def equirectangular_to_rectilinear(equi_img, fov_deg=100, pitch_deg=-15, yaw_deg=0, output_width=1280, output_height=720):
    h, w = equi_img.shape[:2]
    f = (output_width / 2.0) / math.tan(math.radians(fov_deg) / 2.0)
    K = np.array([[f, 0, output_width / 2.0], [0, f, output_height / 2.0], [0, 0, 1]], dtype=np.float32)
    K_inv = np.linalg.inv(K)

    pitch = math.radians(pitch_deg)
    yaw = math.radians(yaw_deg)
    
    R_pitch = np.array([[1, 0, 0], [0, math.cos(pitch), -math.sin(pitch)], [0, math.sin(pitch), math.cos(pitch)]])
    R_yaw = np.array([[math.cos(yaw), 0, math.sin(yaw)], [0, 1, 0], [-math.sin(yaw), 0, math.cos(yaw)]])
    
    R_combined = R_yaw @ R_pitch 
    R_inv = np.linalg.inv(R_combined)

    x, y = np.meshgrid(np.arange(output_width), np.arange(output_height))
    pixels = np.stack((x, y, np.ones_like(x)), axis=-1).reshape(-1, 3).T
    
    rays = K_inv @ pixels
    rays = R_inv @ rays
    
    theta = np.arctan2(rays[0, :], rays[2, :])
    phi = np.arcsin(np.clip(rays[1, :] / np.linalg.norm(rays, axis=0), -1, 1))
    
    u = (theta / (2 * math.pi) + 0.5) * w
    v = (phi / math.pi + 0.5) * h
    
    map_x = u.reshape((output_height, output_width)).astype(np.float32)
    map_y = v.reshape((output_height, output_width)).astype(np.float32)
    
    return cv2.remap(equi_img, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP), K

def get_bev_homography(K, cam_height_m, pitch_deg, gsd=0.01, z_near=2.0, z_far=8.0, x_range=3.0):
    pitch_rad = math.radians(pitch_deg)
    road_pts = np.array([[-x_range, z_near], [x_range, z_near], [x_range, z_far], [-x_range, z_far]], dtype=np.float32)
    
    bev_w = int((2 * x_range) / gsd)
    bev_h = int((z_far - z_near) / gsd)
    bev_pts = np.array([[0, bev_h], [bev_w, bev_h], [bev_w, 0], [0, 0]], dtype=np.float32)

    rect_pts = []
    for pt in road_pts:
        X, Z = pt
        Y = cam_height_m 
        
        Y_rot = Y * math.cos(pitch_rad) - Z * math.sin(pitch_rad)
        Z_rot = Y * math.sin(pitch_rad) + Z * math.cos(pitch_rad)
        
        u = (K[0,0] * X / Z_rot) + K[0,2]
        v = (K[1,1] * Y_rot / Z_rot) + K[1,2]
        rect_pts.append([u, v])

    H_mat = cv2.getPerspectiveTransform(np.array(rect_pts, dtype=np.float32), bev_pts)
    return H_mat, bev_w, bev_h, gsd, x_range, z_far

def process_single_image(equi_img_path, model, base_filename, output_dir, gps_lat, gps_lon, heading, cam_height, pitch, model_lock):
    equi_img = cv2.imread(equi_img_path)
    
    all_defects = {'front': [], 'rear': []}
    all_geojson_features = []
    
    views = {
        'front': {'yaw': 0, 'heading_offset': 0},
        'rear': {'yaw': 180, 'heading_offset': 180}
    }
    
    for view_name, config in views.items():
        rect_img, K = equirectangular_to_rectilinear(equi_img, pitch_deg=pitch, yaw_deg=config['yaw'])
        
        raw_rect_filename = f"raw_rect_{view_name}_{base_filename}"
        cv2.imwrite(os.path.join(output_dir, raw_rect_filename), rect_img)
        
        with model_lock:
            results = model.predict(source=rect_img, conf=0.25, save=False, verbose=False)
            
        annotated_rect = rect_img.copy()
        H_mat, bev_w, bev_h, gsd, x_range, z_far = get_bev_homography(K, cam_height, pitch)
        
        bev_img = cv2.warpPerspective(rect_img, H_mat, (bev_w, bev_h))
        
        # --- NEW: Save the raw, unannotated flattened square ---
        raw_bev_filename = f"raw_bev_{view_name}_{base_filename}"
        cv2.imwrite(os.path.join(output_dir, raw_bev_filename), bev_img)
        # -------------------------------------------------------
        
        bev_overlay = bev_img.copy() 
        rect_h, rect_w = rect_img.shape[:2]
        
        view_heading = (heading + config['heading_offset']) % 360

        for r in results:
            annotated_rect = r.plot()
            if r.masks is not None:
                for i, mask_pts in enumerate(r.masks.xy):
                    cls_id = int(r.boxes.cls[i])
                    class_name = model.names[cls_id]
                    conf = float(r.boxes.conf[i])
                    
                    mask_color_bgr = colors(cls_id, bgr=True)
                    # Convert YOLO's BGR color to HEX for the Map UI & HTML Table
                    b, g, r_color = mask_color_bgr
                    hex_color = f"#{int(r_color):02x}{int(g):02x}{int(b):02x}"
                    
                    mask_canvas = np.zeros((rect_h, rect_w), dtype=np.uint8)
                    int_mask_pts = np.array(mask_pts, dtype=np.int32)
                    cv2.fillPoly(mask_canvas, [int_mask_pts], 255)
                    
                    bev_mask_canvas = cv2.warpPerspective(mask_canvas, H_mat, (bev_w, bev_h))
                    contours, _ = cv2.findContours(bev_mask_canvas, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    
                    for contour in contours:
                        area_sqm = cv2.contourArea(contour) * (gsd ** 2)
                        if area_sqm <= 0.0001: 
                            continue
                            
                        cv2.fillPoly(bev_overlay, [contour], color=mask_color_bgr)
                        
                        geo_coords = []
                        for pt in contour:
                            px, py = pt[0][0], pt[0][1]
                            local_x = (px * gsd) - x_range
                            local_z = z_far - (py * gsd)
                            g_lat, g_lon = local_to_global(gps_lat, gps_lon, view_heading, local_x, local_z)
                            geo_coords.append([g_lon, g_lat])
                        
                        if len(geo_coords) > 0 and geo_coords[0] != geo_coords[-1]:
                            geo_coords.append(geo_coords[0])

                        # Pass the exact hex color to the HTML table
                        all_defects[view_name].append({
                            "class": class_name, 
                            "conf": round(conf, 2), 
                            "area_sqm": round(area_sqm, 4),
                            "color": hex_color
                        })
                        
                        # Pass the exact hex color to the Map
                        all_geojson_features.append({
                            "type": "Feature",
                            "properties": {"class": class_name, "area_sqm": round(area_sqm, 4), "view": view_name, "color": hex_color},
                            "geometry": {"type": "Polygon", "coordinates": [geo_coords]}
                        })

        cv2.addWeighted(bev_overlay, 0.5, bev_img, 0.5, 0, bev_img)
        
        annotated_rect_filename = f"rect_{view_name}_{base_filename}"
        bev_filename = f"bev_{view_name}_{base_filename}"
        
        cv2.imwrite(os.path.join(output_dir, annotated_rect_filename), annotated_rect)
        cv2.imwrite(os.path.join(output_dir, bev_filename), bev_img)
        
    return all_defects, all_geojson_features, base_filename