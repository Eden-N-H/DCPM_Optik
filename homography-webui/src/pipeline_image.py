import os
import cv2
import math
import numpy as np
from ultralytics.utils.plotting import colors

from geo_math import local_to_global
from cv_projections import equirectangular_to_rectilinear, digital_gimbal_warp
from cv_bev import get_bev_homography, apply_bev_feathering, draw_bev_grid, apply_ego_mask

def process_single_image(img_input, model, base_filename, output_dir, telemetry, options, model_lock, original_filename=""):
    gps_lat = telemetry.get('lat')
    gps_lon = telemetry.get('lon')
    heading = telemetry.get('heading', 0.0)
    fov_val = float(telemetry.get('fov')) if telemetry.get('fov') is not None else None
    
    if fov_val is None or gps_lat is None or gps_lon is None:
        raise ValueError("Missing critical metadata (GPS or FOV).")
        
    cam_height = options.get('cam_height', 1.6)
    is_360 = options.get('is_360', True)
    draw_grid = options.get('draw_grid', False)
    
    do_pitch = options.get('comp_pitch', True)
    do_roll = options.get('comp_roll', True)
    
    pitch = telemetry.get('pitch', 0.0) if do_pitch and telemetry.get('pitch') is not None else 0.0
    roll = telemetry.get('roll', 0.0) if do_roll and telemetry.get('roll') is not None else 0.0
    base_pitch = telemetry.get('base_pitch', 0.0) if do_pitch and telemetry.get('base_pitch') is not None else 0.0
    base_roll = telemetry.get('base_roll', 0.0) if do_roll and telemetry.get('base_roll') is not None else 0.0

    img_mat = cv2.imread(img_input) if isinstance(img_input, str) else img_input
    base_name_no_ext = os.path.splitext(base_filename)[0]
    
    all_defects, all_geojson_features, generated_files, bev_footprints, view_meta_all = {}, [], {}, {}, {}
    views_to_process = {'front': {'yaw': 0, 'heading_offset': 0}}
    if is_360: views_to_process['rear'] = {'yaw': 180, 'heading_offset': 180}
        
    for view_name, config in views_to_process.items():
        all_defects[view_name] = []
        if is_360:
            rect_img, K = equirectangular_to_rectilinear(img_mat, fov_deg=fov_val, pitch_deg=pitch, roll_deg=roll, yaw_deg=config['yaw'])
            bev_pitch, bev_roll = 0.0, 0.0
        else:
            rect_img = img_mat.copy()
            h, w = rect_img.shape[:2]
            f = (w / 2.0) / math.tan(math.radians(fov_val) / 2.0)
            K = np.array([[f, 0, w / 2.0], [0, f, h / 2.0], [0, 0, 1]], dtype=np.float32)
            
            if telemetry.get('klns') and len(telemetry['klns']) >= 5 and options.get('undistort', True):
                try:
                    dist_coeffs = np.array(telemetry['klns'][1:6], dtype=np.float32)
                    K_undist = np.array([[telemetry['klns'][0], 0, w/2], [0, telemetry['klns'][0], h/2], [0, 0, 1]], dtype=np.float32)
                    rect_img = cv2.undistort(rect_img, K_undist, dist_coeffs)
                    K = K_undist
                except: pass
                
            rect_img = digital_gimbal_warp(rect_img, K, pitch - base_pitch, roll - base_roll)
            bev_pitch, bev_roll = base_pitch, base_roll

        view_meta = {"K": K.tolist(), "detections": []}

        H_mat, bev_w, bev_h, gsd, x_range, z_far, z_near = get_bev_homography(K, cam_height, bev_pitch, bev_roll)
        raw_rect_filename = f"raw_rect_{view_name}_{base_filename}"
        cv2.imwrite(os.path.join(output_dir, raw_rect_filename), rect_img)
        raw_bev_bgr = cv2.warpPerspective(rect_img, H_mat, (bev_w, bev_h))
        raw_bev_rgba = apply_bev_feathering(raw_bev_bgr)
        raw_bev_filename = f"raw_bev_{view_name}_{base_name_no_ext}.png" 
        cv2.imwrite(os.path.join(output_dir, raw_bev_filename), raw_bev_rgba)

        view_heading = (heading + config['heading_offset']) % 360
        bev_center_lat, bev_center_lon = local_to_global(gps_lat, gps_lon, view_heading, 0, (z_near + z_far) / 2.0)
        def to_lng_lat(x, z):
            lat_out, lon_out = local_to_global(gps_lat, gps_lon, view_heading, x, z)
            return [lon_out, lat_out]
            
        maplibre_corners = [to_lng_lat(-x_range, z_far), to_lng_lat(x_range, z_far), to_lng_lat(x_range, z_near), to_lng_lat(-x_range, z_near)]
        bev_footprints[view_name] = {"lat": bev_center_lat, "lon": bev_center_lon, "heading": view_heading, "width_m": 2 * x_range, "height_m": z_far - z_near, "corners": maplibre_corners}

        annotated_rect = rect_img.copy()
        annotated_bev_bgr = raw_bev_bgr.copy()

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

        conf_thresh = options.get('conf_thresh', 0.25)
        inference_img = apply_ego_mask(rect_img.copy(), mask_pct=0.15) if options.get('ego_mask', True) else rect_img.copy()
        
        with model_lock: results = model.predict(source=inference_img, conf=conf_thresh, save=False, verbose=False)

        for r in results:
            annotated_rect = r.plot(img=annotated_rect)
            if r.masks is not None:
                for i, mask_pts in enumerate(r.masks.xy):
                    cls_id, conf = int(r.boxes.cls[i]), float(r.boxes.conf[i])
                    class_name = model.names[cls_id]
                    mask_color_bgr = colors(cls_id, bgr=True)
                    hex_color = f"#{int(mask_color_bgr[2]):02x}{int(mask_color_bgr[1]):02x}{int(mask_color_bgr[0]):02x}"
                    
                    view_meta["detections"].append({
                        "class_name": class_name,
                        "conf": conf,
                        "color_bgr": mask_color_bgr,
                        "hex_color": hex_color,
                        "polygon": mask_pts.tolist() 
                    })

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
        
        generated_files[view_name] = {"raw_rect": raw_rect_filename, "raw_bev": raw_bev_filename, "rect": rect_filename, "bev": bev_filename}
        view_meta_all[view_name] = view_meta
        
    return all_defects, all_geojson_features, generated_files, bev_footprints, view_meta_all

def generate_grid_preview(raw_rect_path, process_meta, view_name, pitch_offset):
    raw_rect = cv2.imread(raw_rect_path)
    K = np.array(process_meta['view_meta'][view_name]['K'], dtype=np.float32)
    telemetry = process_meta['telemetry']
    options = process_meta['options']
    
    is_360 = options.get('is_360', True)
    cam_height = options.get('cam_height', 1.6)
    
    base_pitch = telemetry.get('base_pitch', 0.0) if options.get('comp_pitch', True) else 0.0
    base_roll = telemetry.get('base_roll', 0.0) if options.get('comp_roll', True) else 0.0
    
    bev_pitch = (0.0 if is_360 else base_pitch) + pitch_offset
    bev_roll = 0.0 if is_360 else base_roll
    
    H_mat, bev_w, bev_h, gsd, x_range, z_far, z_near = get_bev_homography(K, cam_height, bev_pitch, bev_roll)
    
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
            
    preview_img = raw_rect.copy()
    if len(roi_pts_2d) == 4:
        overlay = preview_img.copy()
        pts_array = np.array(roi_pts_2d, np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(overlay, [pts_array], (0, 255, 255))
        cv2.addWeighted(overlay, 0.15, preview_img, 0.85, 0, preview_img)
        cv2.polylines(preview_img, [pts_array], isClosed=True, color=(0, 255, 255), thickness=2)
        
    if options.get('draw_grid', False):
        preview_img = draw_bev_grid(preview_img, K, cam_height, bev_pitch, bev_roll, z_near, z_far, x_range)
        
    return preview_img

def recalculate_view(raw_rect_path, view_meta, telemetry, options, view_name, pitch_offset, original_filename, output_dir, base_filename):
    raw_rect = cv2.imread(raw_rect_path)
    K = np.array(view_meta['K'], dtype=np.float32)
    
    gps_lat = telemetry.get('lat')
    gps_lon = telemetry.get('lon')
    heading = telemetry.get('heading', 0.0)
    
    is_360 = options.get('is_360', True)
    cam_height = options.get('cam_height', 1.6)
    
    base_pitch = telemetry.get('base_pitch', 0.0) if options.get('comp_pitch', True) else 0.0
    base_roll = telemetry.get('base_roll', 0.0) if options.get('comp_roll', True) else 0.0
    
    bev_pitch = (0.0 if is_360 else base_pitch) + pitch_offset
    bev_roll = 0.0 if is_360 else base_roll
    
    H_mat, bev_w, bev_h, gsd, x_range, z_far, z_near = get_bev_homography(K, cam_height, bev_pitch, bev_roll)
    
    raw_bev_bgr = cv2.warpPerspective(raw_rect, H_mat, (bev_w, bev_h))
    raw_bev_rgba = apply_bev_feathering(raw_bev_bgr)
    base_name_no_ext = os.path.splitext(base_filename)[0]
    raw_bev_filename = f"raw_bev_{view_name}_{base_name_no_ext}.png"
    cv2.imwrite(os.path.join(output_dir, raw_bev_filename), raw_bev_rgba)
    
    heading_offset = 180 if view_name == 'rear' else 0
    view_heading = (heading + heading_offset) % 360
    bev_center_lat, bev_center_lon = local_to_global(gps_lat, gps_lon, view_heading, 0, (z_near + z_far) / 2.0)
    def to_lng_lat(x, z):
        lat_out, lon_out = local_to_global(gps_lat, gps_lon, view_heading, x, z)
        return [lon_out, lat_out]
        
    maplibre_corners = [to_lng_lat(-x_range, z_far), to_lng_lat(x_range, z_far), to_lng_lat(x_range, z_near), to_lng_lat(-x_range, z_near)]
    footprint = {"lat": bev_center_lat, "lon": bev_center_lon, "heading": view_heading, "width_m": 2 * x_range, "height_m": z_far - z_near, "corners": maplibre_corners}
    
    annotated_rect = raw_rect.copy()
    annotated_bev_bgr = raw_bev_bgr.copy()
    
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
        
    if options.get('draw_grid', False):
        annotated_rect = draw_bev_grid(annotated_rect, K, cam_height, bev_pitch, bev_roll, z_near, z_far, x_range)
        
    defects, geojson_features = [], []
    
    if options.get('ego_mask', True):
        annotated_rect = apply_ego_mask(annotated_rect, mask_pct=0.15)
        
    for det in view_meta.get("detections", []):
        mask_pts = np.array(det["polygon"], dtype=np.int32)
        class_name = det["class_name"]
        conf = det["conf"]
        mask_color_bgr = tuple(det["color_bgr"])
        hex_color = det["hex_color"]
        
        mask_canvas = np.zeros(raw_rect.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask_canvas, [mask_pts], 255)
        
        rect_overlay = annotated_rect.copy()
        cv2.fillPoly(rect_overlay, [mask_pts], mask_color_bgr)
        cv2.addWeighted(rect_overlay, 0.4, annotated_rect, 0.6, 0, annotated_rect)
        
        contours, _ = cv2.findContours(cv2.warpPerspective(mask_canvas, H_mat, (bev_w, bev_h)), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area_sqm = cv2.contourArea(contour) * (gsd ** 2)
            if area_sqm <= 0.0001: continue
            
            overlay = annotated_bev_bgr.copy()
            cv2.fillPoly(overlay, [contour], color=mask_color_bgr)
            cv2.addWeighted(overlay, 0.4, annotated_bev_bgr, 0.6, 0, annotated_bev_bgr)
            
            geo_coords = [[local_to_global(gps_lat, gps_lon, view_heading, (pt[0][0]*gsd)-x_range, z_far-(pt[0][1]*gsd))[1], local_to_global(gps_lat, gps_lon, view_heading, (pt[0][0]*gsd)-x_range, z_far-(pt[0][1]*gsd))[0]] for pt in contour]
            if geo_coords[0] != geo_coords[-1]: geo_coords.append(geo_coords[0])
            
            defects.append({"class": class_name, "conf": round(conf, 2), "area_sqm": round(area_sqm, 4), "color": hex_color})
            geojson_features.append({"type": "Feature", "properties": {"class": class_name, "area_sqm": round(area_sqm, 4), "view": view_name, "color": hex_color, "filename": original_filename, "conf": round(conf, 2)}, "geometry": {"type": "Polygon", "coordinates": [geo_coords]}})

    annotated_bev_rgba = apply_bev_feathering(annotated_bev_bgr)
    rect_filename = f"rect_{view_name}_{base_filename}"
    bev_filename = f"bev_{view_name}_{base_name_no_ext}.png"
    cv2.imwrite(os.path.join(output_dir, rect_filename), annotated_rect)
    cv2.imwrite(os.path.join(output_dir, bev_filename), annotated_bev_rgba)
    
    return defects, geojson_features, footprint