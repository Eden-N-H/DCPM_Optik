import os
import cv2
import math
import numpy as np
from ultralytics.utils.plotting import colors

from geo_math import local_to_global
from cv_projections import equirectangular_to_rectilinear
from cv_bev import get_bev_homography, apply_bev_feathering, draw_bev_grid, apply_ego_mask, get_camera_rotation_matrix

def _run_sam2_masks(rect_img, results, sam2_predictor):
    """Run SAM2 on YOLO detections to get pixel-precise masks.
    Returns list of (mask_pts, cls_id, conf, class_name, mask_color_bgr) tuples."""
    if sam2_predictor is None:
        return None
    from sam2_integration import run_sam2_on_detections
    all_sam2_results = []
    for r in results:
        if r.boxes is not None and len(r.boxes) > 0:
            image_rgb = cv2.cvtColor(rect_img, cv2.COLOR_BGR2RGB)
            sam2_out = run_sam2_on_detections(image_rgb, r, sam2_predictor)
            for pts, cls_id, conf, class_name in sam2_out:
                mask_color_bgr = colors(cls_id, bgr=True)
                all_sam2_results.append((pts, cls_id, conf, class_name, mask_color_bgr))
    return all_sam2_results if all_sam2_results else None


def _annotate_with_sam2(annotated_rect, sam2_results, model_names):
    """Draw SAM2 masks, bounding boxes, and labels on the annotated image."""
    for pts, cls_id, conf, class_name, mask_color_bgr in sam2_results:
        # Draw filled mask
        pts_int = pts.astype(np.int32)
        overlay = annotated_rect.copy()
        cv2.fillPoly(overlay, [pts_int], color=mask_color_bgr)
        cv2.addWeighted(overlay, 0.4, annotated_rect, 0.6, 0, annotated_rect)
        # Draw contour outline
        cv2.polylines(annotated_rect, [pts_int], isClosed=True, color=mask_color_bgr, thickness=2)
        # Draw label
        x_min, y_min = pts_int.min(axis=0)
        label = f"{class_name} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(annotated_rect, (x_min, y_min - th - 4), (x_min + tw, y_min), mask_color_bgr, -1)
        cv2.putText(annotated_rect, label, (x_min, y_min - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return annotated_rect


def _process_simple_frame(img_input, model, base_filename, output_dir, options, model_lock, original_filename="", sam2_predictor=None):
    """Process a standard image without GPS — just YOLO detection + SAM2 masks."""
    img_mat = cv2.imread(img_input) if isinstance(img_input, str) else img_input
    base_name_no_ext = os.path.splitext(base_filename)[0]
    
    source_filename = f"source_{base_filename}"
    if not os.path.exists(os.path.join(output_dir, source_filename)):
        cv2.imwrite(os.path.join(output_dir, source_filename), img_mat)
    
    conf_thresh = options.get('conf_thresh', 0.25)
    with model_lock:
        results = model.predict(source=img_mat, conf=conf_thresh, save=False, verbose=False)
    
    annotated = img_mat.copy()
    all_defects = []
    
    # Try SAM2 first, fall back to native YOLO
    sam2_results = _run_sam2_masks(img_mat, results, sam2_predictor)
    if sam2_results:
        _annotate_with_sam2(annotated, sam2_results, model.names)
        for pts, cls_id, conf, class_name, color in sam2_results:
            hex_color = f"#{int(color[2]):02x}{int(color[1]):02x}{int(color[0]):02x}"
            area_px = cv2.contourArea(np.array(pts, dtype=np.int32))
            all_defects.append({"class": class_name, "conf": round(conf, 2), "area_sqm": round(area_px, 0), "color": hex_color})
    else:
        for r in results:
            annotated = r.plot(img=annotated)
            if r.boxes is not None:
                for i in range(len(r.boxes)):
                    cls_id = int(r.boxes.cls[i])
                    conf = float(r.boxes.conf[i])
                    class_name = model.names[cls_id]
                    all_defects.append({"class": class_name, "conf": round(conf, 2), "area_sqm": 0, "color": "#ff0000"})
    
    rect_filename = f"rect_front_{base_filename}"
    cv2.imwrite(os.path.join(output_dir, rect_filename), annotated)
    
    generated_files = {"front": {"raw_rect": source_filename, "raw_bev": rect_filename, "rect": rect_filename, "bev": rect_filename}}
    calibrations = {"front": {"pitch_offset": 0, "roll_offset": 0, "yaw_offset": 0, "fov": 100, "cam_height": 1.6, "z_near": 1.5, "z_far": 10, "lane_width": 8}}
    bev_footprints = {"front": {"lat": 0, "lon": 0, "heading": 0, "width_m": 8, "height_m": 8, "corners": [[0,0],[0,0],[0,0],[0,0]]}}
    view_meta = {"front": {"K": [[1,0,0],[0,1,0],[0,0,1]], "detections": []}}
    
    return {"front": all_defects}, [], generated_files, bev_footprints, view_meta, calibrations


def process_single_image(img_input, model, base_filename, output_dir, telemetry, options, model_lock, original_filename="", sam2_predictor=None):
    gps_lat = telemetry.get('lat')
    gps_lon = telemetry.get('lon')
    heading = telemetry.get('heading', 0.0)
    fov_val = float(telemetry.get('fov')) if telemetry.get('fov') is not None else None
    
    if fov_val is None or gps_lat is None or gps_lon is None:
        media_type = options.get('media_type', '360-video')
        has_telemetry = options.get('has_telemetry', False)
        if media_type in ('standard-photos', 'orthographic') or not has_telemetry:
            # Simple frame processing — no BEV, just run detection + SAM2
            return _process_simple_frame(img_input, model, base_filename, output_dir, options, model_lock, original_filename, sam2_predictor)
        raise ValueError("Missing critical metadata (GPS or FOV). Uncheck 'GoPro/Telemetry' or select 'Standard Photos' for images without GPS.")
        
    cam_height = options.get('cam_height', 1.6)
    is_360 = options.get('is_360', True)
    draw_grid = options.get('draw_grid', False)
    
    z_near_base = max(1.5, cam_height * 1.2)
    z_far_base = min(12.0, z_near_base + 8.0)
    lane_width_base = 8.0 
    x_range_base = lane_width_base / 2.0
    
    do_pitch = options.get('comp_pitch', True)
    do_roll = options.get('comp_roll', True)
    
    pitch = telemetry.get('pitch', 0.0) if do_pitch and telemetry.get('pitch') is not None else 0.0
    roll = telemetry.get('roll', 0.0) if do_roll and telemetry.get('roll') is not None else 0.0
    base_pitch = telemetry.get('base_pitch', 0.0) if do_pitch and telemetry.get('base_pitch') is not None else 0.0
    base_roll = telemetry.get('base_roll', 0.0) if do_roll and telemetry.get('base_roll') is not None else 0.0

    img_mat = cv2.imread(img_input) if isinstance(img_input, str) else img_input
    base_name_no_ext = os.path.splitext(base_filename)[0]
    
    source_filename = f"source_{base_filename}"
    if not os.path.exists(os.path.join(output_dir, source_filename)):
        cv2.imwrite(os.path.join(output_dir, source_filename), img_mat)
    
    all_defects, all_geojson_features, generated_files, bev_footprints, view_meta_all, calibrations = {}, [], {}, {}, {}, {}
    views_to_process = {'front': {'yaw': 0, 'heading_offset': 0}}
    if is_360: views_to_process['rear'] = {'yaw': 180, 'heading_offset': 180}
        
    for view_name, config in views_to_process.items():
        all_defects[view_name] = []
        
        calibrations[view_name] = {
            "pitch_offset": 0.0,
            "roll_offset": 0.0,
            "yaw_offset": 0.0,
            "fov": fov_val,
            "cam_height": cam_height,
            "z_near": z_near_base,
            "z_far": z_far_base,
            "lane_width": lane_width_base
        }

        if is_360:
            rect_img, K = equirectangular_to_rectilinear(img_mat, fov_deg=fov_val, pitch_deg=pitch, roll_deg=roll, yaw_deg=config['yaw'])
            bev_pitch, bev_roll, bev_yaw = 0.0, 0.0, 0.0
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
                
            bev_pitch, bev_roll, bev_yaw = pitch, roll, config['yaw']

        view_meta = {"K": K.tolist(), "detections": []}

        H_mat, bev_w, bev_h, gsd = get_bev_homography(K, cam_height, bev_pitch, bev_roll, bev_yaw, z_near_base, z_far_base, x_range_base)
        
        raw_rect_filename = f"raw_rect_{view_name}_{base_filename}"
        cv2.imwrite(os.path.join(output_dir, raw_rect_filename), rect_img)
        raw_bev_bgr = cv2.warpPerspective(rect_img, H_mat, (bev_w, bev_h))
        raw_bev_rgba = apply_bev_feathering(raw_bev_bgr)
        raw_bev_filename = f"raw_bev_{view_name}_{base_name_no_ext}.png" 
        cv2.imwrite(os.path.join(output_dir, raw_bev_filename), raw_bev_rgba)

        view_heading = (heading + config['heading_offset']) % 360
        bev_center_lat, bev_center_lon = local_to_global(gps_lat, gps_lon, view_heading, 0, (z_near_base + z_far_base) / 2.0)
        
        def to_lng_lat(x, z):
            lat_out, lon_out = local_to_global(gps_lat, gps_lon, view_heading, x, z)
            return [lon_out, lat_out]
            
        maplibre_corners = [to_lng_lat(-x_range_base, z_far_base), to_lng_lat(x_range_base, z_far_base), to_lng_lat(x_range_base, z_near_base), to_lng_lat(-x_range_base, z_near_base)]
        bev_footprints[view_name] = {"lat": bev_center_lat, "lon": bev_center_lon, "heading": view_heading, "width_m": 2 * x_range_base, "height_m": z_far_base - z_near_base, "corners": maplibre_corners}

        annotated_rect = rect_img.copy()
        annotated_bev_bgr = raw_bev_bgr.copy()

        R = get_camera_rotation_matrix(bev_pitch, bev_yaw, bev_roll)
        
        roi_pts_3d = [[-x_range_base, cam_height, z_near_base], [x_range_base, cam_height, z_near_base], [x_range_base, cam_height, z_far_base], [-x_range_base, cam_height, z_far_base]]
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
            annotated_rect = draw_bev_grid(annotated_rect, K, cam_height, bev_pitch, bev_roll, bev_yaw, z_near_base, z_far_base, x_range_base)

        conf_thresh = options.get('conf_thresh', 0.25)
        inference_img = apply_ego_mask(rect_img.copy(), mask_pct=0.15) if options.get('ego_mask', True) else rect_img.copy()
        
        with model_lock:
            results = model.predict(source=inference_img, conf=conf_thresh, save=False, verbose=False)

        # Try SAM2 first for pixel-precise masks, fall back to native YOLO masks
        sam2_results = _run_sam2_masks(rect_img, results, sam2_predictor) if sam2_predictor else None

        if sam2_results:
            # Use SAM2 masks — draw them instead of r.plot()
            annotated_rect = _annotate_with_sam2(annotated_rect, sam2_results, model.names)
            
            for pts, cls_id, conf, class_name, mask_color_bgr in sam2_results:
                hex_color = f"#{int(mask_color_bgr[2]):02x}{int(mask_color_bgr[1]):02x}{int(mask_color_bgr[0]):02x}"
                
                view_meta["detections"].append({
                    "class_name": class_name,
                    "conf": conf,
                    "color_bgr": mask_color_bgr,
                    "hex_color": hex_color,
                    "polygon": pts.tolist()
                })

                mask_canvas = np.zeros(rect_img.shape[:2], dtype=np.uint8)
                cv2.fillPoly(mask_canvas, [pts.astype(np.int32)], 255)
                contours, _ = cv2.findContours(cv2.warpPerspective(mask_canvas, H_mat, (bev_w, bev_h)), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                for contour in contours:
                    area_sqm = cv2.contourArea(contour) * (gsd ** 2)
                    if area_sqm <= 0.0001: continue
                    
                    overlay = annotated_bev_bgr.copy()
                    cv2.fillPoly(overlay, [contour], color=mask_color_bgr)
                    cv2.addWeighted(overlay, 0.4, annotated_bev_bgr, 0.6, 0, annotated_bev_bgr)
                    
                    geo_coords = [[local_to_global(gps_lat, gps_lon, view_heading, (pt[0][0]*gsd)-x_range_base, z_far_base-(pt[0][1]*gsd))[1], local_to_global(gps_lat, gps_lon, view_heading, (pt[0][0]*gsd)-x_range_base, z_far_base-(pt[0][1]*gsd))[0]] for pt in contour]
                    if geo_coords[0] != geo_coords[-1]: geo_coords.append(geo_coords[0])
                    
                    all_defects[view_name].append({"class": class_name, "conf": round(conf, 2), "area_sqm": round(area_sqm, 4), "color": hex_color})
                    all_geojson_features.append({
                        "type": "Feature",
                        "properties": {"class": class_name, "area_sqm": round(area_sqm, 4), "view": view_name, "color": hex_color, "filename": original_filename, "conf": round(conf, 2)},
                        "geometry": {"type": "Polygon", "coordinates": [geo_coords]}
                    })
        else:
            # Fallback: use native YOLO masks (original behavior)
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
                            
                            geo_coords = [[local_to_global(gps_lat, gps_lon, view_heading, (pt[0][0]*gsd)-x_range_base, z_far_base-(pt[0][1]*gsd))[1], local_to_global(gps_lat, gps_lon, view_heading, (pt[0][0]*gsd)-x_range_base, z_far_base-(pt[0][1]*gsd))[0]] for pt in contour]
                            if geo_coords[0] != geo_coords[-1]: geo_coords.append(geo_coords[0])
                            
                            all_defects[view_name].append({"class": class_name, "conf": round(conf, 2), "area_sqm": round(area_sqm, 4), "color": hex_color})
                            all_geojson_features.append({
                                "type": "Feature",
                                "properties": {"class": class_name, "area_sqm": round(area_sqm, 4), "view": view_name, "color": hex_color, "filename": original_filename, "conf": round(conf, 2)},
                                "geometry": {"type": "Polygon", "coordinates": [geo_coords]}
                            })

        annotated_bev_rgba = apply_bev_feathering(annotated_bev_bgr)
        rect_filename = f"rect_{view_name}_{base_filename}"
        bev_filename = f"bev_{view_name}_{base_name_no_ext}.png"
        
        cv2.imwrite(os.path.join(output_dir, rect_filename), annotated_rect)
        cv2.imwrite(os.path.join(output_dir, bev_filename), annotated_bev_rgba)
        
        generated_files[view_name] = {"raw_rect": raw_rect_filename, "raw_bev": raw_bev_filename, "rect": rect_filename, "bev": bev_filename}
        view_meta_all[view_name] = view_meta
        
    return all_defects, all_geojson_features, generated_files, bev_footprints, view_meta_all, calibrations


def get_projected_image(source_path, telemetry, options, view_name, calib):
    img_mat = cv2.imread(source_path)
    is_360 = options.get('is_360', True)
    
    base_pitch = telemetry.get('base_pitch', 0.0) if options.get('comp_pitch', True) else 0.0
    base_roll = telemetry.get('base_roll', 0.0) if options.get('comp_roll', True) else 0.0
    base_yaw = 0 if view_name == 'front' else 180
    
    pitch = base_pitch + calib.get('pitch_offset', 0)
    roll = base_roll + calib.get('roll_offset', 0)
    yaw = base_yaw + calib.get('yaw_offset', 0)
    fov = calib.get('fov', 100)

    if is_360:
        rect_img, K = equirectangular_to_rectilinear(img_mat, fov_deg=fov, pitch_deg=pitch, roll_deg=roll, yaw_deg=yaw)
        bev_pitch, bev_roll, bev_yaw = 0.0, 0.0, 0.0
    else:
        rect_img = img_mat.copy()
        h, w = rect_img.shape[:2]
        f = (w / 2.0) / math.tan(math.radians(fov) / 2.0)
        K = np.array([[f, 0, w / 2.0], [0, f, h / 2.0], [0, 0, 1]], dtype=np.float32)
        
        if telemetry.get('klns') and options.get('undistort', True):
            try:
                K_undist = np.array([[telemetry['klns'][0], 0, w/2], [0, telemetry['klns'][0], h/2], [0, 0, 1]], dtype=np.float32)
                rect_img = cv2.undistort(rect_img, K_undist, np.array(telemetry['klns'][1:6], dtype=np.float32))
                K = K_undist
            except: pass
            
        bev_pitch, bev_roll, bev_yaw = pitch, roll, yaw
        
    return rect_img, K, bev_pitch, bev_roll, bev_yaw

def generate_grid_preview(source_path, process_meta, view_name, calib):
    rect_img, K, bev_pitch, bev_roll, bev_yaw = get_projected_image(source_path, process_meta['telemetry'], process_meta['options'], view_name, calib)
    
    cam_h = calib.get('cam_height', 1.6)
    z_n = calib.get('z_near', 1.5)
    z_f = calib.get('z_far', 10.0)
    
    lane_w = calib.get('lane_width', 8.0)
    x_r = lane_w / 2.0
    
    R = get_camera_rotation_matrix(bev_pitch, bev_yaw, bev_roll)
    
    roi_pts_3d = [[-x_r, cam_h, z_n], [x_r, cam_h, z_n], [x_r, cam_h, z_f], [-x_r, cam_h, z_f]]
    roi_pts_2d = []
    
    for pt in roi_pts_3d:
        xyz = R @ np.array(pt)
        if xyz[2] > 1e-5:
            u = int((K[0,0] * xyz[0] / xyz[2]) + K[0,2])
            v = int((K[1,1] * xyz[1] / xyz[2]) + K[1,2])
            roi_pts_2d.append([u, v])
            
    preview_img = rect_img.copy()
    if len(roi_pts_2d) == 4:
        pts_array = np.array(roi_pts_2d, np.int32).reshape((-1, 1, 2))
        overlay = preview_img.copy()
        cv2.fillPoly(overlay, [pts_array], (0, 255, 255))
        cv2.addWeighted(overlay, 0.15, preview_img, 0.85, 0, preview_img)
        cv2.polylines(preview_img, [pts_array], isClosed=True, color=(0, 255, 255), thickness=2)
        
    if process_meta['options'].get('draw_grid', False):
        preview_img = draw_bev_grid(preview_img, K, cam_h, bev_pitch, bev_roll, bev_yaw, z_n, z_f, x_r)
        
    return preview_img

def recalculate_view(source_path, telemetry, options, view_name, calib, original_filename, output_dir, base_filename, model, model_lock):
    rect_img, K, bev_pitch, bev_roll, bev_yaw = get_projected_image(source_path, telemetry, options, view_name, calib)
    
    gps_lat, gps_lon = telemetry.get('lat'), telemetry.get('lon')
    heading = telemetry.get('heading', 0.0)
    
    cam_h = calib.get('cam_height', 1.6)
    z_n = calib.get('z_near', 1.5)
    z_f = calib.get('z_far', 10.0)
    
    lane_w = calib.get('lane_width', 8.0)
    x_r = lane_w / 2.0

    H_mat, bev_w, bev_h, gsd = get_bev_homography(K, cam_h, bev_pitch, bev_roll, bev_yaw, z_n, z_f, x_r)
    
    raw_bev_bgr = cv2.warpPerspective(rect_img, H_mat, (bev_w, bev_h))
    base_name_no_ext = os.path.splitext(base_filename)[0]
    
    cv2.imwrite(os.path.join(output_dir, f"raw_rect_{view_name}_{base_filename}"), rect_img)
    cv2.imwrite(os.path.join(output_dir, f"raw_bev_{view_name}_{base_name_no_ext}.png"), apply_bev_feathering(raw_bev_bgr))
    
    heading_offset = 180 if view_name == 'rear' else 0
    view_heading = (heading + heading_offset) % 360
    bev_center_lat, bev_center_lon = local_to_global(gps_lat, gps_lon, view_heading, 0, (z_n + z_f) / 2.0)
    
    def to_lng_lat(x, z):
        lat_out, lon_out = local_to_global(gps_lat, gps_lon, view_heading, x, z)
        return [lon_out, lat_out]
        
    maplibre_corners = [to_lng_lat(-x_r, z_f), to_lng_lat(x_r, z_f), to_lng_lat(x_r, z_n), to_lng_lat(-x_r, z_n)]
    footprint = {"lat": bev_center_lat, "lon": bev_center_lon, "heading": view_heading, "width_m": 2 * x_r, "height_m": z_f - z_n, "corners": maplibre_corners}
    
    annotated_rect = rect_img.copy()
    annotated_bev_bgr = raw_bev_bgr.copy()
    
    R = get_camera_rotation_matrix(bev_pitch, bev_yaw, bev_roll)
    
    roi_pts_3d = [[-x_r, cam_h, z_n], [x_r, cam_h, z_n], [x_r, cam_h, z_f], [-x_r, cam_h, z_f]]
    roi_pts_2d = []
    for pt in roi_pts_3d:
        xyz = R @ np.array(pt)
        if xyz[2] > 1e-5:
            u = int((K[0,0] * xyz[0] / xyz[2]) + K[0,2])
            v = int((K[1,1] * xyz[1] / xyz[2]) + K[1,2])
            roi_pts_2d.append([u, v])
            
    if len(roi_pts_2d) == 4:
        pts_array = np.array(roi_pts_2d, np.int32).reshape((-1, 1, 2))
        overlay = annotated_rect.copy()
        cv2.fillPoly(overlay, [pts_array], (0, 255, 255))
        cv2.addWeighted(overlay, 0.15, annotated_rect, 0.85, 0, annotated_rect)
        cv2.polylines(annotated_rect, [pts_array], isClosed=True, color=(0, 255, 255), thickness=2)
        
    if options.get('draw_grid', False):
        annotated_rect = draw_bev_grid(annotated_rect, K, cam_h, bev_pitch, bev_roll, bev_yaw, z_n, z_f, x_r)
        
    defects, geojson_features = [], []
    inference_img = apply_ego_mask(rect_img.copy(), mask_pct=0.15) if options.get('ego_mask', True) else rect_img.copy()
    
    with model_lock:
        results = model.predict(source=inference_img, conf=options.get('conf_thresh', 0.25), save=False, verbose=False)
        
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
                    
                    geo_coords = [[local_to_global(gps_lat, gps_lon, view_heading, (pt[0][0]*gsd)-x_r, z_f-(pt[0][1]*gsd))[1], local_to_global(gps_lat, gps_lon, view_heading, (pt[0][0]*gsd)-x_r, z_f-(pt[0][1]*gsd))[0]] for pt in contour]
                    if geo_coords[0] != geo_coords[-1]: geo_coords.append(geo_coords[0])
                    
                    defects.append({"class": class_name, "conf": round(conf, 2), "area_sqm": round(area_sqm, 4), "color": hex_color})
                    geojson_features.append({
                        "type": "Feature",
                        "properties": {"class": class_name, "area_sqm": round(area_sqm, 4), "view": view_name, "color": hex_color, "filename": original_filename, "conf": round(conf, 2)},
                        "geometry": {"type": "Polygon", "coordinates": [geo_coords]}
                    })

    annotated_bev_rgba = apply_bev_feathering(annotated_bev_bgr)
    
    cv2.imwrite(os.path.join(output_dir, f"rect_{view_name}_{base_filename}"), annotated_rect)
    cv2.imwrite(os.path.join(output_dir, f"bev_{view_name}_{base_name_no_ext}.png"), annotated_bev_rgba)
    
    return defects, geojson_features, footprint
