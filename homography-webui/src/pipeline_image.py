import os
import cv2
import math
import numpy as np
from ultralytics.utils.plotting import colors

from geo_math import local_to_global, global_to_local
from cv_projections import equirectangular_to_rectilinear
from cv_bev import get_bev_homography, apply_bev_feathering, draw_bev_grid, apply_ego_mask
from depth_integration import is_pothole_class, attach_depth_to_detection

def _run_sam2_on_points(image_bgr, points, predictor, sam2_lock=None):
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    points_np = np.array(points, dtype=np.float32)
    if points_np.ndim != 2 or points_np.shape[0] < 3:
        return None

    h, w = image_bgr.shape[:2]
    x_min, y_min = points_np.min(axis=0)
    x_max, y_max = points_np.max(axis=0)

    box_w = max(x_max - x_min, 1.0)
    box_h = max(y_max - y_min, 1.0)
    margin_x = box_w * 0.05
    margin_y = box_h * 0.05

    x_min = float(np.clip(x_min - margin_x, 0, w - 1))
    y_min = float(np.clip(y_min - margin_y, 0, h - 1))
    x_max = float(np.clip(x_max + margin_x, 0, w - 1))
    y_max = float(np.clip(y_max + margin_y, 0, h - 1))

    box = np.array([[x_min, y_min, x_max, y_max]], dtype=np.float32)
    
    locked = False
    if sam2_lock:
        sam2_lock.acquire()
        locked = True
    try:
        predictor.set_image(image_rgb)
        
        point_labels = np.ones(len(points_np), dtype=np.int32)
        masks, scores, logits = predictor.predict(
            point_coords=points_np,
            point_labels=point_labels,
            box=box, 
            multimask_output=False
        )
    finally:
        if locked:
            sam2_lock.release()

    if masks.ndim == 4: mask = masks[0, 0]
    elif masks.ndim == 3: mask = masks[0]
    else: mask = masks

    binary_mask = (mask > 0).astype(np.uint8)
    if binary_mask.sum() == 0: return None

    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return None

    largest_contour = max(contours, key=cv2.contourArea)
    epsilon = 0.002 * cv2.arcLength(largest_contour, True)
    approx = cv2.approxPolyDP(largest_contour, epsilon, True)
    pts = approx.reshape(-1, 2).astype(np.float32)
    
    if pts.shape[0] < 3: return None
    return pts

def _run_sam2_masks(rect_img, results, sam2_predictor, sam2_lock=None):
    if sam2_predictor is None: return None
    from sam2_integration import run_sam2_on_detections
    all_sam2_results = []
    
    locked = False
    if sam2_lock:
        sam2_lock.acquire()
        locked = True
    try:
        for r in results:
            if r.boxes is not None and len(r.boxes) > 0:
                image_rgb = cv2.cvtColor(rect_img, cv2.COLOR_BGR2RGB)
                sam2_out = run_sam2_on_detections(image_rgb, r, sam2_predictor)
                for pts, cls_id, conf, class_name in sam2_out:
                    mask_color_bgr = colors(cls_id, bgr=True)
                    all_sam2_results.append((pts, cls_id, conf, class_name, mask_color_bgr))
    finally:
        if locked:
            sam2_lock.release()
            
    return all_sam2_results if all_sam2_results else None

def _annotate_with_sam2(annotated_rect, sam2_results, model_names):
    for pts, cls_id, conf, class_name, mask_color_bgr in sam2_results:
        pts_int = pts.astype(np.int32)
        overlay = annotated_rect.copy()
        cv2.fillPoly(overlay, [pts_int], color=mask_color_bgr)
        cv2.addWeighted(overlay, 0.4, annotated_rect, 0.6, 0, annotated_rect)
        cv2.polylines(annotated_rect, [pts_int], isClosed=True, color=mask_color_bgr, thickness=2)
        
        x_min, y_min = pts_int.min(axis=0)
        label = f"{class_name} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(annotated_rect, (x_min, y_min - th - 4), (x_min + tw, y_min), mask_color_bgr, -1)
        cv2.putText(annotated_rect, label, (x_min, y_min - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return annotated_rect

def _render_simple_view_from_detections(process_meta, view_name, calib, filename, output_dir):
    source_path = os.path.join(output_dir, f"source_{filename}")
    rect_img = cv2.imread(source_path)
    annotated_rect = rect_img.copy()
    
    detections = process_meta.get('view_meta', {}).get(view_name, {}).get('detections', [])
    defects = []
    
    for det_idx, det in enumerate(detections):
        if det.get('hidden', False): continue
        pts = np.array(det['polygon'], dtype=np.int32)
        class_name = det['class_name']
        conf = det['conf']
        color_bgr = tuple(det['color_bgr'])
        hex_color = det['hex_color']
        
        overlay = annotated_rect.copy()
        cv2.fillPoly(overlay, [pts], color=color_bgr)
        cv2.addWeighted(overlay, 0.4, annotated_rect, 0.6, 0, annotated_rect)
        cv2.polylines(annotated_rect, [pts], isClosed=True, color=color_bgr, thickness=2)
        
        area_px = cv2.contourArea(pts)
        # Orthographic/simple frames have no camera model (no K/BEV
        # homography), so depth estimation is intentionally not run here --
        # only the 360/standard-video BEV pipeline supports it.
        defects.append({"class": class_name, "conf": round(conf, 2), "area_sqm": round(area_px, 0), "color": hex_color, "det_idx": det_idx})
        
    rect_filename = f"rect_{view_name}_{filename}"
    cv2.imwrite(os.path.join(output_dir, rect_filename), annotated_rect)
    
    base_name_no_ext = os.path.splitext(filename)[0]
    bev_filename = f"bev_{view_name}_{base_name_no_ext}.png"
    cv2.imwrite(os.path.join(output_dir, bev_filename), annotated_rect)
    
    return defects, []

def render_view_from_detections(process_meta, view_name, calib, filename, output_dir):
    telemetry = process_meta.get('telemetry', {})
    options = process_meta.get('options', {})
    original_filename = process_meta.get('original_name', '')
    media_type = options.get('media_type', '360-video')
    
    if media_type == 'orthographic':
        return _render_simple_view_from_detections(process_meta, view_name, calib, filename, output_dir)
        
    gps_lat = telemetry.get('lat') or 0.0
    gps_lon = telemetry.get('lon') or 0.0
    heading = float(telemetry.get('heading') or 0.0)
    
    source_path = os.path.join(output_dir, f"source_{filename}")
    rect_img_for_proj, K, grav_vec, eff_yaw = get_projected_image(source_path, telemetry, options, view_name, calib)
    base_name_no_ext = os.path.splitext(filename)[0]
    
    cam_h = float(calib.get('cam_height') or 1.6)
    y_min = float(calib.get('z_near') or 1.2)
    y_max = float(calib.get('z_far') or 5.0)
    road_width = float(calib.get('lane_width') or 6.0)
    x_r = road_width / 2.0

    H_mat, bev_w, bev_h, PPM, v_down, v_forward, v_right = get_bev_homography(
        K, cam_h, grav_vec, eff_yaw, y_min, y_max, road_width
    )
    gsd = 1.0 / PPM
    
    raw_bev_bgr = cv2.warpPerspective(rect_img_for_proj, H_mat, (bev_w, bev_h))
    yaw_offset = float(calib.get('yaw_offset', 0.0))
    heading_offset = 180.0 if view_name == 'rear' else 0.0
    view_heading = (heading + heading_offset + yaw_offset) % 360
    
    annotated_rect = rect_img_for_proj.copy()
    annotated_bev_bgr = raw_bev_bgr.copy()
    
    if options.get('draw_grid', False):
        annotated_rect = draw_bev_grid(annotated_rect, K, cam_h, v_down, v_forward, v_right, y_min, y_max, x_r)
        
    detections = process_meta.get('view_meta', {}).get(view_name, {}).get('detections', [])
    defects = []
    geojson_features = []
    
    try:
        H_inv = np.linalg.inv(H_mat)
    except:
        H_inv = np.eye(3)
    
    for det_idx, det in enumerate(detections):
        if det.get('hidden', False): continue
        
        class_name = det['class_name']
        conf = det['conf']
        color_bgr = tuple(det['color_bgr'])
        hex_color = det['hex_color']
        is_stitched = det.get('is_stitched', False)
        is_grouped = det.get('is_grouped', False)
        spanned_frames = det.get('spanned_frames', [])

        # Depth is never recomputed here -- this function only re-renders
        # existing polygons (e.g. after grouping/alignment). Depth stats
        # were computed once at creation time (process_single_image /
        # recalculate_view / modify_defects) and persisted onto the
        # detection dict; grouping.py only ever dict.update()s extra keys
        # onto that same dict, so these fields survive grouping/stitching
        # untouched. We just pass them through into the response here.
        depth_props = {
            "depth_max_mm": det.get("depth_max_mm"),
            "depth_mean_mm": det.get("depth_mean_mm"),
            "depth_quality": det.get("depth_quality"),
            "depth_map_file": det.get("depth_map_file"),
        }
        
        if is_stitched:
            geo_coords = det.get('world_polygon', [])
            if not geo_coords: continue
            
            bev_pts = []
            rect_pts = []
            
            for lon, lat in geo_coords:
                x_local, z_local = global_to_local(gps_lat, gps_lon, view_heading, lat, lon)
                u = (x_local + x_r) / gsd
                v = (y_max - z_local) / gsd
                bev_pts.append([int(u), int(v)])
                
                vec = np.array([u, v, 1.0])
                rect_pt = H_inv @ vec
                if rect_pt[2] != 0:
                    rect_pts.append([int(rect_pt[0]/rect_pt[2]), int(rect_pt[1]/rect_pt[2])])
                    
            if len(rect_pts) > 2:
                pts = np.array(rect_pts, dtype=np.int32)
                overlay = annotated_rect.copy()
                cv2.fillPoly(overlay, [pts], color=color_bgr)
                cv2.addWeighted(overlay, 0.4, annotated_rect, 0.6, 0, annotated_rect)
                cv2.polylines(annotated_rect, [pts], isClosed=True, color=color_bgr, thickness=2)
                
            if len(bev_pts) > 2:
                pts = np.array(bev_pts, dtype=np.int32)
                overlay = annotated_bev_bgr.copy()
                cv2.fillPoly(overlay, [pts], color=color_bgr)
                cv2.addWeighted(overlay, 0.4, annotated_bev_bgr, 0.6, 0, annotated_bev_bgr)
                
            defects.append({"class": class_name, "conf": round(conf, 2), "area_sqm": round(det.get('area_sqm', 0), 4), "color": hex_color, "det_idx": det_idx, "is_stitched": True, "spanned_frames": spanned_frames, **depth_props})
            geojson_features.append({
                "type": "Feature",
                "properties": {"class": class_name, "area_sqm": round(det.get('area_sqm', 0), 4), "view": view_name, "color": hex_color, "filename": filename, "conf": round(conf, 2), "det_idx": det_idx, "is_stitched": True, "spanned_frames": spanned_frames, **depth_props},
                "geometry": {"type": "Polygon", "coordinates": [geo_coords]}
            })
            continue
            
        pts = np.array(det['polygon'], dtype=np.int32)
        
        overlay = annotated_rect.copy()
        cv2.fillPoly(overlay, [pts], color=color_bgr)
        cv2.addWeighted(overlay, 0.4, annotated_rect, 0.6, 0, annotated_rect)
        cv2.polylines(annotated_rect, [pts], isClosed=True, color=color_bgr, thickness=2)
        
        mask_canvas = np.zeros(rect_img_for_proj.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask_canvas, [pts], 255)
        contours, _ = cv2.findContours(cv2.warpPerspective(mask_canvas, H_mat, (bev_w, bev_h)), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area_sqm = cv2.contourArea(contour) * (gsd ** 2)
            if area_sqm <= 1e-5: continue
            
            overlay = annotated_bev_bgr.copy()
            cv2.fillPoly(overlay, [contour], color=color_bgr)
            cv2.addWeighted(overlay, 0.4, annotated_bev_bgr, 0.6, 0, annotated_bev_bgr)
            
            if telemetry.get('lat') is not None and telemetry.get('lon') is not None:
                geo_coords = [[local_to_global(gps_lat, gps_lon, view_heading, (pt[0][0]*gsd)-x_r, y_max-(pt[0][1]*gsd))[1], local_to_global(gps_lat, gps_lon, view_heading, (pt[0][0]*gsd)-x_r, y_max-(pt[0][1]*gsd))[0]] for pt in contour]
                if geo_coords[0] != geo_coords[-1]: geo_coords.append(geo_coords[0])
            else: geo_coords = []
            
            defects.append({"class": class_name, "conf": round(conf, 2), "area_sqm": round(area_sqm, 4), "color": hex_color, "det_idx": det_idx, "is_grouped": is_grouped, "spanned_frames": spanned_frames, **depth_props})
            if geo_coords:
                geojson_features.append({
                    "type": "Feature",
                    "properties": {"class": class_name, "area_sqm": round(area_sqm, 4), "view": view_name, "color": hex_color, "filename": filename, "conf": round(conf, 2), "det_idx": det_idx, "is_grouped": is_grouped, "spanned_frames": spanned_frames, **depth_props},
                    "geometry": {"type": "Polygon", "coordinates": [geo_coords]}
                })
                
    annotated_bev_rgba = apply_bev_feathering(annotated_bev_bgr)
    cv2.imwrite(os.path.join(output_dir, f"rect_{view_name}_{filename}"), annotated_rect)
    cv2.imwrite(os.path.join(output_dir, f"bev_{view_name}_{base_name_no_ext}.png"), annotated_bev_rgba)
    
    return defects, geojson_features

def _process_simple_frame(img_input, model, base_filename, output_dir, options, model_lock, original_filename="", sam2_predictor=None, sam2_lock=None):
    img_mat = cv2.imread(img_input) if isinstance(img_input, str) else img_input
    base_name_no_ext = os.path.splitext(base_filename)[0]
    
    source_filename = f"source_{base_filename}"
    if not os.path.exists(os.path.join(output_dir, source_filename)):
        cv2.imwrite(os.path.join(output_dir, source_filename), img_mat)
    
    conf_thresh = options.get('conf_thresh', 0.25)
    skip_ai = options.get('skip_ai', False)
    
    results = []
    if not skip_ai and model is not None:
        with model_lock:
            results = model.predict(source=img_mat, conf=conf_thresh, save=False, verbose=False)
    
    annotated = img_mat.copy()
    all_defects = []
    view_meta = {"front": {"K": [[1,0,0],[0,1,0],[0,0,1]], "detections": []}}
    
    sam2_results = _run_sam2_masks(img_mat, results, sam2_predictor, sam2_lock) if (not skip_ai and sam2_predictor) else None

    if sam2_results:
        _annotate_with_sam2(annotated, sam2_results, model.names)
        for pts, cls_id, conf, class_name, color in sam2_results:
            hex_color = f"#{int(color[2]):02x}{int(color[1]):02x}{int(color[0]):02x}"
            area_px = cv2.contourArea(np.array(pts, dtype=np.int32))
            
            view_meta["front"]["detections"].append({
                "class_name": class_name, "conf": float(conf),
                "color_bgr": [int(c) for c in color], "hex_color": hex_color,
                "polygon": pts.tolist()
            })
            det_idx = len(view_meta["front"]["detections"]) - 1
            # No camera model in orthographic mode -- depth is skipped here.
            all_defects.append({"class": class_name, "conf": round(conf, 2), "area_sqm": round(area_px, 0), "color": hex_color, "det_idx": det_idx})
    else:
        for r in results:
            annotated = r.plot(img=annotated)
            if r.boxes is not None:
                for i in range(len(r.boxes)):
                    cls_id = int(r.boxes.cls[i])
                    conf = float(r.boxes.conf[i])
                    class_name = model.names[cls_id]
                    mask_color_bgr = colors(cls_id, bgr=True)
                    hex_color = f"#{int(mask_color_bgr[2]):02x}{int(mask_color_bgr[1]):02x}{int(mask_color_bgr[0]):02x}"
                    
                    box = r.boxes.xyxy[i].cpu().numpy()
                    pts = np.array([[box[0], box[1]], [box[2], box[1]], [box[2], box[3]], [box[0], box[3]]])
                    
                    view_meta["front"]["detections"].append({
                        "class_name": class_name, "conf": float(conf),
                        "color_bgr": [int(c) for c in mask_color_bgr], "hex_color": hex_color,
                        "polygon": pts.tolist()
                    })
                    det_idx = len(view_meta["front"]["detections"]) - 1
                    all_defects.append({"class": class_name, "conf": round(conf, 2), "area_sqm": 0, "color": hex_color, "det_idx": det_idx})
    
    rect_filename = f"rect_front_{base_filename}"
    cv2.imwrite(os.path.join(output_dir, rect_filename), annotated)
    
    generated_files = {"front": {"raw_rect": source_filename, "raw_bev": rect_filename, "edit_bev": source_filename, "rect": rect_filename, "bev": rect_filename}}
    calibrations = {"front": {"yaw_offset": 0, "fov": 100, "cam_height": 1.6, "z_near": 1.2, "z_far": 5.0, "lane_width": 6.0}}
    bev_footprints = {"front": {"lat": 0, "lon": 0, "heading": 0, "width_m": 6.0, "height_m": 3.8, "corners": [[0,0],[0,0],[0,0],[0,0]]}}
    
    return {"front": all_defects}, [], generated_files, bev_footprints, view_meta, calibrations

def process_single_image(img_input, model, base_filename, output_dir, telemetry, options, model_lock, original_filename="", sam2_predictor=None, sam2_lock=None):
    media_type = options.get('media_type', '360-video')
    if media_type == 'orthographic':
        return _process_simple_frame(img_input, model, base_filename, output_dir, options, model_lock, original_filename, sam2_predictor, sam2_lock)
        
    gps_lat, gps_lon = telemetry.get('lat') or 0.0, telemetry.get('lon') or 0.0
    heading = telemetry.get('heading', 0.0)
    fov_val = float(telemetry.get('xfov') or telemetry.get('fov') or 100.0)
        
    cam_height = options.get('cam_height', 1.6)
    is_360 = options.get('is_360', True)
    
    y_min_base = float(options.get('z_near') or 1.2)
    y_max_base = float(options.get('z_far') or 5.0)
    road_width_base = float(options.get('lane_width') or 6.0)
    x_range_base = road_width_base / 2.0

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
            "yaw_offset": 0.0,
            "fov": fov_val, "cam_height": cam_height,
            "z_near": y_min_base, "z_far": y_max_base, "lane_width": road_width_base 
        }
        
        rect_img, K, grav_vec, eff_yaw = get_projected_image(os.path.join(output_dir, source_filename), telemetry, options, view_name, calibrations[view_name])
        view_meta = {"K": K.tolist(), "detections": []}

        H_mat, bev_w, bev_h, PPM, v_down, v_forward, v_right = get_bev_homography(
            K, cam_height, grav_vec, eff_yaw, y_min_base, y_max_base, road_width_base
        )
        gsd = 1.0 / PPM
        
        raw_rect_filename = f"raw_rect_{view_name}_{base_filename}"
        cv2.imwrite(os.path.join(output_dir, raw_rect_filename), rect_img)
        raw_bev_bgr = cv2.warpPerspective(rect_img, H_mat, (bev_w, bev_h))
        raw_bev_rgba = apply_bev_feathering(raw_bev_bgr)
        raw_bev_filename = f"raw_bev_{view_name}_{base_name_no_ext}.png" 
        cv2.imwrite(os.path.join(output_dir, raw_bev_filename), raw_bev_rgba)
        
        edit_bev_filename = f"edit_bev_{view_name}_{base_name_no_ext}.png"
        cv2.imwrite(os.path.join(output_dir, edit_bev_filename), raw_bev_bgr)

        yaw_offset = float(calibrations[view_name].get('yaw_offset', 0.0))
        view_heading = (heading + config['heading_offset'] + yaw_offset) % 360
        
        bev_center_lat, bev_center_lon = local_to_global(gps_lat, gps_lon, view_heading, 0, (y_min_base + y_max_base) / 2.0)
        
        def to_lng_lat(x, z):
            lat_out, lon_out = local_to_global(gps_lat, gps_lon, view_heading, x, z)
            return [lon_out, lat_out]
            
        maplibre_corners = [to_lng_lat(-x_range_base, y_max_base), to_lng_lat(x_range_base, y_max_base), to_lng_lat(x_range_base, y_min_base), to_lng_lat(-x_range_base, y_min_base)]
        bev_footprints[view_name] = {"lat": bev_center_lat, "lon": bev_center_lon, "heading": view_heading, "width_m": 2 * x_range_base, "height_m": y_max_base - y_min_base, "corners": maplibre_corners}

        annotated_rect = rect_img.copy()
        annotated_bev_bgr = raw_bev_bgr.copy()
        
        if options.get('draw_grid', False):
            annotated_rect = draw_bev_grid(annotated_rect, K, cam_height, v_down, v_forward, v_right, y_min_base, y_max_base, x_range_base)

        skip_ai = options.get('skip_ai', False)
        results = []
        if not skip_ai and model is not None:
            inference_img = apply_ego_mask(rect_img.copy(), mask_pct=0.15) if options.get('ego_mask', True) else rect_img.copy()
            with model_lock:
                results = model.predict(source=inference_img, conf=options.get('conf_thresh', 0.25), save=False, verbose=False)

        sam2_results = _run_sam2_masks(rect_img, results, sam2_predictor, sam2_lock) if (not skip_ai and sam2_predictor) else None

        if sam2_results:
            annotated_rect = _annotate_with_sam2(annotated_rect, sam2_results, model.names)
            for pts, cls_id, conf, class_name, mask_color_bgr in sam2_results:
                hex_color = f"#{int(mask_color_bgr[2]):02x}{int(mask_color_bgr[1]):02x}{int(mask_color_bgr[0]):02x}"
                
                view_meta["detections"].append({
                    "class_name": class_name, "conf": float(conf),
                    "color_bgr": [int(c) for c in mask_color_bgr], "hex_color": hex_color,
                    "polygon": pts.tolist()
                })
                this_det_idx = len(view_meta["detections"]) - 1

                mask_canvas = np.zeros(rect_img.shape[:2], dtype=np.uint8)
                cv2.fillPoly(mask_canvas, [pts.astype(np.int32)], 255)
                contours, _ = cv2.findContours(cv2.warpPerspective(mask_canvas, H_mat, (bev_w, bev_h)), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                _det_entries = []
                _geo_props_entries = []
                _area_total = 0.0

                for contour in contours:
                    area_sqm = cv2.contourArea(contour) * (gsd ** 2)
                    if area_sqm <= 1e-5: continue
                    
                    overlay = annotated_bev_bgr.copy()
                    cv2.fillPoly(overlay, [contour], color=mask_color_bgr)
                    cv2.addWeighted(overlay, 0.4, annotated_bev_bgr, 0.6, 0, annotated_bev_bgr)
                    
                    geo_coords = [[local_to_global(gps_lat, gps_lon, view_heading, (pt[0][0]*gsd)-x_range_base, y_max_base-(pt[0][1]*gsd))[1], local_to_global(gps_lat, gps_lon, view_heading, (pt[0][0]*gsd)-x_range_base, y_max_base-(pt[0][1]*gsd))[0]] for pt in contour]
                    if geo_coords[0] != geo_coords[-1]: geo_coords.append(geo_coords[0])
                    
                    defect_entry = {"class": class_name, "conf": round(conf, 2), "area_sqm": round(area_sqm, 4), "color": hex_color, "det_idx": this_det_idx}
                    all_defects[view_name].append(defect_entry)
                    _det_entries.append(defect_entry)
                    _area_total += area_sqm

                    geo_props = {"class": class_name, "area_sqm": round(area_sqm, 4), "view": view_name, "color": hex_color, "filename": base_filename, "conf": round(conf, 2), "det_idx": this_det_idx}
                    all_geojson_features.append({
                        "type": "Feature",
                        "properties": geo_props,
                        "geometry": {"type": "Polygon", "coordinates": [geo_coords]}
                    })
                    _geo_props_entries.append(geo_props)

                # Pothole-only, cached, computed once per detection (not per
                # BEV contour fragment) using its total BEV surface area.
                if is_pothole_class(class_name):
                    cache_key = f"{base_filename}_{view_name}_{this_det_idx}"
                    attach_depth_to_detection(
                        view_meta["detections"], this_det_idx, class_name,
                        rect_img, pts, K, cam_height, v_down, _area_total,
                        output_dir, cache_key,
                        extra_targets=[_det_entries, _geo_props_entries],
                        force=False
                    )
        else:
            for r in results:
                annotated_rect = r.plot(img=annotated_rect)
                if r.masks is not None:
                    for i, mask_pts in enumerate(r.masks.xy):
                        cls_id, conf = int(r.boxes.cls[i]), float(r.boxes.conf[i])
                        class_name = model.names[cls_id]
                        mask_color_bgr = colors(cls_id, bgr=True)
                        hex_color = f"#{int(mask_color_bgr[2]):02x}{int(mask_color_bgr[1]):02x}{int(mask_color_bgr[0]):02x}"
                        
                        view_meta["detections"].append({
                            "class_name": class_name, "conf": float(conf),
                            "color_bgr": [int(c) for c in mask_color_bgr], "hex_color": hex_color,
                            "polygon": mask_pts.tolist() 
                        })
                        this_det_idx = len(view_meta["detections"]) - 1

                        mask_canvas = np.zeros(rect_img.shape[:2], dtype=np.uint8)
                        cv2.fillPoly(mask_canvas, [np.array(mask_pts, dtype=np.int32)], 255)
                        contours, _ = cv2.findContours(cv2.warpPerspective(mask_canvas, H_mat, (bev_w, bev_h)), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        
                        _det_entries = []
                        _geo_props_entries = []
                        _area_total = 0.0

                        for contour in contours:
                            area_sqm = cv2.contourArea(contour) * (gsd ** 2)
                            if area_sqm <= 1e-5: continue
                            
                            overlay = annotated_bev_bgr.copy()
                            cv2.fillPoly(overlay, [contour], color=mask_color_bgr)
                            cv2.addWeighted(overlay, 0.4, annotated_bev_bgr, 0.6, 0, annotated_bev_bgr)
                            
                            geo_coords = [[local_to_global(gps_lat, gps_lon, view_heading, (pt[0][0]*gsd)-x_range_base, y_max_base-(pt[0][1]*gsd))[1], local_to_global(gps_lat, gps_lon, view_heading, (pt[0][0]*gsd)-x_range_base, y_max_base-(pt[0][1]*gsd))[0]] for pt in contour]
                            if geo_coords[0] != geo_coords[-1]: geo_coords.append(geo_coords[0])
                            
                            defect_entry = {"class": class_name, "conf": round(conf, 2), "area_sqm": round(area_sqm, 4), "color": hex_color, "det_idx": this_det_idx}
                            all_defects[view_name].append(defect_entry)
                            _det_entries.append(defect_entry)
                            _area_total += area_sqm

                            geo_props = {"class": class_name, "area_sqm": round(area_sqm, 4), "view": view_name, "color": hex_color, "filename": base_filename, "conf": round(conf, 2), "det_idx": this_det_idx}
                            all_geojson_features.append({
                                "type": "Feature",
                                "properties": geo_props,
                                "geometry": {"type": "Polygon", "coordinates": [geo_coords]}
                            })
                            _geo_props_entries.append(geo_props)

                        if is_pothole_class(class_name):
                            cache_key = f"{base_filename}_{view_name}_{this_det_idx}"
                            attach_depth_to_detection(
                                view_meta["detections"], this_det_idx, class_name,
                                rect_img, np.array(mask_pts, dtype=np.float32), K, cam_height, v_down, _area_total,
                                output_dir, cache_key,
                                extra_targets=[_det_entries, _geo_props_entries],
                                force=False
                            )

        annotated_bev_rgba = apply_bev_feathering(annotated_bev_bgr)
        rect_filename = f"rect_{view_name}_{base_filename}"
        bev_filename = f"bev_{view_name}_{base_name_no_ext}.png"
        
        cv2.imwrite(os.path.join(output_dir, rect_filename), annotated_rect)
        cv2.imwrite(os.path.join(output_dir, bev_filename), annotated_bev_rgba)
        
        generated_files[view_name] = {"raw_rect": raw_rect_filename, "raw_bev": raw_bev_filename, "edit_bev": edit_bev_filename, "rect": rect_filename, "bev": bev_filename}
        view_meta_all[view_name] = view_meta
        
    return all_defects, all_geojson_features, generated_files, bev_footprints, view_meta_all, calibrations

def get_projected_image(source_path, telemetry, options, view_name, calib):
    img_mat = cv2.imread(source_path)
    is_360 = options.get('is_360', True)
    
    yaw_offset = float(calib.get('yaw_offset') or 0.0)
    
    if view_name == 'rear':
        yaw_offset += 180.0
        
    fov = float(calib.get('fov') or 100.0)

    if is_360:
        dynamic_pitch = float(telemetry.get('pitch') or 0.0)
        dynamic_roll = float(telemetry.get('roll') or 0.0)

        rect_img, K = equirectangular_to_rectilinear(img_mat, fov, dynamic_pitch, dynamic_roll, yaw_offset)
        grav_vec = [0.0, 1.0, 0.0]
        eff_yaw = 0.0
    else:
        rect_img = img_mat.copy()
        h, w = rect_img.shape[:2]
        
        if options.get('undistort', True):
            from cv_bev import get_fisheye_maps
            x_fov = float(telemetry.get('xfov') or fov)
            y_fov = float(telemetry.get('yfov')) if telemetry.get('yfov') else math.degrees(2 * math.atan(math.tan(math.radians(x_fov) / 2) * (h / w)))
            map1, map2, K = get_fisheye_maps(w, h, x_fov, y_fov)
            rect_img = cv2.remap(rect_img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        else:
            f = (w / 2.0) / math.tan(math.radians(fov) / 2.0)
            K = np.array([[f, 0, w / 2.0], [0, f, h / 2.0], [0, 0, 1]], dtype=np.float64)
        
        raw_grav = telemetry.get('grav_vec')
        grav_vec = list(raw_grav) if raw_grav is not None else [0.0, 1.0, 0.0]
        eff_yaw = yaw_offset
        
    return rect_img, K, grav_vec, eff_yaw

def generate_grid_preview(source_path, process_meta, view_name, calib):
    rect_img, K, grav_vec, eff_yaw = get_projected_image(source_path, process_meta['telemetry'], process_meta['options'], view_name, calib)
    cam_h = float(calib.get('cam_height') or 1.6)
    y_min, y_max = float(calib.get('z_near') or 1.2), float(calib.get('z_far') or 5.0)
    road_width = float(calib.get('lane_width') or 6.0)
    
    H_mat, bev_w, bev_h, PPM, v_down, v_forward, v_right = get_bev_homography(
        K, cam_h, grav_vec, eff_yaw, y_min, y_max, road_width
    )
    x_r = road_width / 2.0
    
    roi_pts_3d = [
        (-x_r * v_right) + (y_min * v_forward) + (cam_h * v_down),
        (x_r * v_right) + (y_min * v_forward) + (cam_h * v_down),
        (x_r * v_right) + (y_max * v_forward) + (cam_h * v_down),
        (-x_r * v_right) + (y_max * v_forward) + (cam_h * v_down)
    ]
    roi_pts_2d = []
    
    for pt in roi_pts_3d:
        p_img = K @ pt
        if p_img[2] > 1e-5:
            roi_pts_2d.append([int(p_img[0]/p_img[2]), int(p_img[1]/p_img[2])])
            
    preview_img = rect_img.copy()
    if len(roi_pts_2d) == 4:
        pts_array = np.array(roi_pts_2d, np.int32).reshape((-1, 1, 2))
        overlay = preview_img.copy()
        cv2.fillPoly(overlay, [pts_array], (0, 255, 255))
        cv2.addWeighted(overlay, 0.15, preview_img, 0.85, 0, preview_img)
        cv2.polylines(preview_img, [pts_array], isClosed=True, color=(0, 255, 255), thickness=2)
        
    if process_meta['options'].get('draw_grid', False):
        preview_img = draw_bev_grid(preview_img, K, cam_h, v_down, v_forward, v_right, y_min, y_max, x_r)
        
    return preview_img

def recalculate_view(source_path, telemetry, options, view_name, calib, original_filename, output_dir, base_filename, model, model_lock, sam2_predictor=None, sam2_lock=None):
    rect_img, K, grav_vec, eff_yaw = get_projected_image(source_path, telemetry, options, view_name, calib)
    gps_lat, gps_lon = telemetry.get('lat') or 0.0, telemetry.get('lon') or 0.0
    heading = float(telemetry.get('heading') or 0.0)
    
    cam_h = float(calib.get('cam_height') or 1.6)
    y_min, y_max = float(calib.get('z_near') or 1.2), float(calib.get('z_far') or 5.0)
    road_width = float(calib.get('lane_width') or 6.0)
    x_r = road_width / 2.0

    H_mat, bev_w, bev_h, PPM, v_down, v_forward, v_right = get_bev_homography(
        K, cam_h, grav_vec, eff_yaw, y_min, y_max, road_width
    )
    gsd = 1.0 / PPM
    
    raw_bev_bgr = cv2.warpPerspective(rect_img, H_mat, (bev_w, bev_h))
    base_name_no_ext = os.path.splitext(base_filename)[0]
    
    cv2.imwrite(os.path.join(output_dir, f"raw_rect_{view_name}_{base_filename}"), rect_img)
    cv2.imwrite(os.path.join(output_dir, f"raw_bev_{view_name}_{base_name_no_ext}.png"), apply_bev_feathering(raw_bev_bgr))
    cv2.imwrite(os.path.join(output_dir, f"edit_bev_{view_name}_{base_name_no_ext}.png"), raw_bev_bgr)
    
    yaw_offset = float(calib.get('yaw_offset', 0.0))
    heading_offset = 180.0 if view_name == 'rear' else 0.0
    view_heading = (heading + heading_offset + yaw_offset) % 360
        
    def to_lng_lat(x, z):
        lat_out, lon_out = local_to_global(gps_lat, gps_lon, view_heading, x, z)
        return [lon_out, lat_out]
        
    maplibre_corners = [to_lng_lat(-x_r, y_max), to_lng_lat(x_r, y_max), to_lng_lat(x_r, y_min), to_lng_lat(-x_r, y_min)]
    bev_center_lat, bev_center_lon = local_to_global(gps_lat, gps_lon, view_heading, 0, (y_min + y_max) / 2.0)
    footprint = {"lat": bev_center_lat, "lon": bev_center_lon, "heading": view_heading, "width_m": 2 * x_r, "height_m": y_max - y_min, "corners": maplibre_corners}
    
    annotated_rect = rect_img.copy()
    annotated_bev_bgr = raw_bev_bgr.copy()
    
    if options.get('draw_grid', False):
        annotated_rect = draw_bev_grid(annotated_rect, K, cam_h, v_down, v_forward, v_right, y_min, y_max, x_r)
        
    skip_ai = options.get('skip_ai', False)
    results = []
    
    if not skip_ai and model is not None:
        inference_img = apply_ego_mask(rect_img.copy(), mask_pct=0.15) if options.get('ego_mask', True) else rect_img.copy()
        with model_lock:
            results = model.predict(source=inference_img, conf=options.get('conf_thresh', 0.25), save=False, verbose=False)
        
    sam2_results = _run_sam2_masks(rect_img, results, sam2_predictor, sam2_lock) if (not skip_ai and sam2_predictor) else None
    
    defects, geojson_features, view_meta_detections = [], [], []
    
    if sam2_results:
        annotated_rect = _annotate_with_sam2(annotated_rect, sam2_results, model.names)
        for pts, cls_id, conf, class_name, mask_color_bgr in sam2_results:
            hex_color = f"#{int(mask_color_bgr[2]):02x}{int(mask_color_bgr[1]):02x}{int(mask_color_bgr[0]):02x}"
            
            view_meta_detections.append({
                "class_name": class_name, "conf": float(conf),
                "color_bgr": [int(c) for c in mask_color_bgr], "hex_color": hex_color,
                "polygon": pts.tolist()
            })
            this_det_idx = len(view_meta_detections) - 1

            mask_canvas = np.zeros(rect_img.shape[:2], dtype=np.uint8)
            cv2.fillPoly(mask_canvas, [pts.astype(np.int32)], 255)
            contours, _ = cv2.findContours(cv2.warpPerspective(mask_canvas, H_mat, (bev_w, bev_h)), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            _det_entries = []
            _geo_props_entries = []
            _area_total = 0.0

            for contour in contours:
                area_sqm = cv2.contourArea(contour) * (gsd ** 2)
                if area_sqm <= 1e-5: continue
                
                overlay = annotated_bev_bgr.copy()
                cv2.fillPoly(overlay, [contour], color=mask_color_bgr)
                cv2.addWeighted(overlay, 0.4, annotated_bev_bgr, 0.6, 0, annotated_bev_bgr)
                
                geo_coords = [[local_to_global(gps_lat, gps_lon, view_heading, (pt[0][0]*gsd)-x_r, y_max-(pt[0][1]*gsd))[1], local_to_global(gps_lat, gps_lon, view_heading, (pt[0][0]*gsd)-x_r, y_max-(pt[0][1]*gsd))[0]] for pt in contour]
                if geo_coords[0] != geo_coords[-1]: geo_coords.append(geo_coords[0])
                
                defect_entry = {"class": class_name, "conf": round(conf, 2), "area_sqm": round(area_sqm, 4), "color": hex_color, "det_idx": this_det_idx}
                defects.append(defect_entry)
                _det_entries.append(defect_entry)
                _area_total += area_sqm

                geo_props = {"class": class_name, "area_sqm": round(area_sqm, 4), "view": view_name, "color": hex_color, "filename": base_filename, "conf": round(conf, 2), "det_idx": this_det_idx}
                geojson_features.append({
                    "type": "Feature",
                    "properties": geo_props,
                    "geometry": {"type": "Polygon", "coordinates": [geo_coords]}
                })
                _geo_props_entries.append(geo_props)

            # Recalibration explicitly changed geometry, so always recompute
            # (never reuse a stale cached depth map from the old calibration).
            if is_pothole_class(class_name):
                cache_key = f"{base_filename}_{view_name}_{this_det_idx}"
                attach_depth_to_detection(
                    view_meta_detections, this_det_idx, class_name,
                    rect_img, pts, K, cam_h, v_down, _area_total,
                    output_dir, cache_key,
                    extra_targets=[_det_entries, _geo_props_entries],
                    force=True
                )
    else:
        for r in results:
            annotated_rect = r.plot(img=annotated_rect)
            if r.masks is not None:
                for i, mask_pts in enumerate(r.masks.xy):
                    cls_id, conf = int(r.boxes.cls[i]), float(r.boxes.conf[i])
                    class_name = model.names[cls_id]
                    mask_color_bgr = colors(cls_id, bgr=True)
                    hex_color = f"#{int(mask_color_bgr[2]):02x}{int(mask_color_bgr[1]):02x}{int(mask_color_bgr[0]):02x}"
                    
                    view_meta_detections.append({
                        "class_name": class_name, "conf": float(conf),
                        "color_bgr": [int(c) for c in mask_color_bgr], "hex_color": hex_color,
                        "polygon": mask_pts.tolist() 
                    })
                    this_det_idx = len(view_meta_detections) - 1

                    mask_canvas = np.zeros(rect_img.shape[:2], dtype=np.uint8)
                    cv2.fillPoly(mask_canvas, [np.array(mask_pts, dtype=np.int32)], 255)
                    contours, _ = cv2.findContours(cv2.warpPerspective(mask_canvas, H_mat, (bev_w, bev_h)), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    
                    _det_entries = []
                    _geo_props_entries = []
                    _area_total = 0.0

                    for contour in contours:
                        area_sqm = cv2.contourArea(contour) * (gsd ** 2)
                        if area_sqm <= 1e-5: continue
                        
                        overlay = annotated_bev_bgr.copy()
                        cv2.fillPoly(overlay, [contour], color=mask_color_bgr)
                        cv2.addWeighted(overlay, 0.4, annotated_bev_bgr, 0.6, 0, annotated_bev_bgr)
                        
                        geo_coords = [[local_to_global(gps_lat, gps_lon, view_heading, (pt[0][0]*gsd)-x_r, y_max-(pt[0][1]*gsd))[1], local_to_global(gps_lat, gps_lon, view_heading, (pt[0][0]*gsd)-x_r, y_max-(pt[0][1]*gsd))[0]] for pt in contour]
                        if geo_coords[0] != geo_coords[-1]: geo_coords.append(geo_coords[0])
                        
                        defect_entry = {"class": class_name, "conf": round(conf, 2), "area_sqm": round(area_sqm, 4), "color": hex_color, "det_idx": this_det_idx}
                        defects.append(defect_entry)
                        _det_entries.append(defect_entry)
                        _area_total += area_sqm

                        geo_props = {"class": class_name, "area_sqm": round(area_sqm, 4), "view": view_name, "color": hex_color, "filename": base_filename, "conf": round(conf, 2), "det_idx": this_det_idx}
                        geojson_features.append({
                            "type": "Feature",
                            "properties": geo_props,
                            "geometry": {"type": "Polygon", "coordinates": [geo_coords]}
                        })
                        _geo_props_entries.append(geo_props)

                    if is_pothole_class(class_name):
                        cache_key = f"{base_filename}_{view_name}_{this_det_idx}"
                        attach_depth_to_detection(
                            view_meta_detections, this_det_idx, class_name,
                            rect_img, np.array(mask_pts, dtype=np.float32), K, cam_h, v_down, _area_total,
                            output_dir, cache_key,
                            extra_targets=[_det_entries, _geo_props_entries],
                            force=True
                        )

    annotated_bev_rgba = apply_bev_feathering(annotated_bev_bgr)
    cv2.imwrite(os.path.join(output_dir, f"rect_{view_name}_{base_filename}"), annotated_rect)
    cv2.imwrite(os.path.join(output_dir, f"bev_{view_name}_{base_name_no_ext}.png"), annotated_bev_rgba)
    
    return defects, geojson_features, footprint, view_meta_detections
