# --- START OF FILE src/depth_integration.py ---
import cv2
import numpy as np
import math
import os
import json
import time

from sklearn.linear_model import RANSACRegressor

# --- Global depth model singleton (lazy-loaded) ---
_depth_model = None

# Substring match against a detection's class name. Kept as a tuple so more
# synonyms (e.g. "crater") can be added later without touching call sites.
POTHOLE_KEYWORDS = ("pothole",)

# Supported depth estimation methods
DEPTH_METHOD_GEOMETRY = "geometry"
DEPTH_METHOD_DEPTHANYTHING = "depthanything"
DEFAULT_DEPTH_METHOD = DEPTH_METHOD_GEOMETRY


def is_pothole_class(class_name):
    """True if this detection's class should get depth estimation."""
    if not class_name:
        return False
    name = str(class_name).lower()
    return any(k in name for k in POTHOLE_KEYWORDS)


# ─────────────────────────────────────────────────────────────────────────────
# Depth Anything V2 model management
# ─────────────────────────────────────────────────────────────────────────────

def load_depth_model(model_id="depth-anything/Depth-Anything-V2-Small-hf"):
    """Lazy-load the Depth Anything V2 pipeline. Returns the model or None on failure."""
    global _depth_model
    if _depth_model is not None:
        return _depth_model
    try:
        import torch
        from transformers import pipeline as hf_pipeline
        device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[*] Loading Depth Anything V2 on {device}...")
        _depth_model = hf_pipeline(task="depth-estimation", model=model_id, device=device)
        print("[+] Depth model loaded successfully.")
        return _depth_model
    except Exception as e:
        print(f"[!] Failed to load Depth Anything V2: {e}")
        _depth_model = None
        return None


def is_depth_model_loaded():
    """Check whether the depth model has already been loaded."""
    return _depth_model is not None


# ─────────────────────────────────────────────────────────────────────────────
# Method 1: Geometry-based morphometric depth (existing)
# ─────────────────────────────────────────────────────────────────────────────

def estimate_pothole_depth_geometry(rect_img, sam2_polygon, K_rect, cam_height_m, v_down, exact_area_sqm):
    """
    Calculates morphometric depth based on exact geometric surface area.

    Returns (max_depth_mm, mean_depth_mm, quality, overlay_img, polygon_list).

    This is a heuristic, not a true measured depth: it assumes a roughly
    bowl-shaped defect and derives depth purely from the already-computed
    BEV surface area plus a normalized distance-transform "crater" shape of
    the mask.
    """
    if exact_area_sqm <= 0.001 or len(sam2_polygon) < 3:
        empty_poly = list(sam2_polygon) if sam2_polygon is not None else []
        return 0.0, 0.0, 0.0, None, empty_poly

    h_img, w_img = rect_img.shape[:2]
    sam2_polygon = np.array(sam2_polygon, dtype=np.int32)

    x_min, y_min = np.min(sam2_polygon, axis=0)
    x_max, y_max = np.max(sam2_polygon, axis=0)

    pad_w = max(int((x_max - x_min) * 0.2), 20)
    pad_h = max(int((y_max - y_min) * 0.2), 20)

    c_x1 = max(0, x_min - pad_w)
    c_y1 = max(0, y_min - pad_h)
    c_x2 = min(w_img, x_max + pad_w)
    c_y2 = min(h_img, y_max + pad_h)

    crop_img = rect_img[c_y1:c_y2, c_x1:c_x2]
    crop_h, crop_w = crop_img.shape[:2]

    if crop_h < 5 or crop_w < 5:
        return 0.0, 0.0, 0.0, None, sam2_polygon.tolist()

    # 1. Create Local Mask
    poly_cropped = sam2_polygon - np.array([c_x1, c_y1])
    pothole_mask = np.zeros((crop_h, crop_w), dtype=np.uint8)
    cv2.fillPoly(pothole_mask, [poly_cropped], 255)

    # 2. OPTICAL MORPHOMETRY (Distance Transform Crater)
    dist_map = cv2.distanceTransform(pothole_mask, cv2.DIST_L2, 5)
    max_dist = np.max(dist_map)
    if max_dist > 0:
        crater_shape = dist_map / max_dist
        crater_shape = np.power(crater_shape, 0.8)
    else:
        crater_shape = np.zeros_like(dist_map)

    # 3. GEOMETRIC DEPTH ESTIMATION
    equiv_diameter_m = 2.0 * math.sqrt(exact_area_sqm / math.pi)
    max_depth_m = equiv_diameter_m * 0.28
    physical_depth = crater_shape * max_depth_m

    mask_bool = pothole_mask > 0
    if np.any(mask_bool):
        mean_depth_m = float(np.mean(physical_depth[mask_bool]))
        quality = float(np.clip(max_dist / (0.5 * max(crop_w, crop_h)), 0.0, 1.0))
    else:
        mean_depth_m = 0.0
        quality = 0.0

    # 4. VISUALIZATION (Topographic Heatmap)
    max_d = np.max(physical_depth) if np.max(physical_depth) > 0 else 0.05
    depth_vis = np.clip((physical_depth / max_d) * 255, 0, 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
    heatmap[pothole_mask == 0] = [0, 0, 0]

    crop_bg = cv2.bitwise_and(crop_img, crop_img, mask=cv2.bitwise_not(pothole_mask))
    combined_crop = cv2.add(crop_bg, heatmap)

    overlay = rect_img.copy()
    cv2.addWeighted(combined_crop, 0.75, crop_img, 0.25, 0, combined_crop)
    overlay[c_y1:c_y2, c_x1:c_x2] = combined_crop

    return max_depth_m * 1000.0, mean_depth_m * 1000.0, quality, overlay, sam2_polygon.tolist()


# ─────────────────────────────────────────────────────────────────────────────
# Method 2: Depth Anything V2 monocular depth estimation
# ─────────────────────────────────────────────────────────────────────────────

def estimate_pothole_depth_depthanything(rect_img, sam2_polygon, K_rect, cam_height_m, area_sqm=None):
    """
    Uses Depth Anything V2 monocular depth model to estimate pothole volume.

    Returns (max_depth_mm, mean_depth_mm, quality, overlay_img, polygon_list).

    Runs the full-image depth model, fits a RANSAC road plane in 3D space around
    the defect, and measures how far the pothole surface deviates below the plane.
    """
    global _depth_model
    if _depth_model is None:
        # Attempt lazy load
        load_depth_model()
    if _depth_model is None:
        return 0.0, 0.0, 0.0, None, (list(sam2_polygon) if sam2_polygon is not None else [])

    if len(sam2_polygon) < 3:
        return 0.0, 0.0, 0.0, None, list(sam2_polygon) if sam2_polygon is not None else []

    h_img, w_img = rect_img.shape[:2]

    # 1. Run Depth Inference on FULL IMAGE to preserve scene context
    from PIL import Image
    pil_full = Image.fromarray(cv2.cvtColor(rect_img, cv2.COLOR_BGR2RGB))
    depth_output = _depth_model(pil_full)
    full_disp_map = np.array(depth_output["depth"], dtype=np.float32)

    # Resize back to original image dimensions
    full_disp_map = cv2.resize(full_disp_map, (w_img, h_img), interpolation=cv2.INTER_LINEAR)

    # Affine Disparity Shift: subtract background offset before inversion
    disp_min = np.percentile(full_disp_map, 1)
    full_disp_shifted = np.maximum(full_disp_map - disp_min, 1e-5)
    full_rel_depth = 1.0 / full_disp_shifted

    # 2. Bounding Box & Cropping
    sam2_polygon = np.array(sam2_polygon, dtype=np.int32)
    x_min, y_min = np.min(sam2_polygon, axis=0)
    x_max, y_max = np.max(sam2_polygon, axis=0)

    pad_w = int((x_max - x_min) * 0.4)
    pad_h = int((y_max - y_min) * 0.4)

    c_x1 = max(0, x_min - pad_w)
    c_y1 = max(0, y_min - pad_h)
    c_x2 = min(w_img, x_max + pad_w)
    c_y2 = min(h_img, y_max + pad_h)

    crop_img = rect_img[c_y1:c_y2, c_x1:c_x2]
    crop_h, crop_w = crop_img.shape[:2]

    if crop_h < 10 or crop_w < 10:
        return 0.0, 0.0, 0.0, None, sam2_polygon.tolist()

    # Slice the global depth maps down to our ROI
    disp_map = full_disp_map[c_y1:c_y2, c_x1:c_x2]
    rel_depth = full_rel_depth[c_y1:c_y2, c_x1:c_x2]

    # 3. Create Masks
    poly_cropped = sam2_polygon - np.array([c_x1, c_y1])
    pothole_mask = np.zeros((crop_h, crop_w), dtype=np.uint8)
    cv2.fillPoly(pothole_mask, [poly_cropped], 255)

    # 15px Structural Dilation Ring for RANSAC road plane fitting
    kernel = np.ones((15, 15), np.uint8)
    dilated_mask = cv2.dilate(pothole_mask, kernel, iterations=1)
    road_mask = cv2.bitwise_xor(dilated_mask, pothole_mask)

    # 4. Unproject to 3D Rays using K_rect
    fx, fy = K_rect[0, 0], K_rect[1, 1]
    cx, cy = K_rect[0, 2] - c_x1, K_rect[1, 2] - c_y1

    u, v = np.meshgrid(np.arange(crop_w), np.arange(crop_h))
    ray_x = (u - cx) / fx
    ray_y = (v - cy) / fy
    ray_z = np.ones_like(ray_x)
    rays = np.stack((ray_x, ray_y, ray_z), axis=-1)
    points_3d = rays * rel_depth[..., np.newaxis]

    # 5. Fit Plane via RANSAC
    road_pts = points_3d[road_mask == 255]
    if len(road_pts) < 10:
        return 0.0, 0.0, 0.0, None, sam2_polygon.tolist()

    X_features = road_pts[:, [0, 1]]
    Z_target = road_pts[:, 2]

    # Subsample to max 5000 points for RANSAC speed
    if len(X_features) > 5000:
        step = len(X_features) // 5000
        X_features = X_features[::step]
        Z_target = Z_target[::step]

    try:
        ransac = RANSACRegressor(residual_threshold=0.05)
        ransac.fit(X_features, Z_target)
        a, b = ransac.estimator_.coef_
        d = ransac.estimator_.intercept_
    except Exception:
        return 0.0, 0.0, 0.0, None, sam2_polygon.tolist()

    normal = np.array([a, b, -1])

    # 6. Calculate Absolute Scale Factor
    distance_to_plane = abs(d) / np.linalg.norm(normal)
    if distance_to_plane < 1e-6:
        return 0.0, 0.0, 0.0, None, sam2_polygon.tolist()
    scale_factor = cam_height_m / distance_to_plane

    # 7. Calculate Pothole Depths
    points_3d_real = points_3d * scale_factor
    d_real = d * scale_factor

    # Z_point is the physical ray depth. (aX + bY + d_real) is the Plane Z.
    # Potholes are further away → Z_point > Plane Z.
    plane_z_expected = points_3d_real[..., 0] * a + points_3d_real[..., 1] * b + d_real
    depths_2d = (points_3d_real[..., 2] - plane_z_expected) / np.linalg.norm(normal)
    depths_2d = depths_2d * (pothole_mask / 255.0)

    # 8. Physical Trimming
    depths_2d[depths_2d < 0.005] = 0.0  # Ignore < 5mm (surface noise)
    depths_2d = np.clip(depths_2d, 0.0, 0.400)  # Cap at 400mm

    physical_mask = (depths_2d > 0).astype(np.uint8) * 255

    # Refine polygon from physical depth mask
    refined_polygon = sam2_polygon.tolist()
    contours, _ = cv2.findContours(physical_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        approx = cv2.approxPolyDP(largest_contour, 0.002 * cv2.arcLength(largest_contour, True), True)
        approx += np.array([c_x1, c_y1])
        refined_polygon = approx.reshape(-1, 2).tolist()

    # 9. Area calculation (use provided area or compute from plane projection)
    if area_sqm is None or area_sqm <= 0:
        denominator = 1.0 - (a * ray_x) - (b * ray_y)
        denominator = np.where(np.abs(denominator) < 1e-6, 1e-6, denominator)
        Z_plane = d_real / denominator
        plane_3d = np.stack((ray_x * Z_plane, ray_y * Z_plane, Z_plane), axis=-1)
        dx = np.zeros_like(plane_3d)
        dy = np.zeros_like(plane_3d)
        dx[:, :-1, :] = plane_3d[:, 1:, :] - plane_3d[:, :-1, :]
        dx[:, -1, :] = dx[:, -2, :]
        dy[:-1, :, :] = plane_3d[1:, :, :] - plane_3d[:-1, :, :]
        dy[-1, :, :] = dy[-2, :, :]
        pixel_areas_sqm = np.linalg.norm(np.cross(dx, dy, axis=-1), axis=-1)
    else:
        n_valid = max(np.sum(physical_mask > 0), 1)
        pixel_areas_sqm = np.ones((crop_h, crop_w)) * (area_sqm / n_valid)

    # 10. Volume Integration
    valid_mask = depths_2d > 0
    valid_depths = depths_2d[valid_mask]

    if len(valid_depths) == 0:
        return 0.0, 0.0, 0.0, None, refined_polygon

    max_depth_m = float(np.percentile(valid_depths, 98))
    mean_depth_m = float(np.mean(valid_depths))

    # Quality metric: ratio of valid depth pixels to mask pixels (higher = more complete)
    mask_pixels = np.sum(pothole_mask > 0)
    quality = float(np.sum(valid_mask)) / max(mask_pixels, 1)
    quality = float(np.clip(quality, 0.0, 1.0))

    # 11. Visualization
    heatmap = cv2.applyColorMap(
        cv2.convertScaleAbs(disp_map, alpha=255.0 / max(np.max(disp_map), 1)),
        cv2.COLORMAP_JET
    )
    heatmap_masked = cv2.bitwise_and(heatmap, heatmap, mask=physical_mask)
    crop_bg = cv2.bitwise_and(crop_img, crop_img, mask=cv2.bitwise_not(physical_mask))
    combined_crop = cv2.add(crop_bg, heatmap_masked)

    overlay = rect_img.copy()
    cv2.addWeighted(combined_crop, 0.7, crop_img, 0.3, 0, combined_crop)
    overlay[c_y1:c_y2, c_x1:c_x2] = combined_crop

    return max_depth_m * 1000.0, mean_depth_m * 1000.0, quality, overlay, refined_polygon


# ─────────────────────────────────────────────────────────────────────────────
# Unified dispatch
# ─────────────────────────────────────────────────────────────────────────────

def estimate_pothole_depth(rect_img, sam2_polygon, K_rect, cam_height_m, v_down, exact_area_sqm,
                           method=DEFAULT_DEPTH_METHOD):
    """
    Unified entry point for pothole depth estimation.

    method: "geometry" (default, fast heuristic) or "depthanything" (ML-based monocular depth).
    Returns (max_depth_mm, mean_depth_mm, quality, overlay_img, polygon_list).
    """
    if method == DEPTH_METHOD_DEPTHANYTHING:
        return estimate_pothole_depth_depthanything(
            rect_img, sam2_polygon, K_rect, cam_height_m, area_sqm=exact_area_sqm
        )
    else:
        return estimate_pothole_depth_geometry(
            rect_img, sam2_polygon, K_rect, cam_height_m, v_down, exact_area_sqm
        )


# ─────────────────────────────────────────────────────────────────────────────
# Caching & attachment helpers
# ─────────────────────────────────────────────────────────────────────────────

def get_pothole_depth(rect_img, polygon, K_rect, cam_height_m, v_down, area_sqm,
                      output_dir, cache_key, force=False, method=DEFAULT_DEPTH_METHOD):
    """
    Cached wrapper around estimate_pothole_depth.

    Persists the rendered depth-map heatmap PNG and a small stats JSON
    sidecar into `output_dir` under `cache_key`.

    Returns a dict: {depth_max_mm, depth_mean_mm, depth_quality, depth_map_file, depth_method}
    """
    stats_path = os.path.join(output_dir, f"depth_{cache_key}.json")
    map_filename = f"depth_{cache_key}.png"
    map_path = os.path.join(output_dir, map_filename)

    if not force and os.path.exists(stats_path) and os.path.exists(map_path):
        try:
            with open(stats_path, 'r') as f:
                cached = json.load(f)
                # Only return cache if method matches
                if cached.get("depth_method", DEPTH_METHOD_GEOMETRY) == method:
                    return cached
        except Exception:
            pass

    max_mm, mean_mm, quality, overlay, _poly_list = estimate_pothole_depth(
        rect_img, polygon, K_rect, cam_height_m, v_down, area_sqm, method=method
    )

    map_ref = None
    if overlay is not None:
        try:
            cv2.imwrite(map_path, overlay)
            map_ref = map_filename
        except Exception:
            map_ref = None

    stats = {
        "depth_max_mm": round(max_mm, 1),
        "depth_mean_mm": round(mean_mm, 1),
        "depth_quality": round(quality, 3),
        "depth_map_file": map_ref,
        "depth_method": method,
    }
    try:
        with open(stats_path, 'w') as f:
            json.dump(stats, f)
    except Exception:
        pass

    return stats


def attach_depth_to_detection(view_meta_detections, det_idx, class_name,
                              rect_img, polygon_pts, K_rect, cam_height_m, v_down,
                              area_sqm_total, output_dir, cache_key,
                              extra_targets=None, force=False, method=DEFAULT_DEPTH_METHOD):
    """
    Runs (or reuses cached) depth estimation for a single detection IF its
    class is pothole-like, and writes the resulting stats onto the detection dict.

    No-ops for non-pothole classes.
    """
    if not is_pothole_class(class_name):
        return None

    stats = get_pothole_depth(
        rect_img, polygon_pts, K_rect, cam_height_m, v_down,
        area_sqm_total, output_dir, cache_key, force=force, method=method
    )

    if 0 <= det_idx < len(view_meta_detections):
        view_meta_detections[det_idx].update(stats)

    if extra_targets:
        for lst in extra_targets:
            for d in lst:
                d.update(stats)

    return stats
# --- END OF FILE ---
