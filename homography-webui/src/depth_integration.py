# --- START OF FILE src/depth_integration.py ---
import cv2
import numpy as np
import math
import os
import json

_depth_model = None
def load_depth_model(model_id=None): return None

# Substring match against a detection's class name. Kept as a tuple so more
# synonyms (e.g. "crater") can be added later without touching call sites.
POTHOLE_KEYWORDS = ("pothole",)


def is_pothole_class(class_name):
    """True if this detection's class should get depth estimation."""
    if not class_name:
        return False
    name = str(class_name).lower()
    return any(k in name for k in POTHOLE_KEYWORDS)


def estimate_pothole_depth(rect_img, sam2_polygon, K_rect, cam_height_m, v_down, exact_area_sqm):
    """
    Calculates morphometric depth based on exact geometric surface area.

    Returns (max_depth_mm, mean_depth_mm, quality, overlay_img, polygon_list).

    This is a heuristic, not a true measured depth: it assumes a roughly
    bowl-shaped defect and derives depth purely from the already-computed
    BEV surface area plus a normalized distance-transform "crater" shape of
    the mask. Accuracy depends on how circular/bowl-like the defect
    actually is and on the accuracy of the upstream BEV area calculation.
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
    # Uses the internal Area to derive Diameter, yielding morphometric depth
    equiv_diameter_m = 2.0 * math.sqrt(exact_area_sqm / math.pi)
    max_depth_m = equiv_diameter_m * 0.28
    physical_depth = crater_shape * max_depth_m

    mask_bool = pothole_mask > 0
    if np.any(mask_bool):
        mean_depth_m = float(np.mean(physical_depth[mask_bool]))
        # Rough proxy for how "well-formed"/crater-like the masked region
        # is (a compact, round region gives a higher, more trustworthy
        # peak distance relative to its own size). Not a calibrated
        # confidence score -- purely a relative quality indicator for the UI.
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


def get_pothole_depth(rect_img, polygon, K_rect, cam_height_m, v_down, area_sqm,
                       output_dir, cache_key, force=False):
    """
    Cached wrapper around estimate_pothole_depth.

    Persists the rendered depth-map heatmap PNG and a small stats JSON
    sidecar into `output_dir` under `cache_key` (served directly as a
    static file, e.g. /static/uploads/depth_<cache_key>.png), so
    re-visiting the same defect (reopening the UI, or a re-render pass
    that doesn't touch the underlying pixels) doesn't recompute anything.

    Returns a dict: {depth_max_mm, depth_mean_mm, depth_quality, depth_map_file}
    """
    stats_path = os.path.join(output_dir, f"depth_{cache_key}.json")
    map_filename = f"depth_{cache_key}.png"
    map_path = os.path.join(output_dir, map_filename)

    if not force and os.path.exists(stats_path) and os.path.exists(map_path):
        try:
            with open(stats_path, 'r') as f:
                return json.load(f)
        except Exception:
            pass

    max_mm, mean_mm, quality, overlay, _poly_list = estimate_pothole_depth(
        rect_img, polygon, K_rect, cam_height_m, v_down, area_sqm
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
        "depth_map_file": map_ref
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
                               extra_targets=None, force=False):
    """
    Runs (or reuses cached) depth estimation for a single detection IF its
    class is pothole-like, and writes the resulting stats onto:
      - the persisted view_meta detection dict at det_idx (source of truth,
        survives grouping/re-render since those only ever dict.update()
        additional keys onto existing detections), and
      - any extra dict lists passed in (e.g. the in-flight `defects` list
        and geojson `properties` dicts for the current response), so the
        very same request that created the defect also returns its depth.

    No-ops (and touches nothing) for non-pothole classes, per the
    requirement not to run depth estimation on irrelevant defects.
    """
    if not is_pothole_class(class_name):
        return None

    stats = get_pothole_depth(
        rect_img, polygon_pts, K_rect, cam_height_m, v_down,
        area_sqm_total, output_dir, cache_key, force=force
    )

    if 0 <= det_idx < len(view_meta_detections):
        view_meta_detections[det_idx].update(stats)

    if extra_targets:
        for lst in extra_targets:
            for d in lst:
                d.update(stats)

    return stats
# --- END OF FILE ---
