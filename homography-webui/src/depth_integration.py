# --- START OF FILE src/depth_integration.py ---
import cv2
import numpy as np
import math

_depth_model = None
def load_depth_model(model_id=None): return None

def estimate_pothole_depth(rect_img, sam2_polygon, K_rect, cam_height_m, v_down, exact_area_sqm):
    """Calculates morphometric depth based on exact geometric surface area."""
    if exact_area_sqm <= 0.001 or len(sam2_polygon) < 3:
        return 0.0, None, sam2_polygon

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
        return 0.0, None, sam2_polygon.tolist()

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
    
    return max_depth_m * 1000.0, overlay, sam2_polygon.tolist()
# --- END OF FILE ---