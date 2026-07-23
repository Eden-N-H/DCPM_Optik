import cv2
import numpy as np
import math
import os
from geo_math import haversine_distance, calculate_bearing

def create_corridor(frames, upload_folder):
    """
    Stitches multiple single-frame BEV images into a single continuous 
    orthographic corridor map. Uses midline-seam compositing to eliminate
    the overlap/gap pattern visible on curves: each frame is clipped to
    the region closer to its own center than to any neighbor's center,
    producing clean trapezoid tiles that seamlessly cover the curve.
    """
    PPM = 50.0
    
    if not frames:
        return None, None
        
    base_f = frames[0]
    base_lat = base_f['footprint']['lat']
    base_lon = base_f['footprint']['lon']
    base_heading = base_f['footprint']['heading']
    
    centers_m = []
    for f in frames:
        dist = haversine_distance(base_lat, base_lon, f['footprint']['lat'], f['footprint']['lon'])
        bearing = calculate_bearing(base_lat, base_lon, f['footprint']['lat'], f['footprint']['lon'])
        angle = math.radians(bearing - base_heading)
        dx = dist * math.sin(angle)
        dy = dist * math.cos(angle)
        centers_m.append((dx, dy, f))
        
    min_x = min(c[0] for c in centers_m) - 10.0
    max_x = max(c[0] for c in centers_m) + 10.0
    min_y = min(c[1] for c in centers_m) - 10.0
    max_y = max(c[1] for c in centers_m) + 10.0
    
    W_canvas = int((max_x - min_x) * PPM)
    H_canvas = int((max_y - min_y) * PPM)
    
    MAX_CANVAS_DIM = 8192
    if W_canvas > MAX_CANVAS_DIM or H_canvas > MAX_CANVAS_DIM:
        scale_factor = MAX_CANVAS_DIM / max(W_canvas, H_canvas)
        PPM = PPM * scale_factor
        W_canvas = int((max_x - min_x) * PPM)
        H_canvas = int((max_y - min_y) * PPM)
    
    canvas = np.zeros((H_canvas, W_canvas, 3), dtype=np.uint8)
    
    # Pre-compute all frame centers in canvas pixel coordinates for
    # nearest-center voronoi assignment during compositing
    canvas_centers = []
    for dx, dy, f in centers_m:
        cx_px = (dx - min_x) * PPM
        cy_px = (max_y - dy) * PPM
        canvas_centers.append((cx_px, cy_px))

    for idx, (dx, dy, f) in enumerate(centers_m):
        img_name = os.path.basename(f['bev_url'].split('?')[0])
        img_path = os.path.join(upload_folder, img_name)
        if not os.path.exists(img_path):
            continue
            
        bev_img = cv2.imread(img_path)
        if bev_img is None:
            continue
            
        H, W = bev_img.shape[:2]
        d_heading = f['footprint']['heading'] - base_heading
        a = math.radians(d_heading)
        
        pts_img = np.float32([[0, 0], [W, 0], [0, H]])
        pts_canvas = []
        
        for u, v in pts_img:
            x_local_m = (u - W/2.0) / 50.0
            y_local_m = (H/2.0 - v) / 50.0
            
            x_base_m = dx + x_local_m * math.cos(a) + y_local_m * math.sin(a)
            y_base_m = dy - x_local_m * math.sin(a) + y_local_m * math.cos(a)
            
            cx = (x_base_m - min_x) * PPM
            cy = (max_y - y_base_m) * PPM
            pts_canvas.append([cx, cy])
            
        M = cv2.getAffineTransform(pts_img, np.float32(pts_canvas))
        warped = cv2.warpAffine(bev_img, M, (W_canvas, H_canvas))
        
        # Mask: only pixels with actual road content (not black BEV corners).
        # BEV frames have black triangular corners where the fisheye FOV doesn't
        # reach. These must be excluded so they don't overwrite valid road content
        # from adjacent frames. Use a brightness threshold instead of strict >0
        # to also exclude near-black noise pixels at BEV edges.
        gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        mask = (gray > 10).astype(np.uint8)

        if mask.sum() == 0:
            continue

        # Voronoi-style seam: for each pixel in this frame's footprint,
        # only keep it if this frame's center is the closest (or within a
        # feather margin of closest) among all frames. This naturally clips
        # each frame to its Voronoi cell, producing seamless tiles on curves.
        if len(canvas_centers) > 1:
            ys, xs = np.where(mask > 0)
            if len(xs) > 0:
                my_cx, my_cy = canvas_centers[idx]
                my_dist = np.sqrt((xs.astype(np.float32) - my_cx)**2 + (ys.astype(np.float32) - my_cy)**2)
                
                # Find minimum distance to any OTHER frame's center
                min_other_dist = np.full(len(xs), np.inf, dtype=np.float32)
                for other_idx, (ocx, ocy) in enumerate(canvas_centers):
                    if other_idx == idx:
                        continue
                    other_dist = np.sqrt((xs.astype(np.float32) - ocx)**2 + (ys.astype(np.float32) - ocy)**2)
                    min_other_dist = np.minimum(min_other_dist, other_dist)
                
                # diff > 0: we're closer (keep), diff < 0: other is closer (discard)
                FEATHER_PX = 10.0
                diff = min_other_dist - my_dist
                alpha_vals = np.clip((diff + FEATHER_PX) / (2.0 * FEATHER_PX), 0.0, 1.0).astype(np.float32)
                
                # Apply alpha mask: discard pixels where other frame is closer
                discard = alpha_vals <= 0
                warped[ys[discard], xs[discard]] = 0
                mask[ys[discard], xs[discard]] = 0
                
                # Partial alpha at boundaries
                partial = (alpha_vals > 0) & (alpha_vals < 1.0)
                if partial.any():
                    partial_ys = ys[partial]
                    partial_xs = xs[partial]
                    partial_alphas = alpha_vals[partial]
                    warped[partial_ys, partial_xs] = (warped[partial_ys, partial_xs].astype(np.float32) * partial_alphas[:, None]).astype(np.uint8)

        # Composite: blend where both have content, fill where only new has content
        existing_mask = (canvas.sum(axis=2) > 0)
        both_mask = existing_mask & (mask > 0)
        new_only_mask = (~existing_mask) & (mask > 0)
        
        # Where both exist: weighted average blend using the Voronoi alpha
        # The warped pixels have already been alpha-premultiplied by the Voronoi step.
        # To blend correctly: result = existing*(1-alpha) + warped_original*alpha
        # Since warped is already scaled by alpha, we need to reconstruct.
        # Simpler approach: just take the newer frame's content at full strength
        # since Voronoi already ensures minimal overlap.
        if both_mask.any():
            # Use the new frame where it has content in the overlap zone
            canvas[both_mask] = warped[both_mask]
        
        # Where only new frame: direct copy
        canvas[new_only_mask] = warped[new_only_mask]
        
    corridor_filename = f"corridor_{base_f['filename']}.png"
    cv2.imwrite(os.path.join(upload_folder, corridor_filename), canvas)
    
    corridor_meta = {
        "base_lat": base_lat,
        "base_lon": base_lon,
        "base_heading": base_heading,
        "min_x_m": min_x,
        "max_y_m": max_y,
        "PPM": PPM,
        "W_canvas": W_canvas,
        "H_canvas": H_canvas
    }
    
    return f"/static/uploads/{corridor_filename}", corridor_meta
