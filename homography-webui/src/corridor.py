import cv2
import numpy as np
import math
import os
from geo_math import haversine_distance, calculate_bearing

def create_corridor(frames, upload_folder):
    """
    Stitches multiple single-frame BEV images into a single continuous 
    orthographic corridor map based on their physical GPS footprint geometries.
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
    
    canvas = np.zeros((H_canvas, W_canvas, 3), dtype=np.uint8)

    # Feather width (in canvas pixels) used to blend each new frame's own
    # boundary into whatever is already stitched onto the canvas. Purely a
    # compositing-quality improvement -- it does not touch any coordinate
    # or homography math, so geo-referenced defect positions are unaffected.
    #
    # Previously frames were composited with a hard boolean cutover
    # (`canvas = np.where(mask > 0, warped, canvas)`), which produces a
    # visible seam line at every frame boundary even when the underlying
    # geometric alignment is perfect (due to per-frame exposure/perspective
    # differences at the very edge). Feathering the new frame's edge and
    # alpha-blending it over existing content removes that visible seam
    # without affecting alignment accuracy.
    FEATHER_PX = 15.0

    for dx, dy, f in centers_m:
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
            x_local_m = (u - W/2.0) / PPM
            y_local_m = (H/2.0 - v) / PPM
            
            x_base_m = dx + x_local_m * math.cos(a) + y_local_m * math.sin(a)
            y_base_m = dy - x_local_m * math.sin(a) + y_local_m * math.cos(a)
            
            cx = (x_base_m - min_x) * PPM
            cy = (max_y - y_base_m) * PPM
            pts_canvas.append([cx, cy])
            
        M = cv2.getAffineTransform(pts_img, np.float32(pts_canvas))
        warped = cv2.warpAffine(bev_img, M, (W_canvas, H_canvas))
        mask = (warped.sum(axis=2) > 0).astype(np.uint8)

        if mask.sum() == 0:
            continue

        existing_mask = (canvas.sum(axis=2) > 0)

        # Distance transform of the new frame's own footprint gives a
        # smooth 0->1 ramp from its boundary inward, used only where we're
        # blending over pre-existing content. Regions of the canvas that
        # are still empty always get the new frame at full opacity, so the
        # overall corridor's leading/trailing edges are never faded out.
        dist = cv2.distanceTransform((mask * 255).astype(np.uint8), cv2.DIST_L2, 5)
        alpha = np.clip(dist / FEATHER_PX, 0.0, 1.0).astype(np.float32)[..., None]

        blended = (canvas.astype(np.float32) * (1.0 - alpha) + warped.astype(np.float32) * alpha)
        blended = np.clip(blended, 0, 255).astype(np.uint8)

        new_content = np.where(existing_mask[..., None], blended, warped)
        canvas = np.where(mask[..., None] > 0, new_content, canvas).astype(np.uint8)
        
    corridor_filename = f"corridor_{base_f['filename']}.png"
    cv2.imwrite(os.path.join(upload_folder, corridor_filename), canvas)
    
    corridor_meta = {
        "base_lat": base_lat,
        "base_lon": base_lon,
        "base_heading": base_heading,
        "min_x_m": min_x,
        "max_y_m": max_y,
        "PPM": PPM,
        # Needed by the frontend/backend to convert normalized (0-1) click
        # coordinates on the rendered corridor image back into actual
        # canvas pixel coordinates before dividing by PPM. Previously
        # omitted, which silently produced wrong world coordinates for
        # any multi-frame manual mask (see /modify_defects add_corridor).
        "W_canvas": W_canvas,
        "H_canvas": H_canvas
    }
    
    return f"/static/uploads/{corridor_filename}", corridor_meta
