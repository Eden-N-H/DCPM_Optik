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
    
    # Prevents catastrophic Out-Of-Memory (OOM) crashes by dynamically scaling 
    # down the output pixel density if the geographic span selected is too large.
    MAX_CANVAS_DIM = 8192
    if W_canvas > MAX_CANVAS_DIM or H_canvas > MAX_CANVAS_DIM:
        scale_factor = MAX_CANVAS_DIM / max(W_canvas, H_canvas)
        PPM = PPM * scale_factor
        W_canvas = int((max_x - min_x) * PPM)
        H_canvas = int((max_y - min_y) * PPM)
    
    canvas = np.zeros((H_canvas, W_canvas, 3), dtype=np.uint8)
    
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
            # We must use the original unscaled 50.0 PPM here since the source BEV 
            # image was inherently generated at 50 PPM, regardless of the canvas scale
            x_local_m = (u - W/2.0) / 50.0
            y_local_m = (H/2.0 - v) / 50.0
            
            x_base_m = dx + x_local_m * math.cos(a) + y_local_m * math.sin(a)
            y_base_m = dy - x_local_m * math.sin(a) + y_local_m * math.cos(a)
            
            cx = (x_base_m - min_x) * PPM
            cy = (max_y - y_base_m) * PPM
            pts_canvas.append([cx, cy])
            
        M = cv2.getAffineTransform(pts_img, np.float32(pts_canvas))
        warped = cv2.warpAffine(bev_img, M, (W_canvas, H_canvas))
        mask = (warped > 0).astype(np.uint8)
        
        canvas = np.where(mask > 0, warped, canvas)
        
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
