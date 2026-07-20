import cv2
import numpy as np
import math

def find_vanishing_point_hough(img_bgr):
    """
    Uses traditional CV (Canny + Hough Lines + Intersection clustering) 
    to automatically estimate the vanishing point of a road.
    """
    h, w = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    
    # Mask out the sky (top 35%) and ego-hood (bottom 15%)
    mask = np.zeros_like(gray)
    top_y = int(h * 0.35)
    bot_y = int(h * 0.85)
    mask[top_y:bot_y, :] = 255
    
    edges = cv2.Canny(gray, 50, 150)
    edges = cv2.bitwise_and(edges, mask)
    
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=40, minLineLength=40, maxLineGap=15)
    
    if lines is None: 
        return None
        
    filtered_lines = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if x2 == x1: continue
        slope = (y2 - y1) / (x2 - x1)
        # VP lines are usually between 16 deg (0.3) and 78 deg (5.0)
        if abs(slope) < 0.3 or abs(slope) > 5.0: 
            continue
        filtered_lines.append((x1, y1, x2, y2, slope))
        
    if len(filtered_lines) < 2: 
        return None
        
    intersections = []
    for i in range(len(filtered_lines)):
        for j in range(i+1, len(filtered_lines)):
            x1, y1, x2, y2, m1 = filtered_lines[i]
            x3, y3, x4, y4, m2 = filtered_lines[j]
            
            # Must be distinct lines (e.g. left vs right lane boundaries)
            if abs(m1 - m2) < 0.3: 
                continue 
            
            A = np.array([[-m1, 1], [-m2, 1]])
            b = np.array([y1 - m1*x1, y3 - m2*x3])
            try:
                pt = np.linalg.solve(A, b)
                ix, iy = int(pt[0]), int(pt[1])
                # Only keep intersections that fall somewhat inside the image bounds
                if -w*0.5 <= ix <= w*1.5 and 0 <= iy < h:
                    intersections.append([ix, iy])
            except np.linalg.LinAlgError: 
                pass
                
    if not intersections: 
        return None
        
    intersections = np.array(intersections)
    
    median_x = np.median(intersections[:, 0])
    median_y = np.median(intersections[:, 1])
    
    # Filter out wild outliers to get a precise mean
    valid = [pt for pt in intersections if abs(pt[0] - median_x) < w*0.15 and abs(pt[1] - median_y) < h*0.15]
    
    if not valid: 
        return int(median_x), int(median_y)
        
    valid = np.array(valid)
    return int(np.mean(valid[:, 0])), int(np.mean(valid[:, 1]))

def calculate_pitch_yaw_deltas(u, v, w, h, fov, is_360):
    """
    Translates a 2D pixel coordinate (u, v) representing the true horizon/VP 
    into absolute Pitch and Yaw compensation offsets.
    """
    f = (w / 2.0) / math.tan(math.radians(fov) / 2.0)
    cx, cy = w / 2.0, h / 2.0
    
    if is_360:
        # For 360 video, the offset is a CORRECTION applied to the spherical projection 
        # to level the virtual camera. If VP is below center, camera is looking UP, 
        # so we need to pitch DOWN (negative correction).
        dx = u - cx
        dy = cy - v 
    else:
        # For standard video, the offset represents the PHYSICAL pose of the camera.
        # If VP is below center, the camera is physically pitched UP (positive pose).
        dx = cx - u
        dy = v - cy

    yaw_angle = math.degrees(math.atan2(dx, f))
    pitch_angle = math.degrees(math.atan2(dy, f)) 
    
    return pitch_angle, yaw_angle
