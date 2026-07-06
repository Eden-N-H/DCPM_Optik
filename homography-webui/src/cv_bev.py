import cv2
import math
import numpy as np

def apply_bev_feathering(bev_bgr):
    h, w = bev_bgr.shape[:2]
    rgba = cv2.cvtColor(bev_bgr, cv2.COLOR_BGR2BGRA)
    alpha = np.ones((h, w), dtype=np.float32)
    
    # Cap the fade amounts to fixed pixel lengths (e.g., ~2m or 200px at a GSD of 0.01)
    # This prevents the image from looking "cropped" when the user
    # manually increases the orthographic projection length (z_far).
    top_fade = min(int(h * 0.3), 200)
    for y in range(top_fade):
        alpha[y, :] *= (y / top_fade) ** 2.0
        
    side_fade = min(int(w * 0.15), 150)
    for x in range(side_fade):
        fade_val = (x / side_fade) ** 1.5
        alpha[:, x] *= fade_val
        alpha[:, w - 1 - x] *= fade_val
        
    rgba[:, :, 3] = (alpha * 255).astype(np.uint8)
    return rgba

def apply_ego_mask(img, mask_pct=0.15):
    h, w = img.shape[:2]
    mask_h = int(h * mask_pct)
    img[h - mask_h:, :] = 0
    return img

def get_camera_rotation_matrix(pitch_deg, yaw_deg, roll_deg):
    p = math.radians(-pitch_deg)  
    y = math.radians(yaw_deg)
    r = math.radians(roll_deg)
    
    Rx = np.array([[1, 0, 0], [0, math.cos(p), -math.sin(p)], [0, math.sin(p), math.cos(p)]])
    Ry = np.array([[math.cos(y), 0, math.sin(y)], [0, 1, 0], [-math.sin(y), 0, math.cos(y)]])
    Rz = np.array([[math.cos(r), -math.sin(r), 0], [math.sin(r), math.cos(r), 0], [0, 0, 1]])
                   
    return Ry @ Rx @ Rz

def get_bev_homography(K, cam_height_m, pitch_deg, roll_deg, yaw_deg, z_near, z_far, x_range, gsd=0.01):
    road_pts = np.array([[-x_range, z_near], [x_range, z_near], [x_range, z_far], [-x_range, z_far]], dtype=np.float32)
    bev_w, bev_h = int((2 * x_range) / gsd), int((z_far - z_near) / gsd)
    bev_pts = np.array([[0, bev_h], [bev_w, bev_h], [bev_w, 0], [0, 0]], dtype=np.float32)
    
    R = get_camera_rotation_matrix(pitch_deg, yaw_deg, roll_deg)
    
    rect_pts = []
    for pt in road_pts:
        xyz = R @ np.array([pt[0], cam_height_m, pt[1]])
        if xyz[2] <= 1e-5: xyz[2] = 1e-5 
        u = (K[0,0] * xyz[0] / xyz[2]) + K[0,2]
        v = (K[1,1] * xyz[1] / xyz[2]) + K[1,2]
        rect_pts.append([u, v])
        
    H = cv2.getPerspectiveTransform(np.array(rect_pts, dtype=np.float32), bev_pts)
    return H, bev_w, bev_h, gsd

def draw_bev_grid(img, K, cam_height_m, pitch_deg, roll_deg, yaw_deg, z_near, z_far, x_range):
    R = get_camera_rotation_matrix(pitch_deg, yaw_deg, roll_deg)
    
    for z in np.arange(math.floor(z_near), math.ceil(z_far) + 1, 1.0):
        pts = []
        for x in np.arange(-x_range, x_range + 0.5, 0.5):
            xyz = R @ np.array([x, cam_height_m, z])
            if xyz[2] <= 0: continue
            u = int((K[0,0] * xyz[0] / xyz[2]) + K[0,2])
            v = int((K[1,1] * xyz[1] / xyz[2]) + K[1,2])
            pts.append((u, v))
        if len(pts) > 1:
            for i in range(len(pts)-1): cv2.line(img, pts[i], pts[i+1], (0, 255, 255), 2)
            
    for x in np.arange(math.floor(-x_range), math.ceil(x_range) + 1, 1.0):
        pts = []
        for z in np.arange(z_near, z_far + 0.5, 0.5):
            xyz = R @ np.array([x, cam_height_m, z])
            if xyz[2] <= 0: continue
            u = int((K[0,0] * xyz[0] / xyz[2]) + K[0,2])
            v = int((K[1,1] * xyz[1] / xyz[2]) + K[1,2])
            pts.append((u, v))
        if len(pts) > 1:
            for i in range(len(pts)-1): cv2.line(img, pts[i], pts[i+1], (0, 255, 255), 2)
            
    return img