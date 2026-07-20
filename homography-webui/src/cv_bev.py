import cv2
import math
import numpy as np

def apply_bev_feathering(bev_bgr):
    # Convert directly to RGBA without applying a feathering gradient 
    # to maintain clean, sharp edges on the orthographic projection preview
    rgba = cv2.cvtColor(bev_bgr, cv2.COLOR_BGR2BGRA)
    return rgba

def apply_ego_mask(img, mask_pct=0.15):
    h, w = img.shape[:2]
    mask_h = int(h * mask_pct)
    img[h - mask_h:, :] = 0
    return img

def get_fisheye_maps(W, H, x_fov_deg, y_fov_deg):
    x_fov_rad = np.radians(x_fov_deg)
    y_fov_rad = np.radians(y_fov_deg)
    
    fx_fish = (W / 2.0) / (x_fov_rad / 2.0)
    fy_fish = (H / 2.0) / (y_fov_rad / 2.0)
    K_fish = np.array([
        [fx_fish, 0, W/2.0], 
        [0, fy_fish, H/2.0], 
        [0, 0, 1]
    ], dtype=np.float64)
    
    fx_rect = (W / 2.0) / math.tan(x_fov_rad / 2.0)
    fy_rect = (H / 2.0) / math.tan(y_fov_rad / 2.0)
    
    K_rect = np.array([
        [fx_rect, 0, W/2.0],
        [0, fy_rect, H/2.0],
        [0, 0, 1]
    ], dtype=np.float64)
    
    D = np.zeros(4)
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(K_fish, D, np.eye(3), K_rect, (W, H), cv2.CV_32FC1)
    return map1, map2, K_rect

def get_camera_to_world_rotation(pitch_deg, yaw_deg, roll_deg):
    """
    Returns the exact Camera-to-World rotation matrix.
    Assumes camera intrinsic orientation: X right, Y down, Z forward.
    Positive pitch = up, Positive yaw = right, Positive roll = clockwise.
    """
    p = math.radians(pitch_deg)
    y = math.radians(yaw_deg)
    r = math.radians(roll_deg)
    
    def Rx(theta):
        c, s = math.cos(theta), math.sin(theta)
        return np.array([
            [1, 0, 0],
            [0, c, -s],
            [0, s, c]
        ], dtype=np.float64)
        
    def Ry(theta):
        c, s = math.cos(theta), math.sin(theta)
        return np.array([
            [c, 0, s],
            [0, 1, 0],
            [-s, 0, c]
        ], dtype=np.float64)
        
    def Rz(theta):
        c, s = math.cos(theta), math.sin(theta)
        return np.array([
            [c, -s, 0],
            [s, c, 0],
            [0, 0, 1]
        ], dtype=np.float64)
        
    # The transformation from Leveled World to Tilted Camera applies
    # Yaw globally, Pitch locally, then Roll locally.
    # Therefore R_c2w applies them in reverse: Roll, then Pitch, then Yaw.
    return Ry(y) @ Rx(p) @ Rz(r)

def get_bev_homography(K_rect, cam_height_m, grav_vec, yaw_offset, y_min, y_max, road_width):
    # 1. Absolute Down from IMU Gravity
    #    This is the SINGLE source of truth for camera pitch/roll. It reflects
    #    true vehicle/camera attitude (body roll, hill pitch, mounting tilt).
    g = np.array(grav_vec, dtype=np.float64)
    n = np.linalg.norm(g)
    v_down_base = g / n if n > 1e-6 else np.array([0, 1, 0], dtype=np.float64)
    
    # 2. Extract true Forward via Z-Axis (Optical Axis) Projection
    #    This is the singular mathematically perfect way to decouple pitch and roll 
    #    without introducing "false yaw". Projecting the Z-axis guarantees recovery 
    #    of the true vehicle forward vector even under extreme simultaneous pitch/roll.
    z_cam = np.array([0, 0, 1], dtype=np.float64)
    v_fwd_base = z_cam - np.dot(z_cam, v_down_base) * v_down_base
    n_fwd = np.linalg.norm(v_fwd_base)
    
    if n_fwd > 1e-6:
        v_fwd_base /= n_fwd
    else:
        # Gimbal lock fallback (camera looking perfectly straight down or straight up)
        y_cam = np.array([0, -1, 0], dtype=np.float64)
        v_fwd_base = y_cam - np.dot(y_cam, v_down_base) * v_down_base
        v_fwd_base /= np.linalg.norm(v_fwd_base)
            
    # 3. True Right is orthogonal to Down and Forward (Cross Product)
    v_right_base = np.cross(v_down_base, v_fwd_base)
        
    # 4. Apply Manual Yaw Calibration
    #    Yaw is a rotation of the world frame around the true Gravity vector (v_down_base).
    #    Positive yaw pans the camera Right, meaning the World rotates Left relative to the camera.
    y_rad = math.radians(yaw_offset)
    c, s = math.cos(y_rad), math.sin(y_rad)
    
    v_right = c * v_right_base - s * v_fwd_base
    v_forward = s * v_right_base + c * v_fwd_base
    v_down = v_down_base

    # 5. Define BEV Output Canvas Scale Constants (PPM = 50 Pixels per Metre)
    PPM = 50.0 
    bev_W = int(road_width * PPM)
    bev_H = int((y_max - y_min) * PPM)
    
    # 6. Build Analytical Ground-to-Image Camera Matrix (H_g2i)
    #    Maps a world coordinate (X, Z_fwd) lying on the explicit plane (Y = cam_height_m)
    #    directly onto internal image pixels (u, v).
    H_g2i = K_rect @ np.column_stack((v_right, v_forward, cam_height_m * v_down))
    
    # 7. Build Analytical BEV-to-Ground Scaling Matrix (M_bev2g)
    #    Provides mathematically robust pixel scaling across the BEV output image mapping.
    M_bev2g = np.array([
        [1.0 / PPM, 0.0, -(road_width / 2.0)],
        [0.0, -1.0 / PPM, y_max],
        [0.0, 0.0, 1.0]
    ], dtype=np.float64)
    
    # 8. Compose mappings securely and extract True Planar Inverse
    H_bev2i = H_g2i @ M_bev2g
    
    try:
        H_mat = np.linalg.inv(H_bev2i)
        if abs(H_mat[2, 2]) > 1e-9:
            H_mat /= H_mat[2, 2]
    except np.linalg.LinAlgError:
        H_mat = np.eye(3)
        
    return H_mat, bev_W, bev_H, PPM, v_down, v_forward, v_right

def draw_bev_grid(img, K_rect, cam_height_m, v_down, v_forward, v_right, y_min, y_max, x_range):
    # Draw horizontal lines (constant y/forward)
    for y_fwd in np.arange(math.floor(y_min), math.ceil(y_max) + 1, 1.0):
        pts = []
        for x in np.arange(-x_range, x_range + 0.5, 0.5):
            pt3d = (x * v_right) + (y_fwd * v_forward) + (cam_height_m * v_down)
            p_img = K_rect @ pt3d
            if p_img[2] > 1e-5:
                u, v = int(p_img[0]/p_img[2]), int(p_img[1]/p_img[2])
                pts.append((u, v))
        if len(pts) > 1:
            for i in range(len(pts)-1): cv2.line(img, pts[i], pts[i+1], (0, 255, 255), 2)
            
    # Draw vertical lines (constant x/right)
    for x_rt in np.arange(math.floor(-x_range), math.ceil(x_range) + 1, 1.0):
        pts = []
        for y_fwd in np.arange(y_min, y_max + 0.5, 0.5):
            pt3d = (x_rt * v_right) + (y_fwd * v_forward) + (cam_height_m * v_down)
            p_img = K_rect @ pt3d
            if p_img[2] > 1e-5:
                u, v = int(p_img[0]/p_img[2]), int(p_img[1]/p_img[2])
                pts.append((u, v))
        if len(pts) > 1:
            for i in range(len(pts)-1): cv2.line(img, pts[i], pts[i+1], (0, 255, 255), 2)
            
    return img
