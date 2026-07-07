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

# --- NEW FISHEYE UNDISTORTION (From Tester) ---
def get_fisheye_maps(W, H, x_fov_deg, y_fov_deg):
    """Precomputes the undistortion maps based on GoPro XFOV and YFOV telemetry."""
    x_fov_rad = np.radians(x_fov_deg)
    y_fov_rad = np.radians(y_fov_deg)
    
    fx_fish = (W / 2.0) / (x_fov_rad / 2.0)
    fy_fish = (H / 2.0) / (y_fov_rad / 2.0)
    K_fish = np.array([
        [fx_fish, 0, W/2.0], 
        [0, fy_fish, H/2.0], 
        [0, 0, 1]
    ])
    
    K_rect = K_fish.copy()
    D = np.zeros(4)
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(K_fish, D, np.eye(3), K_rect, (W, H), cv2.CV_32FC1)
    return map1, map2, K_rect

def apply_ui_offsets_to_vectors(g_vec, z_cam, pitch_deg, roll_deg, yaw_deg):
    """Applies the UI's manual calibration sliders to the base directional vectors."""
    p = math.radians(-pitch_deg)  
    y = math.radians(yaw_deg)
    r = math.radians(roll_deg)
    
    Rx = np.array([[1, 0, 0], [0, math.cos(p), -math.sin(p)], [0, math.sin(p), math.cos(p)]])
    Ry = np.array([[math.cos(y), 0, math.sin(y)], [0, 1, 0], [-math.sin(y), 0, math.cos(y)]])
    Rz = np.array([[math.cos(r), -math.sin(r), 0], [math.sin(r), math.cos(r), 0], [0, 0, 1]])
    
    R = Ry @ Rx @ Rz
    return R @ g_vec, R @ z_cam

# --- NEW VECTOR-BASED HOMOGRAPHY (From Tester) ---
# REPLACES Euler-based 'get_camera_rotation_matrix' and dynamic GSD scaling
def get_bev_homography(K_rect, cam_height_m, grav_vec, pitch_offset, roll_offset, yaw_offset, y_min, y_max, road_width):
    
    # 1. Normalize gravity
    g = np.array(grav_vec, dtype=np.float64)
    n = np.linalg.norm(g)
    if n > 1e-6: g = g / n
    else: g = np.array([0, 1, 0]) # Default down in image space

    z_cam = np.array([0, 0, 1], dtype=np.float64)

    # 2. Apply user UI calibration offsets
    v_down, z_cam_rot = apply_ui_offsets_to_vectors(g, z_cam, pitch_offset, roll_offset, yaw_offset)
    
    # 3. Build Orthonormal Basis
    v_forward = z_cam_rot - (np.dot(z_cam_rot, v_down) * v_down)
    n_fwd = np.linalg.norm(v_forward)
    if n_fwd > 1e-6: v_forward = v_forward / n_fwd
    else: v_forward = np.array([0, 0, 1])

    v_right = np.cross(v_down, v_forward)
    n_right = np.linalg.norm(v_right)
    if n_right > 1e-6: v_right = v_right / n_right
    else: v_right = np.array([1, 0, 0])

    # 4. Define Strict BEV Output Properties (PPM = 50)
    x_min, x_max = -(road_width / 2.0), (road_width / 2.0)
    PPM = 50.0 
    bev_W = int(road_width * PPM)
    bev_H = int((y_max - y_min) * PPM)
    
    dst_pts = np.array([
        [0, 0],            # top-left
        [bev_W, 0],        # top-right
        [0, bev_H],        # bottom-left
        [bev_W, bev_H]     # bottom-right
    ], dtype=np.float32)

    # 5. Project 3D Floor plane bounds to 2D
    bev_pts_3d = [
        (x_min * v_right) + (y_max * v_forward) + (cam_height_m * v_down), # top-left
        (x_max * v_right) + (y_max * v_forward) + (cam_height_m * v_down), # top-right
        (x_min * v_right) + (y_min * v_forward) + (cam_height_m * v_down), # bot-left
        (x_max * v_right) + (y_min * v_forward) + (cam_height_m * v_down)  # bot-right
    ]

    img_pts = []
    for pt in bev_pts_3d:
        p_img = K_rect @ pt
        if p_img[2] <= 1e-5: p_img[2] = 1e-5
        u, v = p_img[0]/p_img[2], p_img[1]/p_img[2]
        img_pts.append([u, v])
    
    img_pts = np.array(img_pts, dtype=np.float32)
    H_mat = cv2.getPerspectiveTransform(img_pts, dst_pts)
    
    return H_mat, bev_W, bev_H, PPM, v_down, v_forward, v_right

# --- REFACTORED TO USE NEW BASIS VECTORS ---
def draw_bev_grid(img, K_rect, cam_height_m, v_down, v_forward, v_right, y_min, y_max, x_range):
    # Draw horizontal lines (constant y/forward)
    for y_fwd in np.arange(math.floor(y_min), math.ceil(y_max) + 1, 1.0):
        pts = []
        for x in np.arange(-x_range, x_range + 0.5, 0.5):
            pt3d = (x * v_right) + (y_fwd * v_forward) + (cam_height_m * v_down)
            p_img = K_rect @ pt3d
            if p_img[2] > 0:
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
            if p_img[2] > 0:
                u, v = int(p_img[0]/p_img[2]), int(p_img[1]/p_img[2])
                pts.append((u, v))
        if len(pts) > 1:
            for i in range(len(pts)-1): cv2.line(img, pts[i], pts[i+1], (0, 255, 255), 2)
            
    return img