import cv2
import math
import numpy as np
from cv_bev import get_camera_to_world_rotation

def equirectangular_to_rectilinear(equi_img, fov_deg, pitch_deg, roll_deg, yaw_deg, output_width=1280, output_height=720):
    h, w = equi_img.shape[:2]
    
    # Determine precise rectilinear focal length required
    f = (output_width / 2.0) / math.tan(math.radians(fov_deg) / 2.0)
    K = np.array([
        [f, 0, output_width / 2.0], 
        [0, f, output_height / 2.0], 
        [0, 0, 1]
    ], dtype=np.float64)
    
    # Retrieve robust rotation matrix based on unified standard
    R_c2w = get_camera_to_world_rotation(pitch_deg, yaw_deg, roll_deg)
    
    x, y = np.meshgrid(np.arange(output_width), np.arange(output_height))
    
    # Analytically apply exact inverse intrinsics instead of brute-force matrix multiplication
    ray_x = (x.flatten() - (output_width / 2.0)) / f
    ray_y = (y.flatten() - (output_height / 2.0)) / f
    ray_z = np.ones_like(ray_x)
    rays_cam = np.vstack((ray_x, ray_y, ray_z))
    
    # Rotate viewing rays into the world frame mapped by the equirectangular projection
    rays = R_c2w @ rays_cam
    
    # Translate world rays into spherical angles
    theta = np.arctan2(rays[0, :], rays[2, :])
    phi = np.arcsin(np.clip(rays[1, :] / np.linalg.norm(rays, axis=0), -1.0, 1.0))
    
    # Project spherical coordinates down into the original equirectangular pixel map
    u = (theta / (2 * math.pi) + 0.5) * w
    v = (phi / math.pi + 0.5) * h
    
    map_x = u.reshape((output_height, output_width)).astype(np.float32)
    map_y = v.reshape((output_height, output_width)).astype(np.float32)
    
    return cv2.remap(equi_img, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP), K