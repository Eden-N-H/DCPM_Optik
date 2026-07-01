import cv2
import math
import numpy as np

def equirectangular_to_rectilinear(equi_img, fov_deg, pitch_deg, roll_deg, yaw_deg, output_width=1280, output_height=720):
    h, w = equi_img.shape[:2]
    f = (output_width / 2.0) / math.tan(math.radians(fov_deg) / 2.0)
    K = np.array([[f, 0, output_width / 2.0], [0, f, output_height / 2.0], [0, 0, 1]], dtype=np.float32)
    pitch, yaw, roll = math.radians(pitch_deg), math.radians(yaw_deg), math.radians(roll_deg)
    R_pitch = np.array([[1, 0, 0], [0, math.cos(pitch), -math.sin(pitch)], [0, math.sin(pitch), math.cos(pitch)]])
    R_yaw = np.array([[math.cos(yaw), 0, math.sin(yaw)], [0, 1, 0], [-math.sin(yaw), 0, math.cos(yaw)]])
    R_roll = np.array([[math.cos(roll), -math.sin(roll), 0], [math.sin(roll), math.cos(roll), 0], [0, 0, 1]])
    R_combined = R_yaw @ R_pitch @ R_roll 
    R_inv = np.linalg.inv(R_combined)
    x, y = np.meshgrid(np.arange(output_width), np.arange(output_height))
    pixels = np.stack((x, y, np.ones_like(x)), axis=-1).reshape(-1, 3).T
    rays = R_inv @ (np.linalg.inv(K) @ pixels)
    theta = np.arctan2(rays[0, :], rays[2, :])
    phi = np.arcsin(np.clip(rays[1, :] / np.linalg.norm(rays, axis=0), -1, 1))
    u, v = (theta / (2 * math.pi) + 0.5) * w, (phi / math.pi + 0.5) * h
    map_x, map_y = u.reshape((output_height, output_width)).astype(np.float32), v.reshape((output_height, output_width)).astype(np.float32)
    return cv2.remap(equi_img, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP), K