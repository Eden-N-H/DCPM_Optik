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

def get_fisheye_maps(W, H, x_fov_deg, y_fov_deg, R=None):
    """
    Build undistortion remap tables for a fisheye (equidistant) lens.
    
    Parameters
    ----------
    R : np.ndarray (3x3), optional
        Rectification rotation applied during undistortion. Use this to
        level the output image (compensate camera roll/pitch) so that the
        horizon is horizontal in pixel space. If None, no rotation is applied.
    """
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
    R_mat = R if R is not None else np.eye(3)
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(K_fish, D, R_mat, K_rect, (W, H), cv2.CV_32FC1)
    return map1, map2, K_rect

def get_bev_from_fisheye(raw_img, cam_height_m, yaw_offset, y_min, y_max, road_width, x_fov_deg, y_fov_deg):
    """
    Generate BEV (Bird's Eye View) directly from a raw fisheye image without
    intermediate undistortion. This produces geometrically perfect orthographic
    output because each BEV pixel is individually mapped through the exact
    fisheye projection model to find its source pixel.
    
    This eliminates the two-step error (undistort with imperfect model → warp
    with K_rect that doesn't match) that caused corridor stitching artifacts.
    
    Returns: bev_bgr, bev_W, bev_H, PPM
    """
    h, w = raw_img.shape[:2]
    x_fov_rad = math.radians(x_fov_deg)
    y_fov_rad = math.radians(y_fov_deg)
    
    # Fisheye (equidistant) focal lengths
    fx_fish = (w / 2.0) / (x_fov_rad / 2.0)
    fy_fish = (h / 2.0) / (y_fov_rad / 2.0)
    cx, cy = w / 2.0, h / 2.0
    
    # BEV canvas parameters
    PPM = 50.0
    bev_W = int(road_width * PPM)
    bev_H = int((y_max - y_min) * PPM)
    
    # Apply yaw rotation (camera mounting angle offset)
    y_rad = math.radians(yaw_offset)
    cos_y, sin_y = math.cos(y_rad), math.sin(y_rad)
    
    # For each BEV pixel (u_bev, v_bev), compute the corresponding ground point,
    # then project through camera model to find source pixel in raw fisheye.
    # BEV coordinate system:
    #   u_bev: 0=left edge (-road_width/2), bev_W=right edge (+road_width/2)
    #   v_bev: 0=far (y_max), bev_H=near (y_min)
    
    u_bev = np.arange(bev_W, dtype=np.float64)
    v_bev = np.arange(bev_H, dtype=np.float64)
    uu, vv = np.meshgrid(u_bev, v_bev)
    
    # Ground coordinates (in road plane, relative to camera)
    # x_ground: lateral (right is positive)
    # z_ground: forward distance from camera
    x_ground = (uu / PPM) - (road_width / 2.0)
    z_ground = y_max - (vv / PPM)
    
    # Apply yaw rotation (rotate ground coordinates by yaw offset)
    x_rot = cos_y * x_ground + sin_y * z_ground
    z_rot = -sin_y * x_ground + cos_y * z_ground
    
    # 3D point in camera frame: ground is at Y = cam_height_m below camera
    # Camera: X=right, Y=down, Z=forward
    x_cam = x_rot
    y_cam = np.full_like(x_rot, cam_height_m)  # ground is cam_height below
    z_cam = z_rot
    
    # Project through equidistant fisheye model: r = f * theta
    r_3d = np.sqrt(x_cam**2 + y_cam**2 + z_cam**2)
    theta = np.arccos(np.clip(z_cam / r_3d, -1.0, 1.0))
    
    # Radial direction in XY plane
    r_xy = np.sqrt(x_cam**2 + y_cam**2)
    # Avoid division by zero at optical axis
    r_xy_safe = np.maximum(r_xy, 1e-9)
    
    dir_x = x_cam / r_xy_safe
    dir_y = y_cam / r_xy_safe
    
    # Fisheye pixel coordinates
    map_x = (cx + fx_fish * theta * dir_x).astype(np.float32)
    map_y = (cy + fy_fish * theta * dir_y).astype(np.float32)
    
    # Mask out points behind camera or outside image bounds
    valid = (z_cam > 0.1) & (map_x >= 0) & (map_x < w) & (map_y >= 0) & (map_y < h)
    
    # Set invalid pixels to map to (0,0) — they'll be black
    map_x[~valid] = 0
    map_y[~valid] = 0
    
    bev_bgr = cv2.remap(raw_img, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    
    # Zero out invalid pixels
    bev_bgr[~valid] = 0
    
    # Erode the valid mask slightly to remove edge artifacts (interpolation
    # at the boundary of valid/invalid produces dark fringe pixels that show
    # up as visible seams in corridor stitching)
    valid_mask = valid.astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    valid_mask = cv2.erode(valid_mask, kernel)
    bev_bgr[valid_mask == 0] = 0
    
    return bev_bgr, bev_W, bev_H, PPM


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
    # 1. Road-Surface-Relative Ground Plane (Pitch-Only from GRAV)
    #
    #    For a vehicle-mounted camera doing road inspection:
    #    - Roll is corrected during undistortion (image horizon is leveled)
    #    - Pitch (camera tilt toward/away from road) must be accounted for
    #      so the grid projects at the correct perspective angle
    #    - The "ground plane" is the camera's local road surface
    #
    #    After roll correction in undistortion, the effective gravity in the
    #    leveled image frame has gx=0 (no roll component). The remaining
    #    pitch is visible as gz != 0 (gravity has a forward/backward lean).
    #
    #    We compute v_down from pitch only (roll already removed from image),
    #    which ensures the grid perspective matches what the camera sees.
    
    g = np.array(grav_vec, dtype=np.float64)
    n = np.linalg.norm(g)
    if n < 1e-6:
        g_norm = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    else:
        g_norm = g / n
    
    # After roll-correction in undistortion, the effective gravity vector
    # in the leveled frame has gx≈0. The pitch component remains in gy, gz.
    # For cameras where undistortion is not applied (or roll correction is
    # skipped), we still zero out gx to prevent roll from tilting the grid.
    gy = g_norm[1]
    gz = g_norm[2]
    
    # Reconstruct v_down in the YZ plane only (gx forced to 0 = no roll)
    yz_norm = math.sqrt(gy*gy + gz*gz)
    if yz_norm > 1e-6:
        v_down_base = np.array([0.0, gy / yz_norm, gz / yz_norm], dtype=np.float64)
    else:
        v_down_base = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    
    # 2. Forward = Z-axis projected onto the plane perpendicular to v_down
    z_cam = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    v_fwd_base = z_cam - np.dot(z_cam, v_down_base) * v_down_base
    n_fwd = np.linalg.norm(v_fwd_base)
    
    if n_fwd > 1e-6:
        v_fwd_base /= n_fwd
    else:
        y_cam = np.array([0.0, -1.0, 0.0], dtype=np.float64)
        v_fwd_base = y_cam - np.dot(y_cam, v_down_base) * v_down_base
        v_fwd_base /= np.linalg.norm(v_fwd_base)
            
    # 3. Right is orthogonal to Down and Forward (always [1,0,0] since no roll)
    v_right_base = np.cross(v_down_base, v_fwd_base)
    v_right_base /= np.linalg.norm(v_right_base)
        
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

def _project_world_grid_line(K_rect, cam_height_m, v_down, v_forward, v_right,
                              world_points, cam_offset_x, cam_offset_z, cos_d, sin_d,
                              y_min, y_max, x_range, fisheye_params=None):
    """
    Projects a sequence of world-frame (x, z) sample points for a single grid
    line into image pixels, returning a list the same length as
    `world_points` where each entry is either an (u, v) pixel or None if
    that particular sample falls outside the currently visible local
    frustum (or behind the camera).

    If fisheye_params is provided (a dict with 'fx', 'fy', 'cx', 'cy'),
    points are projected through the equidistant fisheye model (r = f*theta)
    onto the raw distorted image rather than through the rectilinear K_rect.
    This ensures the grid matches the actual fisheye image perfectly.
    """
    projected = []
    for world_x, world_z in world_points:
        dx = world_x - cam_offset_x
        dz = world_z - cam_offset_z
        local_x = dx * cos_d - dz * sin_d
        local_z = dx * sin_d + dz * cos_d

        if (local_z < y_min - 0.5 or local_z > y_max + 0.5
                or local_x < -x_range - 0.5 or local_x > x_range + 0.5):
            projected.append(None)
            continue

        pt3d = (local_x * v_right) + (local_z * v_forward) + (cam_height_m * v_down)
        
        if fisheye_params is not None:
            # Equidistant fisheye projection: r = f * theta
            # theta = angle from optical axis (Z)
            x3, y3, z3 = pt3d[0], pt3d[1], pt3d[2]
            
            # Must be in front of camera
            if z3 <= 0.1:
                projected.append(None)
                continue
            
            # Angle from optical axis
            r_3d = math.sqrt(x3*x3 + y3*y3 + z3*z3)
            theta = math.acos(np.clip(z3 / r_3d, -1.0, 1.0))
            
            # Direction in image plane (XY components give the radial direction)
            r_xy = math.sqrt(x3*x3 + y3*y3)
            if r_xy < 1e-9:
                # Point is on the optical axis
                px = fisheye_params['cx']
                py = fisheye_params['cy']
            else:
                # Equidistant: image radius = f * theta
                r_img_x = fisheye_params['fx'] * theta
                r_img_y = fisheye_params['fy'] * theta
                
                # Project to pixel using direction in XY plane
                dir_x = x3 / r_xy
                dir_y = y3 / r_xy
                
                px = fisheye_params['cx'] + r_img_x * dir_x
                py = fisheye_params['cy'] + r_img_y * dir_y
            
            if abs(px) < 16000 and abs(py) < 16000:
                projected.append((int(px), int(py)))
            else:
                projected.append(None)
        else:
            # Standard rectilinear (pinhole) projection
            p_img = K_rect @ pt3d
            
            if p_img[2] > 0.5:
                px, py = p_img[0] / p_img[2], p_img[1] / p_img[2]
                
                if abs(px) < 16000 and abs(py) < 16000:
                    projected.append((int(px), int(py)))
                else:
                    projected.append(None)
            else:
                projected.append(None)

    return projected


def _draw_polyline_segments(img, projected_points, color=(0, 255, 255), thickness=2):
    """Draws cv2.line only between consecutive (adjacent-index) valid points."""
    for i in range(len(projected_points) - 1):
        a, b = projected_points[i], projected_points[i + 1]
        if a is not None and b is not None:
            cv2.line(img, a, b, color, thickness)


def draw_bev_grid(img, K_rect, cam_height_m, v_down, v_forward, v_right, y_min, y_max, x_range,
                   cam_offset_x=0.0, cam_offset_z=0.0, delta_heading=0.0, fisheye_params=None):
    """
    Draws a metric grid onto the rectilinear image.

    The grid is anchored to a single, fixed world frame -- the project's
    grid baseline -- rather than to the camera's own current heading, so
    integer-metre grid lines stay planted at the same physical ground
    locations as the camera moves and turns through the scene. As the
    vehicle turns a corner the grid will (correctly) appear to rotate in
    the image, because it is drawn on ground that is fixed in the world
    while the camera's viewing direction changes -- that rotation is
    expected. What this function guarantees is that the *positions* of
    the grid lines themselves don't warp, and -- critically -- that lines
    are never drawn between two points that are not adjacent samples of
    the same physical line (see _project_world_grid_line /
    _draw_polyline_segments above). Without that guarantee, a world-fixed
    line that sweeps in and out of the visible frustum (routine once
    delta_heading != 0) produces spurious long diagonals connecting
    unrelated visible fragments of the line -- the "starburst" artifact.

    cam_offset_x, cam_offset_z: camera position relative to the baseline,
      expressed in the baseline's OWN (world-fixed) heading frame -- i.e.
      NOT pre-rotated into this view's local frame. This is the single
      coordinate system every frame's grid-line integers are chosen in.
    delta_heading: rotation, in radians, from the baseline frame to this
      view's local (v_right/v_forward) frame -- i.e.
      radians(view_heading - base_heading). Used to rotate each
      world-frame grid vertex into local ground coordinates immediately
      before projecting it through K_rect.

    With the defaults (0, 0, 0) the grid is purely camera-relative,
    identical to the original (pre-baseline) behaviour.
    """
    cos_d, sin_d = math.cos(delta_heading), math.sin(delta_heading)

    def local_to_world(local_x, local_z):
        # Inverse rotation (transpose of the 2D rotation matrix used in
        # _project_world_grid_line), used only to find which world-frame
        # integer grid lines are potentially visible from the current
        # local viewing rectangle.
        world_x = cam_offset_x + (local_x * cos_d + local_z * sin_d)
        world_z = cam_offset_z + (-local_x * sin_d + local_z * cos_d)
        return world_x, world_z

    # Determine the visible world-frame bounding box by transforming the
    # four corners of the current local viewing frustum's ground rectangle
    # into world coordinates. Rotation can turn an axis-aligned local
    # rectangle into a tilted one in world space, so we take the bounding
    # box of the four transformed corners (a safe superset -- individual
    # out-of-frustum samples are filtered per-point in
    # _project_world_grid_line regardless).
    corners_local = [(-x_range, y_min), (x_range, y_min), (x_range, y_max), (-x_range, y_max)]
    world_xs, world_zs = [], []
    for lx, lz in corners_local:
        wx, wz = local_to_world(lx, lz)
        world_xs.append(wx)
        world_zs.append(wz)

    world_x_min, world_x_max = min(world_xs), max(world_xs)
    world_z_min, world_z_max = min(world_zs), max(world_zs)

    # Cap how much world-space extent we ever sample across. A very large
    # delta_heading combined with a long z-range can otherwise blow this
    # bounding box out to an enormous area (slow, and pointless -- almost
    # none of it is visible). This is a safety net, not a behavioural
    # change for the normal case.
    MAX_EXTENT_M = 200.0
    world_x_min = max(world_x_min, world_x_min if (world_x_max - world_x_min) <= MAX_EXTENT_M else world_x_max - MAX_EXTENT_M)
    world_z_min = max(world_z_min, world_z_min if (world_z_max - world_z_min) <= MAX_EXTENT_M else world_z_max - MAX_EXTENT_M)

    # Horizontal lines (constant world-z / forward distance)
    for world_z in np.arange(math.floor(world_z_min), math.ceil(world_z_max) + 1, 1.0):
        world_points = [(wx, world_z) for wx in np.arange(math.floor(world_x_min), math.ceil(world_x_max) + 0.5, 0.5)]
        projected = _project_world_grid_line(K_rect, cam_height_m, v_down, v_forward, v_right,
                                              world_points, cam_offset_x, cam_offset_z, cos_d, sin_d,
                                              y_min, y_max, x_range, fisheye_params)
        _draw_polyline_segments(img, projected)

    # Vertical lines (constant world-x / lateral position)
    for world_x in np.arange(math.floor(world_x_min), math.ceil(world_x_max) + 1, 1.0):
        world_points = [(world_x, wz) for wz in np.arange(math.floor(world_z_min), math.ceil(world_z_max) + 0.5, 0.5)]
        projected = _project_world_grid_line(K_rect, cam_height_m, v_down, v_forward, v_right,
                                              world_points, cam_offset_x, cam_offset_z, cos_d, sin_d,
                                              y_min, y_max, x_range, fisheye_params)
        _draw_polyline_segments(img, projected)

    return img
