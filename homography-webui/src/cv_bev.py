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

def _project_world_grid_line(K_rect, cam_height_m, v_down, v_forward, v_right,
                              world_points, cam_offset_x, cam_offset_z, cos_d, sin_d,
                              y_min, y_max, x_range):
    """
    Projects a sequence of world-frame (x, z) sample points for a single grid
    line into image pixels, returning a list the same length as
    `world_points` where each entry is either an (u, v) pixel or None if
    that particular sample falls outside the currently visible local
    frustum (or behind the camera).

    Returning a per-sample list (instead of silently dropping invalid
    samples) is the key fix: it lets the caller only ever draw a segment
    between two ADJACENT samples that are both valid, and never between
    two arbitrary valid samples that happen to still be in the list after
    invalid ones were removed. Connecting non-adjacent samples is exactly
    what produced the spurious long diagonal lines crossing the whole
    image whenever a world-fixed grid line swept in and out of view
    (which happens routinely once the camera's heading differs from the
    baseline, e.g. mid-corner) -- the grid geometry itself was correct,
    but the rendering was joining points that were never meant to be
    connected.
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
        p_img = K_rect @ pt3d
        
        # REQUIREMENT: Depth along optical axis (p_img[2]) must be reasonably in front of the lens.
        # If it approaches 0 (the focal plane), the pixel division explodes towards infinity.
        # Previously this was "fixed" by clamping extreme pixel values, but dragging the endpoint 
        # of an exploded 2D vector back toward the principal point radically alters the line's slope,
        # forcing parallel grid lines to physically converge into a drawn "starburst".
        # Safely discarding the point instead (assigning None) naturally breaks the drawn segment.
        if p_img[2] > 0.5:
            px, py = p_img[0] / p_img[2], p_img[1] / p_img[2]
            
            # Rely on strict coordinate bounds check rather than clamping to avoid OpenCV 16-bit 
            # integer overflow while preserving true projective geometry.
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
                   cam_offset_x=0.0, cam_offset_z=0.0, delta_heading=0.0):
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
                                              y_min, y_max, x_range)
        _draw_polyline_segments(img, projected)

    # Vertical lines (constant world-x / lateral position)
    for world_x in np.arange(math.floor(world_x_min), math.ceil(world_x_max) + 1, 1.0):
        world_points = [(world_x, wz) for wz in np.arange(math.floor(world_z_min), math.ceil(world_z_max) + 0.5, 0.5)]
        projected = _project_world_grid_line(K_rect, cam_height_m, v_down, v_forward, v_right,
                                              world_points, cam_offset_x, cam_offset_z, cos_d, sin_d,
                                              y_min, y_max, x_range)
        _draw_polyline_segments(img, projected)

    return img
