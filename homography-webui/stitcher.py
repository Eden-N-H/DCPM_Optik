import os
import cv2
import numpy as np
import math
from core_math import haversine_distance, calculate_bearing

def M_to_H(M):
    """Convert a 2x3 Affine matrix to a 3x3 Homography matrix."""
    H = np.eye(3, dtype=np.float64)
    H[:2, :] = M
    return H

def create_photogrammetry_map(frames_data, pure_output_path, overlay_output_path, progress_callback=None):
    if not frames_data: return None
    
    valid_frames = [f for f in frames_data if os.path.exists(f['path'])]
    total_frames = len(valid_frames)
    if total_frames == 0: return None

    base_lat = valid_frames[0]['lat']
    base_lon = valid_frames[0]['lon']
    R = 6378137.0

    first_img = cv2.imread(valid_frames[0]['path'])
    bev_h, bev_w = first_img.shape[:2]
    cw, ch = bev_w / 2.0, bev_h / 2.0
    gsd = valid_frames[0]['w_m'] / bev_w

    sift = cv2.SIFT_create(nfeatures=2000)
    bf = cv2.BFMatcher()

    H_matrices = []
    H_curr = np.eye(3, dtype=np.float64)
    H_matrices.append((valid_frames[0]['path'], H_curr))

    prev_gray = cv2.cvtColor(first_img, cv2.COLOR_BGR2GRAY)
    prev_kp, prev_des = sift.detectAndCompute(prev_gray, None)

    # --- PHASE 1: Build the Pure Ribbon (Local Coordinates) ---
    for i in range(1, total_frames):
        fd_prev = valid_frames[i-1]
        fd_curr = valid_frames[i]
        img_path = fd_curr['path']
        img = cv2.imread(img_path)
        
        # 1. Telemetry Fallback Guess (Relative to previous frame)
        d_m = haversine_distance(fd_prev['lat'], fd_prev['lon'], fd_curr['lat'], fd_curr['lon'])
        brng = calculate_bearing(fd_prev['lat'], fd_prev['lon'], fd_curr['lat'], fd_curr['lon'])
        
        rel_bearing = brng - fd_prev['heading']
        dy_m = d_m * math.cos(math.radians(rel_bearing))  # Forward
        dx_m = d_m * math.sin(math.radians(rel_bearing))  # Right
        
        dy_px = -dy_m / gsd  # BEV Up is -Y
        dx_px = dx_m / gsd
        
        d_heading = fd_curr['heading'] - fd_prev['heading']
        M_tele = cv2.getRotationMatrix2D((cw, ch), -d_heading, 1.0)
        M_tele[0, 2] += dx_px
        M_tele[1, 2] += dy_px
        H_step = M_to_H(M_tele)

        # 2. SIFT Optical Alignment
        if img is not None:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            kp, des = sift.detectAndCompute(gray, None)
            
            if prev_des is not None and des is not None:
                matches = bf.knnMatch(des, prev_des, k=2)
                good = [m for m, n in matches if m.distance < 0.75 * n.distance]
                
                if len(good) >= 10:
                    src_pts = np.float32([kp[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
                    dst_pts = np.float32([prev_kp[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
                    
                    M_affine, _ = cv2.estimateAffinePartial2D(src_pts, dst_pts, method=cv2.RANSAC, ransacReprojThreshold=3.0)
                    
                    if M_affine is not None:
                        # Strip out the hallucinatory SIFT Scale Factor (Lock to 1.0)
                        scale = math.hypot(M_affine[0, 0], M_affine[1, 0])
                        M_rigid = M_affine.copy()
                        M_rigid[0, 0] /= scale
                        M_rigid[0, 1] /= scale
                        M_rigid[1, 0] /= scale
                        M_rigid[1, 1] /= scale
                        
                        H_sift = M_to_H(M_rigid)
                        
                        # Clamping: Reject if SIFT tries to snap to an extreme physics-breaking drift
                        dist_drift = math.hypot(H_sift[0, 2] - H_step[0, 2], H_sift[1, 2] - H_step[1, 2])
                        angle_sift = math.degrees(math.atan2(H_sift[1, 0], H_sift[0, 0]))
                        angle_tele = math.degrees(math.atan2(H_step[1, 0], H_step[0, 0]))
                        angle_diff = abs(angle_sift - angle_tele) % 360
                        angle_diff = min(angle_diff, 360 - angle_diff)
                        
                        if dist_drift < (bev_h * 0.5) and angle_diff < 5.0:
                            H_step = H_sift

            prev_kp, prev_des = kp, des

        H_curr = H_curr @ H_step
        H_matrices.append((img_path, H_curr))
        
        if progress_callback: 
            progress_callback(int((i / total_frames) * 33), 100, f"Optically stitching frame {i} of {total_frames}...")

    # Rendering Core: Optimized with Local ROI Bounding Boxes
    def render_canvas(matrices, output_path, progress_range=(0, 100), step_name=""):
        global_min_x, global_min_y = float('inf'), float('inf')
        global_max_x, global_max_y = float('-inf'), float('-inf')
        pts = np.array([[0,0], [bev_w,0], [bev_w,bev_h], [0,bev_h]], dtype=np.float32).reshape(-1,1,2)
        
        # Calculate overall canvas bounds and local frame bounds
        frame_bboxes = []
        for p, H in matrices:
            transformed = cv2.transform(pts, H[:2, :])
            xs, ys = transformed[:,0,0], transformed[:,0,1]
            min_x, min_y = int(np.floor(np.min(xs))), int(np.floor(np.min(ys)))
            max_x, max_y = int(np.ceil(np.max(xs))), int(np.ceil(np.max(ys)))
            
            global_min_x, global_min_y = min(global_min_x, min_x), min(global_min_y, min_y)
            global_max_x, global_max_y = max(global_max_x, max_x), max(global_max_y, max_y)
            
            frame_bboxes.append((p, H, min_x, min_y, max_x, max_y))

        canvas_w = global_max_x - global_min_x
        canvas_h = global_max_y - global_min_y

        try:
            # Memory mapping approach isn't necessary with local bounding boxes unless the drive is dozens of miles.
            final_map = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
            depth_buffer = np.zeros((canvas_h, canvas_w), dtype=np.float32)
            alpha_mask = np.zeros((canvas_h, canvas_w), dtype=np.uint8)
        except MemoryError:
            raise MemoryError(f"Memory crashed allocating {canvas_w}x{canvas_h} canvas. Drive is too long to fit in physical RAM.")

        depth_base = np.repeat(np.linspace(0.01, 1.0, bev_h).reshape(bev_h, 1), bev_w, axis=1).astype(np.float32)
        start_pct, end_pct = progress_range
        total_m = len(frame_bboxes)

        for i, (path, H, min_x, min_y, max_x, max_y) in enumerate(frame_bboxes):
            img = cv2.imread(path)
            if img is None: continue
            
            # Local bounds for isolated fast rendering
            local_w, local_h = max_x - min_x, max_y - min_y
            
            M_local = H[:2, :].copy()
            M_local[0, 2] -= min_x
            M_local[1, 2] -= min_y
            
            # Warp the image ONLY to the size of the small local bounding box (100x efficiency gain)
            warped_img = cv2.warpAffine(img, M_local, (local_w, local_h), flags=cv2.INTER_LINEAR)
            warped_depth = cv2.warpAffine(depth_base, M_local, (local_w, local_h), flags=cv2.INTER_LINEAR)
            
            # Where this small local box lands on the giant canvas array
            gx, gy = min_x - global_min_x, min_y - global_min_y
            
            # Slicing logic
            canvas_depth_slice = depth_buffer[gy:gy+local_h, gx:gx+local_w]
            winning_mask = warped_depth > canvas_depth_slice
            
            # Apply only the winning pixels
            canvas_depth_slice[winning_mask] = warped_depth[winning_mask]
            final_map[gy:gy+local_h, gx:gx+local_w][winning_mask] = warped_img[winning_mask]
            alpha_mask[gy:gy+local_h, gx:gx+local_w][winning_mask] = 255

            if progress_callback:
                cur_pct = start_pct + int(((i + 1) / total_m) * (end_pct - start_pct))
                progress_callback(cur_pct, 100, f"{step_name} {i+1}/{total_m}...")

        final_bgra = cv2.cvtColor(final_map, cv2.COLOR_BGR2BGRA)
        final_bgra[:, :, 3] = alpha_mask
        cv2.imwrite(output_path, final_bgra)
        
        # Adjust matrices to global coordinate anchor
        shifted_matrices = []
        for path, H, _, _, _, _ in frame_bboxes:
            H_shifted = H.copy()
            H_shifted[0, 2] -= global_min_x
            H_shifted[1, 2] -= global_min_y
            shifted_matrices.append((path, H_shifted))

        return global_min_x, global_min_y, shifted_matrices, canvas_w, canvas_h

    # --- PHASE 2: Render Pure Ribbon ---
    pure_min_x, pure_min_y, pure_matrices, _, _ = render_canvas(
        H_matrices, pure_output_path, progress_range=(33, 65), step_name="Rendering Pure Ribbon"
    )

    # --- PHASE 3: Calculate Map Overlay Projection ---
    if progress_callback: progress_callback(66, 100, "Calculating Global Map Rotation...")
    
    # Anchor rotations based on absolute first and last frames
    c0 = pure_matrices[0][1] @ np.array([cw, ch, 1.0])
    cN = pure_matrices[-1][1] @ np.array([cw, ch, 1.0])
    dx_pure, dy_pure = cN[0] - c0[0], cN[1] - c0[1]

    lat_N, lon_N = valid_frames[-1]['lat'], valid_frames[-1]['lon']
    XN = math.radians(lon_N - base_lon) * R * math.cos(math.radians(base_lat))
    YN = -math.radians(lat_N - base_lat) * R
    
    if math.hypot(XN, YN) < 1.0 or total_frames < 2:
        theta = math.radians(valid_frames[0]['heading'])
    else:
        angle_map = math.atan2(YN, XN)
        angle_pure = math.atan2(dy_pure, dx_pure)
        theta = angle_map - angle_pure

    M_rot = cv2.getRotationMatrix2D((c0[0], c0[1]), math.degrees(theta), 1.0)
    H_global_align = M_to_H(M_rot)

    overlay_matrices = []
    for path, H_pure in pure_matrices:
        overlay_matrices.append((path, H_global_align @ H_pure))

    overlay_min_x, overlay_min_y, _, final_w, final_h = render_canvas(
        overlay_matrices, overlay_output_path, progress_range=(67, 100), step_name="Rendering Map Overlay"
    )

    # Convert Bounding Box to Geo-coordinates
    c0_overlay = H_global_align @ np.array([c0[0], c0[1], 1.0])
    pixel_dx_from_left = c0_overlay[0] - overlay_min_x
    pixel_dy_from_top = c0_overlay[1] - overlay_min_y
    
    offset_x_m = -pixel_dx_from_left * gsd
    offset_y_m = -pixel_dy_from_top * gsd
    
    nw_lat = base_lat + math.degrees(-offset_y_m / R)
    nw_lon = base_lon + math.degrees(offset_x_m / (R * math.cos(math.radians(base_lat))))
    
    se_lat = base_lat + math.degrees(-(offset_y_m + (final_h * gsd)) / R)
    se_lon = base_lon + math.degrees((offset_x_m + (final_w * gsd)) / (R * math.cos(math.radians(base_lat))))

    return [[se_lat, nw_lon], [nw_lat, se_lon]]