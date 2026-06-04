from flask import Flask, render_template, request, jsonify, send_file
import cv2
import numpy as np
import base64
import piexif
import piexif.helper
import json
import io
import math

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024 # 50MB max upload size

# ==========================================
# 1. HELPER / MANUAL HOMOGRAPHY FUNCTIONS
# ==========================================
def order_points(pts):
    """Sorts 4 points into Top-Left, Top-Right, Bottom-Right, Bottom-Left."""
    rect = np.zeros((4, 2), dtype="float32")
    pts = sorted(pts, key=lambda x: x[1])
    top_pts = sorted(pts[:2], key=lambda x: x[0])
    bottom_pts = sorted(pts[2:], key=lambda x: x[0])
    rect[0] = top_pts[0]
    rect[1] = top_pts[1]
    rect[2] = bottom_pts[1]
    rect[3] = bottom_pts[0]
    return rect

def get_full_warp_manual(img, pts, out_w, out_h):
    """Original Manual 4-Point Homography Warp"""
    src_pts = order_points(pts)
    dst_pts = np.array([[0, 0], [out_w, 0], [out_w, out_h], [0, out_h]], dtype="float32")
    H, _ = cv2.findHomography(src_pts, dst_pts)
    h, w = img.shape[:2]
    
    corners_hom = np.array([[0,0,1], [w,0,1], [w,h,1], [0,h,1]]).T
    trans_hom = H @ corners_hom
    valid_x, valid_y = [0, out_w], [0, out_h]
    
    for i in range(4):
        z = trans_hom[2, i]
        if z > 0.01:
            valid_x.append(trans_hom[0, i] / z)
            valid_y.append(trans_hom[1, i] / z)
            
    x_min, x_max = min(valid_x), max(valid_x)
    y_min, y_max = min(valid_y), max(valid_y)
    
    MAX_DIM = 6000
    if x_max - x_min > MAX_DIM:
        center_x = out_w / 2
        x_min, x_max = center_x - MAX_DIM/2, center_x + MAX_DIM/2
    if y_max - y_min > MAX_DIM:
        center_y = out_h / 2
        y_min, y_max = center_y - MAX_DIM/2, center_y + MAX_DIM/2

    T = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]], dtype=np.float32)
    H_full = T @ H
    final_w, final_h = int(x_max - x_min), int(y_max - y_min)
    return cv2.warpPerspective(img, H_full, (final_w, final_h))

# ==========================================
# 2. HARDCODED TELEMETRY MAPPING
# ==========================================
def get_telemetry_warp(img, cam_h_mm, pitch_deg, fov_deg, mm_px, out_w, out_h):
    """Automated Inverse Perspective Mapping based on Telemetry"""
    H_img, W_img = img.shape[:2]
    
    # Calculate focal length in pixels from Field of View
    f = W_img / (2 * math.tan(math.radians(fov_deg) / 2))
    pitch_rad = math.radians(pitch_deg)

    def pixel_to_world(u, v):
        """Maps an image pixel to real-world metric coordinates on the ground plane"""
        cx, cy = W_img / 2, H_img / 2
        ray_c = np.array([u - cx, v - cy, f], dtype=float)
        ray_c /= np.linalg.norm(ray_c)
        
        # Rotate ray based on camera pitch (tilt down = rotate around X axis)
        Rx = np.array([
            [1, 0, 0],
            [0, math.cos(-pitch_rad), -math.sin(-pitch_rad)],
            [0, math.sin(-pitch_rad), math.cos(-pitch_rad)]
        ])
        ray_w = Rx @ ray_c
        
        if ray_w[1] <= 0: return None # Ray is pointing at the sky (above horizon)
        
        scale = cam_h_mm / ray_w[1]
        return ray_w[0] * scale, ray_w[2] * scale # Returns (X right, Z forward)

    # 1. Dynamically find the horizon line to pick safe source points
    horizon_y = H_img/2 - f * math.tan(pitch_rad)
    safe_y1 = max(horizon_y + (H_img - horizon_y) * 0.2, 0)
    safe_y2 = H_img * 0.9

    pts_img = [
        (W_img * 0.2, safe_y1), (W_img * 0.8, safe_y1),
        (W_img * 0.2, safe_y2), (W_img * 0.8, safe_y2)
    ]

    src_pts, world_pts = [], []
    for u, v in pts_img:
        w_coord = pixel_to_world(u, v)
        if w_coord:
            src_pts.append([u, v])
            world_pts.append(w_coord)

    # Calculate offset so the bottom of the image sits at the bottom of the canvas
    closest_w = pixel_to_world(W_img/2, H_img)
    Z_min = closest_w[1] if closest_w else 0

    # 2. Map calculated metric real-world points into the final destination canvas
    dst_pts = []
    for X, Z in world_pts:
        dst_x = out_w / 2 + (X / mm_px)
        dst_y = out_h - ((Z - Z_min) / mm_px)
        dst_pts.append([dst_x, dst_y])

    M = cv2.getPerspectiveTransform(np.float32(src_pts), np.float32(dst_pts))
    return cv2.warpPerspective(img, M, (out_w, out_h))

# ==========================================
# 3. AI / AUTO VANISHING POINT MAPPING
# ==========================================
def prepare_edges(img, roi_top_frac=0.35, gaussian_k=7, morph_k=3, canny_lo=40, canny_hi=110):
    h, w = img.shape[:2]
    roi_y = int(h * roi_top_frac)
    roi = img[roi_y:]
    gk = gaussian_k | 1
    
    if len(img.shape) == 3 and img.shape[2] == 3:
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        sat = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)[:, :, 1]
        edges_sat = cv2.Canny(cv2.GaussianBlur(sat, (gk, gk), 0), canny_lo // 2, canny_hi // 2)
    else:
        gray = roi
        edges_sat = np.zeros_like(gray)
        
    edges_lum = cv2.Canny(cv2.GaussianBlur(gray, (gk, gk), 0), canny_lo, canny_hi)
    edges = cv2.bitwise_or(edges_lum, edges_sat)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (morph_k, morph_k))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    return edges, roi_y

def detect_lines(edges, roi_y, img_shape, angle_lo=30, angle_hi=85):
    h, w = img_shape[:2]
    min_len = max(60, int(w * 0.07))
    raw = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=min_len, maxLineGap=80)
    if raw is None:
        return np.empty((0, 4), dtype=int)

    def _filter(lines_raw, a_lo, a_hi):
        out = []
        half_w = w / 2.0
        for x1, y1, x2, y2 in lines_raw.reshape(-1, 4):
            y1g, y2g = y1 + roi_y, y2 + roi_y
            dx, dy = float(x2 - x1), float(y2g - y1g)
            angle = 90.0 if abs(dx) < 1e-3 else abs(np.degrees(np.arctan2(abs(dy), abs(dx))))
            if not (a_lo <= angle <= a_hi): continue
            if abs(dx) > 1e-3:
                intercept_y = y1g + (dy / dx) * (half_w - x1)
                if not (h * 0.15 <= intercept_y <= h * 0.45): continue
            out.append([x1, y1g, x2, y2g])
        return out

    out = _filter(raw, angle_lo, angle_hi)
    if len(out) < 8:
        out = _filter(raw, 15, angle_hi)
    return np.array(out, dtype=int) if out else np.empty((0, 4), dtype=int)

def _line_to_h(x1, y1, x2, y2):
    return np.cross([float(x1), float(y1), 1.0], [float(x2), float(y2), 1.0])

def _intersect(l1, l2):
    pt = np.cross(l1, l2)
    if abs(pt[2]) < 1e-10: return None
    return pt[:2] / pt[2]

def ransac_vp(lines, h, w, n_iter=3000, inlier_thresh=15.0, min_inliers=4):
    if len(lines) < 2:
        return np.array([w / 2.0, h * 0.35]), np.zeros(len(lines), dtype=bool)

    hl = [_line_to_h(*l) for l in lines]
    lengths = np.array([np.hypot(float(x2 - x1), float(y2 - y1)) for x1, y1, x2, y2 in lines])
    lengths /= lengths.max()
    sides = np.array([(lines[k][0] + lines[k][2]) / 2.0 >= w / 2.0 for k in range(len(lines))], dtype=int)

    best, best_n, best_inliers = None, 0.0, []
    half_w = w / 2.0

    for _ in range(n_iter):
        i, j = np.random.choice(len(hl), 2, replace=False)
        vp = _intersect(hl[i], hl[j])
        if vp is None or vp[1] > h * 0.50 or abs(vp[0] - half_w) > half_w * 2:
            continue

        inlier_idx = []
        for k, l in enumerate(hl):
            if abs(l[0]*vp[0] + l[1]*vp[1] + l[2]) / (np.hypot(l[0], l[1]) + 1e-9) >= inlier_thresh:
                continue
            x1, y1, x2, y2 = lines[k]
            mx, my = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            tx, ty = vp[0] - mx, vp[1] - my
            dx, dy = float(x2 - x1), float(y2 - y1)
            cos_a = abs(tx*dx + ty*dy) / ((np.hypot(tx, ty) + 1e-9) * (np.hypot(dx, dy) + 1e-9))
            if cos_a < 0.7: continue
            inlier_idx.append(k)

        if len(inlier_idx) < min_inliers: continue
        inlier_sides = sides[inlier_idx]
        if not (np.any(inlier_sides == 0) and np.any(inlier_sides == 1)): continue

        weighted_n = sum(lengths[k] for k in inlier_idx)
        weighted_n *= 1.0 + max(0.0, (h * 0.50 - vp[1]) / (h * 0.50)) * 0.3
        if weighted_n > best_n:
            best_n, best, best_inliers = weighted_n, vp.copy(), inlier_idx

    if best is None:
        return np.array([w / 2.0, h * 0.35]), np.zeros(len(lines), dtype=bool)

    mask = np.zeros(len(lines), dtype=bool)
    mask[best_inliers] = True
    return best, mask

def road_trapezoid(vp, h, w, near_spread=0.38, far_spread=0.16):
    vx, vy = float(vp[0]), float(vp[1])
    near_y = h * 0.92
    far_y = max(h * 0.45, vy + h * 0.05)
    cx = w / 2.0
    near_hw = w * near_spread

    near_l = np.array([cx - near_hw, near_y])
    near_r = np.array([cx + near_hw, near_y])

    def _ray_at_y(corner, target_y):
        t = (target_y - vy) / (corner[1] - vy + 1e-9)
        return np.array([vx + t * (corner[0] - vx), target_y])

    far_l = _ray_at_y(near_l, far_y)
    far_r = _ray_at_y(near_r, far_y)

    return np.array([far_l, far_r, near_r, near_l], dtype=np.float32)

def get_autovp_warp(img, cam_h_mm, fov_deg, mm_px, out_w, out_h, manual_vp=None):
    """Detects VP (or uses manual one) and infers camera tilt/yaw"""
    h, w = img.shape[:2]
    
    if manual_vp:
        vp = np.array([manual_vp['x'], manual_vp['y']])
        lines, mask = [], []
    else:
        edges, roi_y = prepare_edges(img)
        lines = detect_lines(edges, roi_y, img.shape)
        vp, mask = ransac_vp(lines, h, w)
    
    src_pts = road_trapezoid(vp, h, w)
    
    f = w / (2 * math.tan(math.radians(fov_deg) / 2))
    cx, cy = w / 2, h / 2
    
    pitch_rad = -math.atan2(vp[1] - cy, f)
    yaw_rad = -math.atan2(vp[0] - cx, f)
    
    def pixel_to_world(u, v):
        ray_c = np.array([u - cx, v - cy, f], dtype=float)
        ray_c /= np.linalg.norm(ray_c)
        
        Rx = np.array([[1,0,0], [0,math.cos(-pitch_rad),-math.sin(-pitch_rad)], [0,math.sin(-pitch_rad),math.cos(-pitch_rad)]])
        Ry = np.array([[math.cos(yaw_rad),0,math.sin(yaw_rad)], [0,1,0], [-math.sin(yaw_rad),0,math.cos(yaw_rad)]])
        ray_w = Ry @ (Rx @ ray_c)
        
        if ray_w[1] <= 0: return None
        scale = cam_h_mm / ray_w[1]
        return ray_w[0] * scale, ray_w[2] * scale

    world_pts, valid_src = [], []
    for pt in src_pts:
        w_coord = pixel_to_world(pt[0], pt[1])
        if w_coord:
            valid_src.append(pt)
            world_pts.append(w_coord)

    if len(valid_src) < 4:
        pitch_deg, yaw_deg = math.degrees(pitch_rad), math.degrees(yaw_rad)
        warped = get_telemetry_warp(img, cam_h_mm, pitch_deg, fov_deg, mm_px, out_w, out_h)
        return warped, vp, src_pts, pitch_deg, yaw_deg, lines, mask

    Z_min = min([Z for X, Z in world_pts]) if world_pts else 0

    dst_pts = []
    for X, Z in world_pts:
        dst_x = out_w / 2 + (X / mm_px)
        dst_y = out_h - ((Z - Z_min) / mm_px)
        dst_pts.append([dst_x, dst_y])

    M = cv2.getPerspectiveTransform(np.float32(valid_src), np.float32(dst_pts))
    warped = cv2.warpPerspective(img, M, (out_w, out_h))
    
    return warped, vp, src_pts, math.degrees(pitch_rad), math.degrees(yaw_rad), lines, mask

# ==========================================
# 4. FLASK ENDPOINTS
# ==========================================
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/parse_exif', methods=['POST'])
def parse_exif():
    """Extracts Focal Length and calculates FOV automatically."""
    try:
        data = request.json
        img_bytes = base64.b64decode(data['image'].split(',')[1])
        exif_dict = piexif.load(img_bytes)
        
        fov = 96.7 # Default fallback for GoPro
        focal_35 = exif_dict.get('Exif', {}).get(41989)
        if focal_35:
            fov = math.degrees(2 * math.atan(36.0 / (2.0 * focal_35)))
            
        return jsonify({'success': True, 'fov': round(fov, 2)})
    except Exception as e:
        return jsonify({'success': False, 'fov': 96.7, 'error': str(e)})

@app.route('/transform', methods=['POST'])
def transform():
    """Handles Manual, Hardcoded Telemetry, and AutoVP execution."""
    try:
        data = request.json
        img_bytes = base64.b64decode(data['image'].split(',')[1])
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        mode = data.get('mode', 'autovp')
        out_w, out_h = int(data.get('out_w', 800)), int(data.get('out_h', 1000))
        extra_data = {}

        if mode == 'manual':
            warped = get_full_warp_manual(img, data['points'], out_w, out_h)
        elif mode == 'telemetry':
            warped = get_telemetry_warp(
                img, float(data['cam_h']), float(data['pitch']), 
                float(data['fov']), float(data['mm_px']), out_w, out_h
            )
        elif mode == 'autovp':
            manual_vp = data.get('manual_vp')
            warped, vp, src_pts, pitch, yaw, lines, mask = get_autovp_warp(
                img, float(data['cam_h']), float(data['fov']), 
                float(data['mm_px']), out_w, out_h, manual_vp=manual_vp
            )
            extra_data['vp'] = {'x': float(vp[0]), 'y': float(vp[1])}
            extra_data['src_pts'] = src_pts.tolist()
            extra_data['pitch'] = float(pitch)
            extra_data['yaw'] = float(yaw)
            extra_data['lines'] = lines.tolist() if isinstance(lines, np.ndarray) else lines
            extra_data['mask'] = mask.tolist() if isinstance(mask, np.ndarray) else mask

        _, buffer = cv2.imencode('.jpg', warped)
        warped_b64 = base64.b64encode(buffer).decode('utf-8')
        
        resp = {'success': True, 'image': 'data:image/jpeg;base64,' + warped_b64}
        resp.update(extra_data)
        return jsonify(resp)
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/download', methods=['POST'])
def download():
    try:
        data = request.json
        img_bytes = base64.b64decode(data['image'].split(',')[1])
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        mode = data.get('mode', 'autovp')
        out_w, out_h = int(data.get('out_w', 800)), int(data.get('out_h', 1000))

        if mode == 'manual':
            warped = get_full_warp_manual(img, data['points'], out_w, out_h)
            mm_per_pixel = data.get('mm_per_pixel', 1.0)
        elif mode == 'telemetry':
            mm_per_pixel = float(data['mm_px'])
            warped = get_telemetry_warp(
                img, float(data['cam_h']), float(data['pitch']), 
                float(data['fov']), mm_per_pixel, out_w, out_h
            )
        elif mode == 'autovp':
            mm_per_pixel = float(data['mm_px'])
            manual_vp = data.get('manual_vp')
            warped, _, _, _, _, _, _ = get_autovp_warp(
                img, float(data['cam_h']), float(data['fov']), 
                mm_per_pixel, out_w, out_h, manual_vp=manual_vp
            )

        _, buffer = cv2.imencode('.jpg', warped)
        output_bytes = buffer.tobytes()

        try:
            exif_dict = piexif.load(img_bytes)
            if piexif.ExifIFD.MakerNote in exif_dict.get("Exif", {}):
                del exif_dict["Exif"][piexif.ExifIFD.MakerNote]
            exif_dict.pop("thumbnail", None)
        except Exception:
            exif_dict = {"0th": {}, "Exif": {}, "GPS": {}, "1st": {}, "thumbnail": None}

        ratio_data = json.dumps({"mm_per_pixel": mm_per_pixel})
        exif_dict["Exif"][piexif.ExifIFD.UserComment] = b"ASCII\x00\x00\x00" + ratio_data.encode('ascii')
        
        exif_bytes = piexif.dump(exif_dict)
        final_file_bytes = piexif.insert(exif_bytes, output_bytes)

        return send_file(
            io.BytesIO(final_file_bytes), 
            mimetype='image/jpeg', 
            as_attachment=True, 
            download_name='ml_ready_defect.jpg'
        )
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True, port=5000)