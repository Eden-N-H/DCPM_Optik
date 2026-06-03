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
        # EXIF Tag 41989 = FocalLengthIn35mmFilm
        focal_35 = exif_dict.get('Exif', {}).get(41989)
        if focal_35:
            # Calculate FOV based on standard 36mm full-frame width
            fov = math.degrees(2 * math.atan(36.0 / (2.0 * focal_35)))
            
        return jsonify({'success': True, 'fov': round(fov, 2)})
    except Exception as e:
        return jsonify({'success': False, 'fov': 96.7, 'error': str(e)})

@app.route('/transform', methods=['POST'])
def transform():
    """Handles both Manual and Telemetry generation for the UI Preview."""
    try:
        data = request.json
        img_bytes = base64.b64decode(data['image'].split(',')[1])
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        mode = data.get('mode', 'manual')
        out_w, out_h = int(data.get('out_w', 800)), int(data.get('out_h', 1000))

        if mode == 'manual':
            warped = get_full_warp_manual(img, data['points'], out_w, out_h)
        else:
            warped = get_telemetry_warp(
                img, float(data['cam_h']), float(data['pitch']), 
                float(data['fov']), float(data['mm_px']), out_w, out_h
            )

        _, buffer = cv2.imencode('.jpg', warped)
        warped_b64 = base64.b64encode(buffer).decode('utf-8')
        return jsonify({'success': True, 'image': 'data:image/jpeg;base64,' + warped_b64})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/download', methods=['POST'])
def download():
    try:
        data = request.json
        img_bytes = base64.b64decode(data['image'].split(',')[1])
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        mode = data.get('mode', 'manual')
        out_w, out_h = int(data.get('out_w', 800)), int(data.get('out_h', 1000))

        # Perform correct warp based on mode
        if mode == 'manual':
            warped = get_full_warp_manual(img, data['points'], out_w, out_h)
            mm_per_pixel = data.get('mm_per_pixel', 1.0)
        else:
            mm_per_pixel = float(data['mm_px'])
            warped = get_telemetry_warp(
                img, float(data['cam_h']), float(data['pitch']), 
                float(data['fov']), mm_per_pixel, out_w, out_h
            )

        _, buffer = cv2.imencode('.jpg', warped)
        output_bytes = buffer.tobytes()

        # Handle EXIF transfer
        try:
            exif_dict = piexif.load(img_bytes)
            if piexif.ExifIFD.MakerNote in exif_dict.get("Exif", {}):
                del exif_dict["Exif"][piexif.ExifIFD.MakerNote]
            exif_dict.pop("thumbnail", None)
        except Exception:
            exif_dict = {"0th": {}, "Exif": {}, "GPS": {}, "1st": {}, "thumbnail": None}

        # Inject metric scale ratio into EXIF UserComment for Machine Learning models downstream
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