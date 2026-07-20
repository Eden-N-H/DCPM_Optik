# --- START OF FILE debug/test_depth_video.py ---
import os
import sys
import time
import cv2
import numpy as np
import math
from ultralytics import YOLO

if len(sys.argv) > 1:
    VIDEO_PATH = os.path.abspath(sys.argv[1])
else:
    print("[-] Usage: python test_depth_video.py \"<path_to_video.mp4>\" [cam_height_m] [fov_override] [pitch_override]")
    print("    Example: python test_depth_video.py dashcam.mp4 1.5")
    sys.exit(1)

if not os.path.exists(VIDEO_PATH):
    print(f"[-] Video not found: {VIDEO_PATH}")
    sys.exit(1)

# Argument Parsing
CAM_HEIGHT_M = float(sys.argv[2]) if len(sys.argv) > 2 else 1.5
MANUAL_FOV = float(sys.argv[3]) if len(sys.argv) > 3 else None
MANUAL_PITCH = float(sys.argv[4]) if len(sys.argv) > 4 else None

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
os.chdir(PROJECT_ROOT)
sys.path.append(os.path.join(PROJECT_ROOT, 'src'))

from parser_gpmf import extract_streams_with_time
from telemetry import get_telemetry_interpolators
from depth_integration import estimate_pothole_depth
from sam2_integration import load_sam2
from cv_bev import get_fisheye_maps, apply_ui_offsets_to_vectors

# Globals for Manual Bounding Box Fallback
drawing = False
ix, iy = -1, -1
fx, fy = -1, -1
box_drawn = False

def draw_bbox(event, x, y, flags, param):
    global ix, iy, fx, fy, drawing, box_drawn
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
        fx, fy = x, y
        box_drawn = False
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            fx, fy = x, y
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        fx, fy = x, y
        box_drawn = True

def main():
    print(f"\n[*] Scanning video for GPMF telemetry: {os.path.basename(VIDEO_PATH)}")
    try:
        streams, constants = extract_streams_with_time(VIDEO_PATH)
        interpolators = get_telemetry_interpolators(streams)
        print("[+] GPMF Telemetry successfully extracted and interpolated.")
    except Exception as e:
        print(f"[-] Failed to extract GPMF from video: {e}")
        streams, constants, interpolators = {}, {}, {}

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print("[-] OpenCV could not open the video.")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    
    # 1. Video Scrubbing UI
    print("\n=======================================================")
    print("[*] VIDEO SCRUBBER CONTROLS:")
    print("    [A] / [D] : Backward / Forward 30 frames")
    print("    [Z] / [C] : Backward / Forward 1 frame")
    print("    [ENTER]   : Select current frame for AI automated analysis")
    print("    [Q]       : Quit")
    print("=======================================================")
    
    cv2.namedWindow("Video Scrubber", cv2.WINDOW_NORMAL)
    frame_idx = 0
    selected_frame = None
    
    while True:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            frame_idx = max(0, frame_idx - 1)
            continue
            
        display = frame.copy()
        timestamp = frame_idx / fps
        cv2.putText(display, f"Frame: {frame_idx}/{total_frames} | Time: {timestamp:.2f}s", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.imshow("Video Scrubber", display)
        key = cv2.waitKey(0) & 0xFF
        
        if key == ord('d'): frame_idx = min(total_frames - 1, frame_idx + 30)
        elif key == ord('a'): frame_idx = max(0, frame_idx - 30)
        elif key == ord('c'): frame_idx = min(total_frames - 1, frame_idx + 1)
        elif key == ord('z'): frame_idx = max(0, frame_idx - 1)
        elif key == 13: # ENTER
            selected_frame = frame.copy()
            break
        elif key == ord('q') or key == 27:
            cap.release()
            cv2.destroyAllWindows()
            return
            
    cap.release()
    cv2.destroyAllWindows()
    
    # 2. Extract Telemetry for Specific Frame
    print(f"\n[*] Extracting Telemetry at {timestamp:.2f} seconds...")
    grav_x_interp = interpolators.get("grav_x")
    grav_y_interp = interpolators.get("grav_y")
    grav_z_interp = interpolators.get("grav_z")
    
    if all([grav_x_interp, grav_y_interp, grav_z_interp]):
        gx = float(grav_x_interp(timestamp))
        gy = float(grav_y_interp(timestamp))
        gz = float(grav_z_interp(timestamp))
        grav_vec = [gx, gy, gz]
        telemetry_pitch = -math.degrees(math.atan2(gz, gy))
        telemetry_roll = math.degrees(math.atan2(gx, gy))
        print(f"[+] Interpolated Gravity Vector: [{gx:.3f}, {gy:.3f}, {gz:.3f}]")
        print(f"[+] Derived Telemetry Pitch: {telemetry_pitch:.2f}° | Roll: {telemetry_roll:.2f}°")
    else:
        print("[-] Missing Gravity interpolators. Falling back to default [0, 1, 0].")
        grav_vec = [0.0, 1.0, 0.0]
        telemetry_pitch, telemetry_roll = 0.0, 0.0

    telemetry_fov_x = constants.get('XFOV')
    telemetry_fov_y = constants.get('YFOV')
    if telemetry_fov_x is None:
        zfov, aruw = constants.get('ZFOV'), constants.get('ARUW')
        if zfov is not None and aruw is not None:
            try: telemetry_fov_x = math.degrees(2.0 * math.atan(math.tan(math.radians(float(zfov)) / 2.0) * (float(aruw) / math.sqrt(float(aruw)**2 + 1))))
            except: pass

    # Apply FOV Overrides
    h_raw, w_raw = selected_frame.shape[:2]
    if MANUAL_FOV is not None:
        final_xfov = MANUAL_FOV
        final_yfov = MANUAL_FOV * (h_raw / w_raw)
        print(f"[!] OVERRIDE: Using manual FOV: {final_xfov}°")
    else:
        final_xfov = float(telemetry_fov_x) if telemetry_fov_x is not None else 120.0
        final_yfov = float(telemetry_fov_y) if telemetry_fov_y is not None else 120.0 * (h_raw / w_raw)

    # 3. Vector Projection Math
    if MANUAL_PITCH is not None:
        print(f"[!] OVERRIDE: Using manual Pitch: {MANUAL_PITCH}°")
        g = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        pitch_offset = MANUAL_PITCH
    else:
        g = np.array(grav_vec, dtype=np.float64)
        if np.linalg.norm(g) > 1e-6: g = g / np.linalg.norm(g)
        else: g = np.array([0.0, 1.0, 0.0])
        pitch_offset = 0.0

    v_down, _ = apply_ui_offsets_to_vectors(g, np.array([0,0,1], dtype=np.float64), pitch_offset, 0.0, 0.0)
    
    z_cam = np.array([0.0, 0.0, 1.0])
    v_forward = z_cam - (np.dot(z_cam, v_down) * v_down)
    if np.linalg.norm(v_forward) > 1e-6: v_forward = v_forward / np.linalg.norm(v_forward)
    else: v_forward = np.array([0, 0, 1])
    
    v_right = np.cross(v_down, v_forward)
    if np.linalg.norm(v_right) > 1e-6: v_right = v_right / np.linalg.norm(v_right)
    else: v_right = np.array([1, 0, 0])

    print("[*] Rectifying Image to 3D Plane...")
    map1, map2, _ = get_fisheye_maps(w_raw, h_raw, final_xfov, final_yfov)
    img_rect = cv2.remap(selected_frame, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

    f_rect_x = (w_raw / 2.0) / math.tan(math.radians(final_xfov) / 2.0)
    f_rect_y = (h_raw / 2.0) / math.tan(math.radians(final_yfov) / 2.0)
    K_rect_true = np.array([[f_rect_x, 0, w_raw / 2.0], [0, f_rect_y, h_raw / 2.0], [0, 0, 1]], dtype=np.float32)

    # 4. Automated AI Pipeline (YOLO)
    print("\n=======================================================")
    print("[*] STARTING AUTOMATED SEGMENTATION & DEPTH ESTIMATION")
    
    yolo_model_path = os.path.join(PROJECT_ROOT, 'models', 'RMCC_8_classes.pt')
    yolo_model = YOLO(yolo_model_path) if os.path.exists(yolo_model_path) else None
    
    boxes_to_process = []
    
    if yolo_model:
        print("[*] Running YOLO AI Inference...")
        yolo_results = yolo_model.predict(img_rect, conf=0.15, verbose=False)
        for r in yolo_results:
            if r.boxes is not None and len(r.boxes) > 0:
                for i in range(len(r.boxes)):
                    cls_id = int(r.boxes.cls[i].cpu().numpy())
                    class_name = yolo_model.names[cls_id]
                    box = r.boxes.xyxy[i].cpu().numpy().astype(np.float32)
                    boxes_to_process.append((box, class_name))

    # 5. Fallback: Manual Bounding Box if AI finds nothing
    if len(boxes_to_process) == 0:
        print("[-] No Potholes or structural defects detected by AI in this frame.")
        print("[*] FALLING BACK TO MANUAL SELECTION: Draw a box around the target object.")
        
        clone = img_rect.copy()
        cv2.namedWindow("Manual Override - Draw BBox", cv2.WINDOW_NORMAL)
        cv2.setMouseCallback("Manual Override - Draw BBox", draw_bbox)

        while True:
            temp = clone.copy()
            if drawing or box_drawn:
                cv2.rectangle(temp, (ix, iy), (fx, fy), (0, 165, 255), 2)
            cv2.imshow("Manual Override - Draw BBox", temp)
            
            key = cv2.waitKey(1) & 0xFF
            if key == 13 and box_drawn: break # ENTER
            elif key == 27 or key == ord('q'): 
                cv2.destroyAllWindows()
                return

        cv2.destroyAllWindows()
        cv2.waitKey(1)
        
        x_min, x_max = max(0, min(ix, fx)), min(w_raw - 1, max(ix, fx))
        y_min, y_max = max(0, min(iy, fy)), min(h_raw - 1, max(iy, fy))
        boxes_to_process.append((np.array([x_min, y_min, x_max, y_max], dtype=np.float32), "Manual Testing Cutout"))

    # 6. SAM2 Segmentation and Depth Processing
    sam2_predictor = load_sam2()
    img_rgb = cv2.cvtColor(img_rect, cv2.COLOR_BGR2RGB)
    sam2_predictor.set_image(img_rgb)
    
    K_inv = np.linalg.inv(K_rect_true)
    overlay_canvas = img_rect.copy()

    for box, class_name in boxes_to_process:
        box_array = np.array([box], dtype=np.float32)
        
        masks, scores, _ = sam2_predictor.predict(box=box_array, multimask_output=False)
        
        if masks.ndim == 4: mask_2d = masks[0, 0]
        elif masks.ndim == 3: mask_2d = masks[0]
        else: mask_2d = masks
            
        binary_mask = (mask_2d > 0).astype(np.uint8)
        if binary_mask.sum() == 0: continue
            
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours: continue
            
        largest_contour = max(contours, key=cv2.contourArea)
        approx_polygon = cv2.approxPolyDP(largest_contour, 0.002 * cv2.arcLength(largest_contour, True), True)
        sam2_polygon_points = approx_polygon.reshape(-1, 2).tolist()
        
        if len(sam2_polygon_points) < 3: continue

        # Internal ray casting for mathematical reference (not displayed)
        poly_3d_local = []
        for pt in sam2_polygon_points:
            u, v = pt
            ray = K_inv @ np.array([u, v, 1.0])
            dot = np.dot(v_down, ray)
            if dot > 1e-5:
                Z_road = CAM_HEIGHT_M / dot
                P_3d = Z_road * ray
                X_local = np.dot(P_3d, v_right)
                Y_local = np.dot(P_3d, v_forward)
                poly_3d_local.append((X_local, Y_local))
                
        exact_area_sqm = 0.0
        n_pts = len(poly_3d_local)
        if n_pts < 3: continue
        
        for j in range(n_pts):
            k = (j + 1) % n_pts
            exact_area_sqm += poly_3d_local[j][0] * poly_3d_local[k][1]
            exact_area_sqm -= poly_3d_local[k][0] * poly_3d_local[j][1]
        exact_area_sqm = abs(exact_area_sqm) / 2.0

        # Execute Morphometric Depth Estimation
        max_depth_mm, depth_overlay, refined_polygon = estimate_pothole_depth(
            overlay_canvas, sam2_polygon_points, K_rect_true, CAM_HEIGHT_M, v_down, exact_area_sqm
        )

        if depth_overlay is not None:
            overlay_canvas = depth_overlay
            cv2.polylines(overlay_canvas, [np.array(refined_polygon)], isClosed=True, color=(0, 255, 0), thickness=2)
            
            pts_arr = np.array(refined_polygon)
            cx, cy = np.mean(pts_arr, axis=0).astype(int)
            cv2.putText(overlay_canvas, f"{class_name} Depth: {max_depth_mm:.1f} mm", (cx - 50, cy - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            print(f"[+] Processed: {class_name} -> Estimated Depth: {max_depth_mm:.1f} mm")

    print("\n[+] Displaying Final Overlay...")
    cv2.imshow("Automated Morphometric Depth", overlay_canvas)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
# --- END OF FILE ---