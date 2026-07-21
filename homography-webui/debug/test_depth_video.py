import os
import sys
import cv2
import math
import numpy as np
import argparse

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from ultralytics import YOLO
from parser_gpmf import extract_streams_with_time
from telemetry import get_telemetry_interpolators
from cv_projections import equirectangular_to_rectilinear
from cv_bev import get_bev_homography, get_fisheye_maps
from sam2_integration import load_sam2, run_sam2_on_detections
from depth_integration import estimate_pothole_depth

def main():
    parser = argparse.ArgumentParser(description="Standalone depth estimation test for video files.")
    parser.add_argument("video_path", type=str, help="Path to the video file")
    parser.add_argument("projection_distance", type=float, help="Extraction interval in meters")
    parser.add_argument("--standard", action="store_true", help="Force processing as standard video instead of 360")
    
    args = parser.parse_args()
    video_path = args.video_path
    proj_dist = args.projection_distance

    if not os.path.exists(video_path):
        print(f"Error: Video file {video_path} not found.")
        sys.exit(1)

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    model_path = os.path.join(project_root, 'models', 'RMCC_8_classes.pt')
    
    if os.path.exists(model_path):
        print("Loading YOLO model...")
        yolo_model = YOLO(model_path)
    else:
        print(f"Error: YOLO model not found at {model_path}.")
        sys.exit(1)

    print("Loading SAM2 model...")
    try:
        sam2_ckpt = os.path.join(project_root, 'models', 'sam2.1_hiera_large.pt')
        sam2_predictor = load_sam2(checkpoint_path=sam2_ckpt)
    except Exception as e:
        print(f"SAM2 failed to load: {e}")
        sys.exit(1)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    
    is_360 = (abs(w / h - 2.0) < 0.1)
    if args.standard:
        is_360 = False

    print(f"Video identified as {'360°' if is_360 else 'Standard'}")

    print("Extracting GPMF telemetry...")
    try:
        streams, constants = extract_streams_with_time(video_path)
        interpolators = get_telemetry_interpolators(streams)
    except Exception as e:
        print(f"Telemetry extraction failed: {e}")
        sys.exit(1)

    speed_interp = interpolators.get("speed")
    grav_x_interp = interpolators.get("grav_x")
    grav_y_interp = interpolators.get("grav_y")
    grav_z_interp = interpolators.get("grav_z")

    if not all([speed_interp, grav_x_interp, grav_y_interp, grav_z_interp]):
        print("Error: Missing required telemetry streams (speed or GRAV).")
        sys.exit(1)

    xfov = constants.get("XFOV")
    yfov = constants.get("YFOV")
    
    if xfov is None:
        zfov, aruw = constants.get('ZFOV'), constants.get('ARUW')
        if zfov is not None and aruw is not None:
            xfov = math.degrees(2.0 * math.atan(math.tan(math.radians(float(zfov)) / 2.0) * (float(aruw) / math.sqrt(float(aruw)**2 + 1))))
    if xfov is None:
        xfov = 100.0

    dist_accum = 0.0
    last_time = 0.0
    frame_idx = -1

    cam_height = 1.6
    y_min = 1.2
    y_max = 5.0
    road_width = 6.0
    fov_deg = float(xfov)
    gps_lag = 0.8

    output_dir = os.path.join(project_root, 'debug', 'depth_output')
    os.makedirs(output_dir, exist_ok=True)
    print(f"Outputs will be saved to: {output_dir}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        msec = cap.get(cv2.CAP_PROP_POS_MSEC)
        elapsed_sec = msec / 1000.0 if msec > 0 else frame_idx / fps
        sample_time = elapsed_sec + gps_lag
        
        try:
            speed = max(0.0, float(speed_interp(sample_time)))
        except ValueError:
            speed = 0.0
            
        dist_accum += speed * (elapsed_sec - last_time)
        last_time = elapsed_sec

        if dist_accum >= proj_dist or frame_idx == 0:
            if frame_idx > 0:
                intervals_consumed = max(1, int(dist_accum // proj_dist))
                dist_accum -= proj_dist * intervals_consumed
            else:
                dist_accum = 0.0
            
            print(f"\n--- Processing Frame {frame_idx} (Time: {elapsed_sec:.2f}s) ---")
            
            try:
                gx = float(grav_x_interp(sample_time))
                gy = float(grav_y_interp(sample_time))
                gz = float(grav_z_interp(sample_time))
            except ValueError:
                gx, gy, gz = 0.0, 1.0, 0.0

            norm = math.sqrt(gx*gx + gy*gy + gz*gz)
            if norm > 1e-6:
                gx, gy, gz = gx/norm, gy/norm, gz/norm

            pitch = math.degrees(math.asin(np.clip(-gz, -1.0, 1.0)))
            roll = math.degrees(math.atan2(gx, gy))

            if is_360:
                rect_img, K = equirectangular_to_rectilinear(frame, fov_deg, pitch, roll, 0.0)
                eff_grav_vec = [0.0, 1.0, 0.0]
            else:
                rect_img = frame.copy()
                h_img, w_img = rect_img.shape[:2]
                y_fov_deg = float(yfov) if yfov else math.degrees(2 * math.atan(math.tan(math.radians(fov_deg) / 2) * (h_img / w_img)))
                map1, map2, K = get_fisheye_maps(w_img, h_img, fov_deg, y_fov_deg)
                rect_img = cv2.remap(rect_img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
                eff_grav_vec = [gx, gy, gz]

            results = yolo_model.predict(source=rect_img, conf=0.25, verbose=False)
            
            if len(results) > 0 and len(results[0].boxes) > 0:
                print(f"Found {len(results[0].boxes)} detections. Running SAM2...")
                sam2_out = run_sam2_on_detections(cv2.cvtColor(rect_img, cv2.COLOR_BGR2RGB), results[0], sam2_predictor)
                
                if not sam2_out:
                    print("SAM2 did not produce any valid masks.")
                    continue

                H_mat, bev_w, bev_h, PPM, v_down, v_forward, v_right = get_bev_homography(
                    K, cam_height, eff_grav_vec, 0.0, y_min, y_max, road_width
                )
                
                for det_idx, (pts, cls_id, conf, class_name) in enumerate(sam2_out):
                    mask_canvas = np.zeros(rect_img.shape[:2], dtype=np.uint8)
                    cv2.fillPoly(mask_canvas, [pts.astype(np.int32)], 255)
                    bev_mask = cv2.warpPerspective(mask_canvas, H_mat, (bev_w, bev_h))
                    contours, _ = cv2.findContours(bev_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    
                    area_sqm = 0.0
                    for c in contours:
                        area_sqm += cv2.contourArea(c) * ((1.0 / PPM) ** 2)
                    
                    if area_sqm > 0:
                        depth_mm, overlay, poly = estimate_pothole_depth(rect_img, pts, K, cam_height, v_down, area_sqm)
                        print(f" -> {class_name} (Conf: {conf:.2f}): Area = {area_sqm:.4f} m², Depth = {depth_mm:.2f} mm")
                        
                        if overlay is not None:
                            out_name = os.path.join(output_dir, f"frame_{frame_idx:05d}_{class_name}_det{det_idx}.jpg")
                            cv2.imwrite(out_name, overlay)
                    else:
                        print(f" -> {class_name}: Mask projection resulted in 0 area.")
            else:
                print("No defects detected.")

    cap.release()
    print("\nProcessing complete.")

if __name__ == "__main__":
    main()
