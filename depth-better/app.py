import os
import uuid
import traceback
import shutil
import threading
import numpy as np
import trimesh
import cv2
import imageio
import matplotlib
from flask import Flask, render_template, request, jsonify
import torch

from depth_anything_3.api import DepthAnything3
from depth_anything_3.utils.memory import cleanup_cuda_memory

# Pulling directly from the local file in the same directory
from gopro_telemetry import process_gopro_telemetry, get_rotation_between_vectors

app = Flask(__name__)

# Configure directories
UPLOAD_FOLDER = os.path.join('static', 'uploads')
OUTPUT_FOLDER = os.path.join('static', 'outputs')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Global dictionary to hold background task statuses
TASKS = {}
CURRENT_MODEL_NAME = None
model = None

# Dynamic Model Loading to handle constraints
def load_model(model_name):
    global model, CURRENT_MODEL_NAME
    if CURRENT_MODEL_NAME == model_name and model is not None:
        return model
    
    print(f"Loading {model_name}...")
    if model is not None:
        del model
        cleanup_cuda_memory()
        
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Load model onto the best available compute device
    model = DepthAnything3(model_name=model_name).to(device)
    CURRENT_MODEL_NAME = model_name
    print(f"Model {model_name} loaded on {device.upper()}.")
    return model


def extract_frames(video_path, max_frames=8):
    """
    Extracts evenly spaced frames. Full resolution, no cropping,
    capped at max_frames to prevent OOM on laptops.
    """
    cap = cv2.VideoCapture(video_path)
    frames = []
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames == 0 or fps == 0:
        cap.release()
        raise ValueError("Could not read video properties.")
        
    skip = max(1, total_frames // max_frames)
    count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if count % skip == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
        count += 1
        if len(frames) >= max_frames:
            break
            
    cap.release()
    return frames, fps


def create_heatmap_video(prediction, output_path, fps=10):
    """
    Generates a side-by-side video of the original RGB frames and the depth heatmaps.
    """
    depths = prediction.depth             # (N, H, W)
    images = prediction.processed_images  # (N, H, W, 3)
    
    valid_mask = depths > 0
    inv_depths = np.zeros_like(depths)
    inv_depths[valid_mask] = 1.0 / depths[valid_mask]
    
    if valid_mask.sum() > 0:
        d_min = np.percentile(inv_depths[valid_mask], 2)
        d_max = np.percentile(inv_depths[valid_mask], 98)
    else:
        d_min, d_max = 0.0, 1.0
        
    if d_min == d_max:
        d_min -= 1e-6
        d_max += 1e-6
        
    cm = matplotlib.colormaps['Spectral']
    writer = imageio.get_writer(output_path, fps=fps, codec='libx264')
    
    for i in range(len(depths)):
        d = inv_depths[i]
        norm_d = 1 - np.clip((d - d_min) / (d_max - d_min), 0, 1)
        
        heatmap_rgba = cm(norm_d)
        heatmap_rgb = (heatmap_rgba[:, :, :3] * 255).astype(np.uint8)
        
        orig_img = images[i]
        combined = np.concatenate([orig_img, heatmap_rgb], axis=1)
        writer.append_data(combined)
        
    writer.close()


def fit_plane_ransac(points, distance_threshold=0.05, max_iterations=100):
    """
    Fits a mathematical plane to a set of 3D points using RANSAC.
    Handles inclined, banked, or crowned roads much better than a global Y-threshold.
    """
    best_inliers = []
    best_plane = None
    np.random.seed(42)
    
    for _ in range(max_iterations):
        # Pick 3 random points to form a plane
        sample = points[np.random.choice(points.shape[0], 3, replace=False)]
        v1 = sample[1] - sample[0]
        v2 = sample[2] - sample[0]
        normal = np.cross(v1, v2)
        norm = np.linalg.norm(normal)
        if norm < 1e-6:
            continue
        normal /= norm
        
        # Calculate plane equation: ax + by + cz + d = 0
        d = -np.dot(normal, sample[0])
        
        # Calculate distance of all points to the plane
        distances = np.abs(np.dot(points, normal) + d)
        inliers = np.where(distances < distance_threshold)[0]
        
        if len(inliers) > len(best_inliers):
            best_inliers = inliers
            best_plane = (normal, d)
            
    return best_plane, best_inliers


def export_pothole_mesh(prediction, export_path, telemetry_data, cam_height, frame_idx=None):
    """
    Converts the depth map to a solid mesh, aligning scale mathematically based
    on the user-provided camera height rather than drifty GPS telemetry.
    """
    if frame_idx is None:
        frame_idx = len(prediction.depth) // 2
        
    depth = prediction.depth[frame_idx]             # (H, W)
    image = prediction.processed_images[frame_idx].copy()  # (H, W, 3)
    
    H, W = depth.shape
    focal = max(H, W)
    
    # Using standard W/2 and H/2 since we did NOT crop the image
    K = np.array([[focal, 0, W/2], [0, focal, H/2], [0, 0, 1]])
    
    us, vs = np.meshgrid(np.arange(W), np.arange(H))
    pix = np.stack([us, vs, np.ones_like(us)], axis=-1).reshape(-1, 3)
    
    # Unproject to 3D Camera Space
    K_inv = np.linalg.inv(K)
    rays = (K_inv @ pix.T).T
    pts_cam = rays * depth.reshape(-1, 1)
    
    # Generate Faces
    idx = np.arange(H * W).reshape(H, W)
    tl = idx[:-1, :-1].flatten()
    tr = idx[:-1, 1:].flatten()
    bl = idx[1:, :-1].flatten()
    br = idx[1:, 1:].flatten()
    faces1 = np.stack([tl, bl, tr], axis=1)
    faces2 = np.stack([tr, bl, br], axis=1)
    faces = np.vstack([faces1, faces2])
    
    # Filter Rubber-sheet faces
    depth_flat = depth.flatten()
    v0, v1, v2 = depth_flat[faces[:, 0]], depth_flat[faces[:, 1]], depth_flat[faces[:, 2]]
    max_diff = np.maximum(np.maximum(np.abs(v0 - v1), np.abs(v1 - v2)), np.abs(v2 - v0))
    min_depth = np.minimum(np.minimum(v0, v1), v2) + 1e-6
    
    # Relaxed threshold to 1.5 to prevent deleting pothole cliffs while still 
    # culling severe rubber-sheet stretching on distant background objects
    valid_faces = faces[(max_diff / min_depth) < 1.5]
    
    # Remove sky faces if available
    if getattr(prediction, 'sky', None) is not None:
        sky_flat = prediction.sky[frame_idx].flatten()
        is_sky_face = sky_flat[valid_faces[:, 0]] | sky_flat[valid_faces[:, 1]] | sky_flat[valid_faces[:, 2]]
        valid_faces = valid_faces[~is_sky_face]
        
    pts_gltf = pts_cam.copy()
    pts_gltf[:, 1] *= -1  # Flip Y
    pts_gltf[:, 2] *= -1  # Flip Z

    colors = image.reshape(-1, 3)

    # =========================================================================
    # RANSAC + TELEMETRY ROAD PLANE FITTING
    # =========================================================================
    if telemetry_data is not None:
        # Orient using GoPro Gravity so down is negative Y
        gravity_vec = telemetry_data['gravity_vectors'][frame_idx]
        target_down = np.array([0, -1, 0])
        R_align = get_rotation_between_vectors(gravity_vec, target_down)
        pts_gltf = (R_align @ pts_gltf.T).T
        
        # Because we didn't crop the sky, we must ensure RANSAC only fits to the bottom of the image
        # We take the lowest 40% of the Y points (since Y is UP, lowest Y is the road)
        y_values = pts_gltf[:, 1]
        lower_center_mask = (y_values < np.percentile(y_values, 40))
        road_points = pts_gltf[lower_center_mask]
        
        if len(road_points) > 100:
            # 3cm is good for initial RANSAC tight fitting to find the overall road plane
            (normal, d), inliers = fit_plane_ransac(road_points, distance_threshold=0.03)
            
            # Ensure normal points UP (towards the camera positive Y)
            if normal[1] < 0:
                normal = -normal
                d = -d
            
            # Absolute metric scaling: We force the distance from the camera (origin)
            # to the road plane to be exactly `cam_height`.
            dist_to_plane_unscaled = abs(d) / np.linalg.norm(normal)
            scale_factor = cam_height / dist_to_plane_unscaled
            
            pts_gltf *= scale_factor
            d *= scale_factor
            
            # Recalculate distance to the accurately scaled road plane
            distances = (np.dot(pts_gltf, normal) + d)
            
            # Potholes are points deeper than 5cm below the road plane to account for natural crown/noise
            is_pothole = distances < -0.05
            
            # Visual Styling: Dim the healthy road to Gray, highlight potholes in pure Red
            gray_tint = (colors.mean(axis=1, keepdims=True) * 0.7).astype(np.uint8)
            colors = np.repeat(gray_tint, 3, axis=1)
            colors[is_pothole] = [255, 0, 0]

    # Center the mesh for viewing
    valid_vert_indices = np.unique(valid_faces.flatten())
    if len(valid_vert_indices) > 0:
        center = np.median(pts_gltf[valid_vert_indices], axis=0)
        pts_gltf -= center
    
    mesh = trimesh.Trimesh(vertices=pts_gltf, faces=valid_faces, vertex_colors=colors)
    mesh.remove_unreferenced_vertices()
    mesh.export(export_path)


def process_video_task(task_id, file_path, cam_height, model_size):
    """
    Background worker thread function that runs the heavy AI multi-view logic.
    """
    try:
        active_model = load_model(model_size)
        
        # Only the Giant models support Gaussian Splatting
        infer_gs = model_size in ['da3-giant', 'da3nested-giant-large']
        export_format = "glb-gs_ply" if infer_gs else "glb"
        
        TASKS[task_id]['progress'] = "Parsing media file..."
        export_dir = os.path.join(OUTPUT_FOLDER, task_id)
        os.makedirs(export_dir, exist_ok=True)
        
        ext = os.path.splitext(file_path)[1].lower()
        if ext in ['.mp4', '.webm', '.mov', '.avi']:
            TASKS[task_id]['progress'] = "Extracting frames and Physical Telemetry..."
            # Using 8 frames to save laptop memory. Full resolution (no crop).
            frames, orig_fps = extract_frames(file_path, max_frames=8)
            out_fps = max(5, int(orig_fps / max(1, (int(orig_fps) // 5)))) if orig_fps > 0 else 5
            
            telemetry_data = process_gopro_telemetry(file_path, len(frames))
        else:
            frame = cv2.imread(file_path)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames = [frame_rgb]
            out_fps = 1
            telemetry_data = None

        TASKS[task_id]['progress'] = f"Running AI Depth Inference on {len(frames)} frames..."
        
        # Run inference. We purposefully DO NOT pass GPS extrinsics because they
        # introduce translation drift that breaks scaling.
        prediction = active_model.inference(
            image=frames,
            align_to_input_ext_scale=False,
            export_dir=export_dir,
            export_format=export_format,
            infer_gs=infer_gs,
            ref_view_strategy="middle", 
            process_res=1008, # Increased processing resolution to preserve fine pothole detail
            export_kwargs={
                "glb": {
                    "num_max_points": 500000, 
                    "show_cameras": False
                },
                "gs_ply": {
                    "gs_views_interval": 1
                }
            }
        )
        
        TASKS[task_id]['progress'] = "Organizing 3D Files..."
        native_ply = os.path.join(export_dir, "gs_ply", "0000.ply")
        native_glb = os.path.join(export_dir, "scene.glb")
        
        # Assign the correct extension based on what was rendered
        pc_extension = "ply" if (infer_gs and os.path.exists(native_ply)) else "glb"
        pointcloud_path = os.path.join(export_dir, f"pointcloud.{pc_extension}")
        
        if pc_extension == "ply":
            shutil.move(native_ply, pointcloud_path)
        elif os.path.exists(native_glb):
            shutil.move(native_glb, pointcloud_path)
            
        TASKS[task_id]['progress'] = "Analyzing Road Plane & Highlighting Potholes..."
        mesh_path = os.path.join(export_dir, "mesh.glb")
        export_pothole_mesh(prediction, mesh_path, telemetry_data, cam_height)

        TASKS[task_id]['progress'] = "Rendering Heatmap Video..."
        heatmap_path = os.path.join(export_dir, "heatmap.mp4")
        create_heatmap_video(prediction, heatmap_path, fps=out_fps)
        
        cleanup_cuda_memory()
        
        TASKS[task_id]['status'] = 'completed'
        TASKS[task_id]['result'] = {
            'pointcloud_url': f"/{export_dir}/pointcloud.{pc_extension}",
            'mesh_url': f"/{export_dir}/mesh.glb",
            'heatmap_url': f"/{export_dir}/heatmap.mp4"
        }
        
    except Exception as e:
        cleanup_cuda_memory()
        traceback.print_exc()
        TASKS[task_id]['status'] = 'error'
        TASKS[task_id]['error'] = f"Processing failed: {str(e)}"


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/process', methods=['POST'])
def process():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded.'}), 400
    
    file = request.files['file']
    cam_height = float(request.form.get('cam_height', 1.5))
    model_size = request.form.get('model_size', 'da3nested-giant-large')
    
    if file.filename == '':
        return jsonify({'error': 'Empty filename.'}), 400

    session_id = str(uuid.uuid4())
    
    ext = os.path.splitext(file.filename)[1].lower()
    upload_path = os.path.join(UPLOAD_FOLDER, f"{session_id}{ext}")
    file.save(upload_path)
    
    TASKS[session_id] = {
        'status': 'processing',
        'progress': 'Initializing task...',
        'result': None,
        'error': None
    }
    
    thread = threading.Thread(target=process_video_task, args=(session_id, upload_path, cam_height, model_size))
    thread.daemon = True
    thread.start()
    
    return jsonify({
        'success': True,
        'task_id': session_id
    })


@app.route('/status/<task_id>', methods=['GET'])
def status(task_id):
    if task_id not in TASKS:
        return jsonify({'error': 'Invalid Task ID'}), 404
        
    return jsonify(TASKS[task_id])


# Custom static routing for outputs so the frontend can retrieve the processed GLBs and PLYs
@app.route('/static/outputs/<task_id>/<filename>')
def serve_output(task_id, filename):
    return app.send_static_file(os.path.join('outputs', task_id, filename))


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
