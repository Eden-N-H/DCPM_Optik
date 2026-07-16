import os
import uuid
import traceback
import shutil
import numpy as np
import trimesh
from flask import Flask, render_template, request, jsonify
import torch

from depth_anything_3.api import DepthAnything3
from depth_anything_3.utils.memory import cleanup_cuda_memory

app = Flask(__name__)

# Configure directories
UPLOAD_FOLDER = os.path.join('static', 'uploads')
OUTPUT_FOLDER = os.path.join('static', 'outputs')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Load the DepthAnything3 model globally so it's ready for inference
print("Initializing DepthAnything3 model (this may take a moment)...")
device = "cuda" if torch.cuda.is_available() else "cpu"

# Using 'da3-large' as it's the recommended balance of performance and quality
model = DepthAnything3.from_pretrained("depth-anything/DA3-LARGE").to(device)

print(f"Model loaded successfully on {device.upper()}.")


def export_solid_mesh(prediction, export_path):
    """Converts the depth map to a solid 2.5D polygon mesh and saves as GLB."""
    depth = prediction.depth[0]             # (H, W)
    image = prediction.processed_images[0]  # (H, W, 3)
    K = prediction.intrinsics[0]            # (3, 3)
    
    H, W = depth.shape
    
    # 1. Generate 2D pixel grid
    us, vs = np.meshgrid(np.arange(W), np.arange(H))
    pix = np.stack([us, vs, np.ones_like(us)], axis=-1).reshape(-1, 3)
    
    # 2. Unproject to 3D Camera Space
    K_inv = np.linalg.inv(K)
    rays = (K_inv @ pix.T).T  # (H*W, 3)
    pts_cam = rays * depth.reshape(-1, 1)  # (H*W, 3)
    
    # 3. Generate Faces (connecting adjacent pixels into triangles)
    idx = np.arange(H * W).reshape(H, W)
    tl = idx[:-1, :-1].flatten()
    tr = idx[:-1, 1:].flatten()
    bl = idx[1:, :-1].flatten()
    br = idx[1:, 1:].flatten()
    
    faces1 = np.stack([tl, bl, tr], axis=1)
    faces2 = np.stack([tr, bl, br], axis=1)
    faces = np.vstack([faces1, faces2])
    
    # 4. Filter Stretched Faces (Rubber-sheet effect)
    depth_flat = depth.flatten()
    v0 = depth_flat[faces[:, 0]]
    v1 = depth_flat[faces[:, 1]]
    v2 = depth_flat[faces[:, 2]]
    
    max_diff = np.maximum(np.maximum(np.abs(v0 - v1), np.abs(v1 - v2)), np.abs(v2 - v0))
    min_depth = np.minimum(np.minimum(v0, v1), v2) + 1e-6
    
    valid_faces = faces[(max_diff / min_depth) < 0.10]
    
    # 5. Remove sky faces if a sky mask is available
    if getattr(prediction, 'sky', None) is not None:
        sky_flat = prediction.sky[0].flatten()
        is_sky_face = sky_flat[valid_faces[:, 0]] | sky_flat[valid_faces[:, 1]] | sky_flat[valid_faces[:, 2]]
        valid_faces = valid_faces[~is_sky_face]
        
    if len(valid_faces) == 0:
        valid_faces = faces 
        
    # 6. Transform to glTF coordinates (Y-up, Z-backward)
    pts_gltf = pts_cam.copy()
    pts_gltf[:, 1] *= -1  # Flip Y
    pts_gltf[:, 2] *= -1  # Flip Z
    
    valid_vert_indices = np.unique(valid_faces.flatten())
    if len(valid_vert_indices) > 0:
        center = np.median(pts_gltf[valid_vert_indices], axis=0)
        pts_gltf -= center
    
    # 7. Create trimesh with vertex colors
    colors = image.reshape(-1, 3)
    mesh = trimesh.Trimesh(vertices=pts_gltf, faces=valid_faces, vertex_colors=colors)
    mesh.remove_unreferenced_vertices()
    
    mesh.export(export_path)


@app.route('/')
def index():
    """Render the main UI."""
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process():
    """Handle image upload and model inference."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded.'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Empty filename.'}), 400

    try:
        session_id = str(uuid.uuid4())
        upload_path = os.path.join(UPLOAD_FOLDER, f"{session_id}.png")
        file.save(upload_path)
        
        export_dir = os.path.join(OUTPUT_FOLDER, session_id)
        
        # Run DepthAnything3 Inference (Export Point Cloud AND Heatmap)
        prediction = model.inference(
            image=[upload_path],
            export_dir=export_dir,
            export_format="glb-depth_vis",
            export_kwargs={
                "glb": {
                    "num_max_points": 250000,
                    "show_cameras": False
                }
            }
        )
        
        # The native GLB export saves to `scene.glb`. Rename it to `pointcloud.glb` safely.
        native_glb = os.path.join(export_dir, "scene.glb")
        pointcloud_path = os.path.join(export_dir, "pointcloud.glb")
        if os.path.exists(native_glb):
            shutil.move(native_glb, pointcloud_path)
            
        # Generate our solid 2.5D mesh
        mesh_path = os.path.join(export_dir, "mesh.glb")
        export_solid_mesh(prediction, mesh_path)
        
        cleanup_cuda_memory()
        
        return jsonify({
            'success': True,
            'pointcloud_url': f"/{export_dir}/pointcloud.glb",
            'mesh_url': f"/{export_dir}/mesh.glb",
            'heatmap_url': f"/{export_dir}/depth_vis/0000.jpg"
        })
        
    except Exception as e:
        cleanup_cuda_memory()
        traceback.print_exc()
        return jsonify({'error': f"Processing failed: {str(e)}"}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
