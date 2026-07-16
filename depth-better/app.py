import os
import uuid
import traceback
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
# FIXED: Using .from_pretrained() to ensure the model weights are actually downloaded and loaded.
model = DepthAnything3.from_pretrained("depth-anything/DA3-LARGE").to(device)

print(f"Model loaded successfully on {device.upper()}.")

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
        # Generate a unique ID for this session to prevent overwriting
        session_id = str(uuid.uuid4())
        
        # Save uploaded image
        upload_path = os.path.join(UPLOAD_FOLDER, f"{session_id}.png")
        file.save(upload_path)
        
        # Define output directory for DA3
        export_dir = os.path.join(OUTPUT_FOLDER, session_id)
        
        # Run DepthAnything3 Inference
        # We request both 'glb' (3D mesh) and 'depth_vis' (heatmap) formats
        prediction = model.inference(
            image=[upload_path],
            export_dir=export_dir,
            export_format="glb-depth_vis",
            # Cap the points to 250k so the web browser's 3D viewer doesn't lag
            export_kwargs={
                "glb": {
                    "num_max_points": 250000,
                    "show_cameras": False # Hide camera wireframes for cleaner UI
                }
            }
        )
        
        # Clean up GPU memory after inference
        cleanup_cuda_memory()
        
        # DA3 saves the mesh to {export_dir}/scene.glb
        # DA3 saves the depth heatmap to {export_dir}/depth_vis/0000.jpg
        # We format these as absolute URL paths for the frontend
        mesh_url = f"/{export_dir}/scene.glb"
        heatmap_url = f"/{export_dir}/depth_vis/0000.jpg"
        
        return jsonify({
            'success': True,
            'mesh_url': mesh_url,
            'heatmap_url': heatmap_url
        })
        
    except Exception as e:
        cleanup_cuda_memory()
        traceback.print_exc()
        return jsonify({'error': f"Processing failed: {str(e)}"}), 500

if __name__ == '__main__':
    # Run the app on port 5000
    app.run(host='0.0.0.0', port=5000, debug=True)