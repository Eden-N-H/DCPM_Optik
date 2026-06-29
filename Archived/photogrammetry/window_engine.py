import os
# 🛠️ Fixes the OMP library crash
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
import torch
import numpy as np
from PIL import Image
import pathlib

def run_ai_depth_engine():
    print("🧠 Initializing AI Monocular Depth Engine...")
    
    try:
        from transformers import pipeline
    except ImportError:
        print("❌ Error: 'transformers' library not found.")
        sys.exit(1)

    base_dir = pathlib.Path(__file__).parent.resolve()
    image_dir = base_dir / "clean_images"
    
    if not image_dir.exists() or not os.listdir(image_dir):
        print("❌ Error: 'clean_images' folder is empty or missing.")
        sys.exit(1)

    image_files = sorted([f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    
    target_image = image_dir / image_files[0]
    print(f"📸 Analyzing full environment matrix on: {target_image.name}")

    img = Image.open(target_image).convert("RGB")
    width, height = img.size

    print("⏳ Loading AI weights from HuggingFace (Forcing CPU to prevent Mac memory crashes)...")
    # Explicitly force CPU inference (-1) to guarantee no Apple Silicon GPU memory faults
    pipe = pipeline(task="depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf", device=-1)
    
    print("🔮 Predicting dense 3D surface geometry (this might take a few seconds)...")
    result = pipe(img)
    
    predicted_depth = result["predicted_depth"]
    if hasattr(predicted_depth, "numpy"):
        depth_map = predicted_depth.detach().cpu().numpy()
    else:
        depth_map = np.array(predicted_depth)

    depth_map = np.squeeze(depth_map)
    if depth_map.shape != (height, width):
        depth_pil = Image.fromarray(depth_map).resize((width, height), Image.Resampling.BILINEAR)
        depth_map = np.array(depth_pil)

    depth_map = depth_map.astype(np.float32)
    depth_min, depth_max = depth_map.min(), depth_map.max()
    if depth_max - depth_min > 0:
        depth_norm = (depth_map - depth_min) / (depth_max - depth_min)
    else:
        depth_norm = depth_map

    z = 1.0 / (depth_norm + 0.05) * 7.0 
    
    x_indices, y_indices = np.meshgrid(np.arange(width), np.arange(height))
    focal_length = max(width, height) * 0.9  
    cx, cy = width / 2.0, height / 2.0

    X = (x_indices - cx) * z / focal_length
    Y = (y_indices - cy) * z / focal_length
    Z = z

    points = np.stack((X, Y, Z), axis=-1).reshape(-1, 3)
    colors = (np.array(img).astype(np.float32) / 255.0).reshape(-1, 3)

    # 🛠️ Set to 100 to keep the trees, sky, and full surroundings!
    # (If you only want the road again, change this 100 back to 82)
    road_mask = Z.reshape(-1) < np.percentile(Z, 100)
    points = points[road_mask]
    colors = colors[road_mask]

    output_ply = base_dir / "road_model.ply"
    print("💾 Manually writing PLY file (bypassing Open3D to prevent SegFault)...")
    
    # Write the PLY file directly using pure Python to prevent crashes
    colors_uint8 = (colors * 255.0).clip(0, 255).astype(np.uint8)
    
    with open(output_ply, 'w') as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for i in range(len(points)):
            f.write(f"{points[i,0]:.5f} {points[i,1]:.5f} {points[i,2]:.5f} "
                    f"{colors_uint8[i,0]} {colors_uint8[i,1]} {colors_uint8[i,2]}\n")
            
    print(f"✅ SUCCESS! Dense point cloud built with {len(points)} points: {output_ply.name}")

if __name__ == "__main__":
    run_ai_depth_engine()
