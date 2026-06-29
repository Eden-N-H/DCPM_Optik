import os
import glob
from concurrent.futures import ProcessPoolExecutor
import cv2
import numpy as np

# Try to import Ultralytics YOLO; if using raw TorchScript, see alternative note below
try:
    from ultralytics import YOLO
except ImportError:
    print("Please install ultralytics: pip install ultralytics")

# Define global canvas constants for the perspective warp output
OUT_W = 1020
OUT_H = 1020

def warp_single_image(args):
    """
    Worker function executed in parallel across CPU cores.
    Handles the heavy matrix transformation and perspective warping.
    """
    image_path, temp_dir = args
    try:
        img = cv2.imread(image_path)
        if img is None:
            return None
        
        height, width = img.shape[:2]

        # Source Points (The Trapezoid from your camera setup)
        src_points = np.float32([
            [int(width * 0.15), int(height * 0.95)],  # Bottom Left
            [int(width * 0.40), int(height * 0.50)],  # Top Left
            [int(width * 0.60), int(height * 0.50)],  # Top Right
            [int(width * 0.85), int(height * 0.95)]   # Bottom Right
        ])

        dst_points = np.float32([
            [0, OUT_H], [0, 0], [OUT_W, 0], [OUT_W, OUT_H]
        ])

        # Apply Perspective Transform (Bird's-Eye View)
        matrix = cv2.getPerspectiveTransform(src_points, dst_points)
        corrected_image = cv2.warpPerspective(img, matrix, (OUT_W, OUT_H))

        # Save warped frame to a temporary caching directory for the AI phase
        base_name = os.path.basename(image_path)
        temp_path = os.path.join(temp_dir, f"warped_{base_name}")
        cv2.imwrite(temp_path, corrected_image)
        
        return (image_path, temp_path)

    except Exception as e:
        print(f"Error warping {os.path.basename(image_path)}: {str(e)}")
        return None


def run_ai_inference(warped_mappings, output_folder, model_rmcc_path, model_unsealed_path, road_lane_width_mm):
    """
    Sequentially passes warped frames through the dual neural networks.
    Calculates metric areas and saves the combined diagnostic results.
    """
    print("\nInitializing AI Neural Networks into memory...")
    # Load your trained PyTorch models
    model_rmcc = YOLO(model_rmcc_path)
    model_unsealed = YOLO(model_unsealed_path)

    # Metric scaling constants
    pixel_scale_mm = road_lane_width_mm / OUT_W
    sq_mm_per_pixel = pixel_scale_mm ** 2

    print(f"Commencing neural network inference across {len(warped_mappings)} frames...")

    for orig_path, warped_path in warped_mappings:
        base_name = os.path.basename(orig_path)
        img = cv2.imread(warped_path)
        if img is None:
            continue

        output_visual = img.copy()
        defect_id = 0

        # Run inference using both models on the current frame
        # stream=True handles memory efficiently
        results_rmcc = model_rmcc(warped_path, conf=0.25, verbose=False)
        results_unsealed = model_unsealed(warped_path, conf=0.25, verbose=False)

        # Combine results to process bounding boxes sequentially
        all_predictions = []
        for r in results_rmcc + results_unsealed:
            if r.boxes:
                for box in r.boxes:
                    # Extract coordinates, confidence values, and class names
                    xyxy = box.xyxy[0].cpu().numpy().astype(int) # [xmin, ymin, xmax, ymax]
                    conf = float(box.conf[0].cpu().numpy())
                    cls_id = int(box.cls[0].cpu().numpy())
                    cls_name = r.names[cls_id]
                    all_predictions.append((xyxy, conf, cls_name))

        for xyxy, conf, cls_name in all_predictions:
            defect_id += 1
            xmin, ymin, xmax, ymax = xyxy

            # Compute physical metric area based on the bounding box dimension boundaries
            box_width_pixels = xmax - xmin
            box_height_pixels = ymax - ymin
            pixel_area = box_width_pixels * box_height_pixels
            
            area_mm2 = pixel_area * sq_mm_per_pixel
            area_m2 = area_mm2 / 1_000_000.0

            # Assign distinctive colors based on model asset classing
            color = (0, 0, 255) if "pothole" in cls_name.lower() else (0, 255, 255) # Red vs Yellow

            # Draw AI Bounding Box
            cv2.rectangle(output_visual, (xmin, ymin), (xmax, ymax), color, 3)
            
            # Label with Class Name, Confidence score, and calculated Metric Area
            label = f"{cls_name} ({conf:.2f}) {area_m2:.3f} m2"
            cv2.putText(output_visual, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Save finished diagnostic visualization image
        output_path = os.path.join(output_folder, f"measured_{base_name}")
        cv2.imwrite(output_path, output_visual)
        print(f"[{base_name}] Model Inference Complete. Detected Defects: {defect_id}")

        # Clean up temporary warped cache file
        if os.path.exists(warped_path):
            os.remove(warped_path)


def main_pipeline(input_folder, output_folder):
    # Absolute paths to your custom models and tracking directories
    MODEL_RMCC = r"C:\OPTIK\ExtractEXIF\models\RMCC.pt"
    MODEL_UNSEALED = r"C:\OPTIK\ExtractEXIF\models\Unsealed Damage.pt"
    TEMP_DIR = r"C:\OPTIK\ExtractEXIF\Temp_Cache"
    
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(TEMP_DIR, exist_ok=True)

    valid_extensions = ('*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG')
    image_paths = []
    for ext in valid_extensions:
        image_paths.extend(glob.glob(os.path.join(input_folder, ext)))

    if not image_paths:
        print("No images located.")
        return

    # --- STEP 1: PARALLEL PERSPECTIVE WARPING (CPU CORES) ---
    print(f"Spinning up CPU pool to warp {len(image_paths)} images...")
    tasks = [(path, TEMP_DIR) for path in image_paths]
    
    warped_mappings = []
    with ProcessPoolExecutor() as executor:
        results = executor.map(warp_single_image, tasks)
        for res in results:
            if res is not None:
                warped_mappings.append(res)

    # --- STEP 2: SEQUENTIAL/BATCHED MODEL INFERENCE (GPU/AI) ---
    if warped_mappings:
        run_ai_inference(warped_mappings, output_folder, MODEL_RMCC, MODEL_UNSEALED, road_lane_width_mm=3500)
    
    # Remove empty temp cache directory
    try:
        os.rmdir(TEMP_DIR)
    except Exception:
        pass
    print("\n--- Project Management System Pipeline Execution Finished! ---")


if __name__ == "__main__":
    INPUT_DIR = r"C:\OPTIK\ExtractEXIF\Images"
    OUTPUT_DIR = r"C:\OPTIK\ExtractEXIF\Measured_Outputs"
    main_pipeline(INPUT_DIR, OUTPUT_DIR)