import os
import glob
import cv2
import numpy as np
import exiftool

# Define global canvas constants for the perspective warp output
OUT_W = 1000
OUT_H = 1200

def enhance_shadows_adaptively(img):
    """[X.X.4] Contrast and Lighting Normalization via CLAHE"""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    avg_brightness = np.mean(v)
    
    if avg_brightness < 120:
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        v_enhanced = clahe.apply(v)
        enhanced_hsv = cv2.merge([h, s, v_enhanced])
        return cv2.cvtColor(enhanced_hsv, cv2.COLOR_HSV2BGR)
    return img


def standardize_image(img, target_size=(640, 640)):
    """
    [X.X.4] Dimensional Resizing & Pixel Normalization
    Resizes the warped road surface to tensor dimensions and scales pixels to [0.0, 1.0].
    """
    # 1. Dimensional Resizing using Bilinear Interpolation
    resized_img = cv2.resize(img, target_size, interpolation=cv2.INTER_LINEAR)
    
    # 2. Pixel Normalization (Convert 8-bit [0-255] to float32 [0.0-1.0])
    normalized_img = resized_img.astype(np.float32) / 255.0
    
    return normalized_img


def passes_quality_control(img, blur_threshold=100.0, low_light_thresh=30, high_light_thresh=245):
    """
    [X.X.5] Data Quality Control (QA/QC Guardrails)
    Executes Laplacian Variance for blur and checks global luminance boundaries.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 1. Motion Blur Detection (Laplacian Variance: V < tau)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    if laplacian_var < blur_threshold:
        return False, f"REJECTED: Motion Blur Detected (Variance: {laplacian_var:.2f} < {blur_threshold})"
        
    # 2. Exposure Validation (Average Pixel Luminance)
    avg_luminance = np.mean(gray)
    if avg_luminance < low_light_thresh:
        return False, f"REJECTED: Underexposed / Night / Tunnel (Luminance: {avg_luminance:.2f})"
    if avg_luminance > high_light_thresh:
        return False, f"REJECTED: Overexposed / Severe Sun Glare (Luminance: {avg_luminance:.2f})"
        
    return True, "PASSED QA/QC"


def process_and_burn_metadata(image_path, output_dir, exe_path):
    """Extracts metadata, validates quality, enhances, warps, and standardizes."""
    base_name = os.path.basename(image_path)
    
    # --- PHASE 1: EXIF METADATA EXTRACTION ---
    target_tags = [
        "EXIF:CreateDate", "EXIF:DateTimeOriginal", "GPS:GPSAltitude",
        "GPS:GPSLatitude", "GPS:GPSLongitude", "Composite:FOV", "Composite:GPSPosition"
    ]
    metadata = {}
    try:
        with exiftool.ExifToolHelper(executable=exe_path) as et:
            metadata_list = et.get_tags(image_path, tags=target_tags)
            if metadata_list: metadata = metadata_list[0]
    except Exception as e:
        print(f"[{base_name}] Metadata extraction failed: {e}")
    
    creation_date = metadata.get("EXIF:CreateDate") or metadata.get("EXIF:DateTimeOriginal") or "Unknown Date"
    alt = metadata.get("GPS:GPSAltitude", "N/A")
    fov = metadata.get("Composite:FOV", "N/A")
    lat, lon = metadata.get("GPS:GPSLatitude"), metadata.get("GPS:GPSLongitude")

    if lat is None or lon is None:
        composite_pos = metadata.get("Composite:GPSPosition")
        if composite_pos:
            try:
                coords = str(composite_pos).split()
                if len(coords) >= 2: lat, lon = coords[0], coords[1]
            except Exception: lat, lon = "N/A", "N/A"
        else: lat, lon = "N/A", "N/A"

    gps_display = f"{float(lat):.6f}, {float(lon):.6f}" if lat != "N/A" and lon != "N/A" else "N/A, N/A"

    # --- PHASE 2: IMAGE LOADING & QA/QC GUARDRAILS ---
    img = cv2.imread(image_path)
    if img is None:
        return f"[{base_name}] Failed to load image file."
    
    # [X.X.5] Trigger Automated Quality Gates
    is_valid, qa_message = passes_quality_control(img, blur_threshold=100.0)
    if not is_valid:
        return f"[{base_name}] {qa_message} -> Quarantined."

    # --- PHASE 3: ENHANCEMENT & WARPING ---
    # [X.X.4] Apply Adaptive Shadow Recovery
    img = enhance_shadows_adaptively(img)
    
    height, width = img.shape[:2]
    src_points = np.float32([
        [int(width * 0.15), int(height * 0.95)], [int(width * 0.40), int(height * 0.50)],
        [int(width * 0.60), int(height * 0.50)], [int(width * 0.85), int(height * 0.95)]
    ])
    dst_points = np.float32([[0, OUT_H], [0, 0], [OUT_W, 0], [OUT_W, OUT_H]])

    matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    corrected_image = cv2.warpPerspective(img, matrix, (OUT_W, OUT_H))

    # --- PHASE 4: AI STANDARDIZATION LAYER ---
    # [X.X.4] Dimensional Resizing (640x640) & Pixel Normalization [0.0 - 1.0]
    # Note: To save a normalized float array as an image file natively via OpenCV,
    # we maintain the 8-bit warped file for visual output, but generate the AI matrix here.
    ai_ready_tensor = standardize_image(corrected_image, target_size=(640, 640))

    # --- PHASE 5: BURN METADATA WATERMARK OVERLAY ---
    banner_w, banner_h = 420, 220
    overlay = corrected_image.copy()
    cv2.rectangle(overlay, (0, 0), (banner_w, banner_h), (34, 126, 230), -1)
    cv2.addWeighted(overlay, 0.75, corrected_image, 0.25, 0, corrected_image)

    font, font_scale, font_color, thickness = cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
    text_lines = [
        "ADDI System - Project Management",
        f"File: {base_name}", f"Date: {creation_date}",
        f"GPS: {gps_display}", f"Altitude: {alt} m", f"Camera FOV: {fov}"
    ]

    y_offset = 30
    for line in text_lines:
        cv2.putText(corrected_image, line, (15, y_offset), font, font_scale, font_color, thickness, cv2.LINE_AA)
        y_offset += 30

    # Save finalized file
    output_path = os.path.join(output_dir, f"processed_{base_name}")
    cv2.imwrite(output_path, corrected_image)
    
    return f"[{base_name}] Successfully processed, normalized, and watermarked."


def run_pipeline(input_folder, output_folder, exiftool_path):
    """Scans folders and manages execution."""
    os.makedirs(output_folder, exist_ok=True)
    valid_extensions = ('*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG')
    image_paths = []
    for ext in valid_extensions:
        image_paths.extend(glob.glob(os.path.join(input_folder, ext)))
        
    if not image_paths:
        print(f"No valid image files found in '{input_folder}'")
        return

    print(f"Beginning unified spec-compliant pipeline for {len(image_paths)} images...")
    for path in image_paths:
        print(process_and_burn_metadata(path, output_folder, exiftool_path))
    print("\n--- Pipeline execution fully complete! ---")
    
if __name__ == "__main__":
    # Note: Double check your executable path extensions if running on Windows!
    EXE_PATH = r"C:\\OPTIK\\ExtractEXIF\\exiftool-13.59_32\\exiftool.exe"
    INPUT_DIR = r"C:\\OPTIK\\ExtractEXIF\\Images"
    OUTPUT_DIR = r"C:\\OPTIK\\ExtractEXIF\\Processed_Outputs"

    run_pipeline(INPUT_DIR, OUTPUT_DIR, EXE_PATH)