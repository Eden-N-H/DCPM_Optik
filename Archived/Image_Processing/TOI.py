import cv2
import numpy as np

def get_corrected_road_surface(image_path):
    # 1. Load the image
    img = cv2.imread(image_path)
    height, width = img.shape[:2]

    # 2. Define the source points (The Trapezoid on the original image)
    # These match the coordinates from your ROI mask
    src_points = np.float32([
        [int(width * 0.15), int(height * 0.95)],  # Bottom Left
        [int(width * 0.40), int(height * 0.50)],  # Top Left (near horizon)
        [int(width * 0.60), int(height * 0.50)],  # Top Right (near horizon)
        [int(width * 0.85), int(height * 0.95)]   # Bottom Right
    ])

    # 3. Define the destination dimensions for the corrected output
    # We want a clean, rectangular top-down bird's-eye view canvas
    out_w = 800
    out_h = 1000

    # 4. Map the trapezoid points to the corners of the new rectangle
    dst_points = np.float32([
        [0, out_h],          # Bottom Left corner of new image
        [0, 0],              # Top Left corner of new image
        [out_w, 0],          # Top Right corner of new image
        [out_w, out_h]       # Bottom Right corner of new image
    ])

    # 5. Compute the Perspective Transformation Matrix
    matrix = cv2.getPerspectiveTransform(src_points, dst_points)

    # 6. Warp the original image to get the flat, corrected output
    corrected_image = cv2.warpPerspective(img, matrix, (out_w, out_h))

    # Save the corrected view
    cv2.imwrite("corrected_birds_eye_view.jpg", corrected_image)
    print("Perspective correction applied successfully. Output saved as 'corrected_birds_eye_view.jpg'")
    
    return corrected_image, matrix

# Example usage:
if __name__ == "__main__":
    sample_image = r"C:\\OPTIK\\ExtractEXIF\\Images\\Chichester Dam Road - G0052068.JPG"
    corrected_road, transform_matrix = get_corrected_road_surface(sample_image)