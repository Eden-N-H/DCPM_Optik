import cv2
import numpy as np
import csv
import os
from pathlib import Path

# --------------------------------------------------
# INPUTS
# --------------------------------------------------


#Notes: Activate enviroment by running" Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
#.\.venv\Scripts\Activate.ps1
#then run "python batch_red_boundary_area.py" in the terminal

input_folder = "images"
output_folder = "output"

# These are the real-world dimensions represented by EACH top-down image.
# This assumes all images use the same rectified scale.
scale_csv_path = "image_scales.csv"

scale_lookup = {}

with open(scale_csv_path, "r") as file:
    reader = csv.DictReader(file)

    for row in reader:
        image_name = row["image_name"]
        real_width_m = float(row["real_width_m"])
        real_length_m = float(row["real_length_m"])

        scale_lookup[image_name] = {
            "real_width_m": real_width_m,
            "real_length_m": real_length_m
        }

# Ignore tiny red areas/noise
min_contour_area_px = 100

# Accepted image formats
image_extensions = [".png", ".jpg", ".jpeg"]

os.makedirs(output_folder, exist_ok=True)

# --------------------------------------------------
# FIND ALL IMAGES
# --------------------------------------------------

image_files = []

for ext in image_extensions:
    image_files.extend(Path(input_folder).glob(f"*{ext}"))

image_files = sorted(image_files)

if len(image_files) == 0:
    raise FileNotFoundError("No PNG/JPG/JPEG images found inside the images folder.")

print(f"Found {len(image_files)} image(s).")
print("Processing images...")

# --------------------------------------------------
# STORE ALL RESULTS
# --------------------------------------------------

all_results = []

# --------------------------------------------------
# PROCESS EACH IMAGE
# --------------------------------------------------

for image_file in image_files:
    image_path = str(image_file)
    image_name = image_file.stem
    image_filename = image_file.name
    
    if image_filename not in scale_lookup:
        print(f"Skipping {image_filename}: no scale information found in image_scales.csv.")
        continue

    real_width_m = scale_lookup[image_filename]["real_width_m"]
    real_length_m = scale_lookup[image_filename]["real_length_m"]

    print("")
    print(f"Processing: {image_filename}")

    # Create a separate output folder for each image
    image_output_folder = os.path.join(output_folder, image_name)
    os.makedirs(image_output_folder, exist_ok=True)

    # --------------------------------------------------
    # READ IMAGE
    # --------------------------------------------------

    image = cv2.imread(image_path)

    if image is None:
        print(f"Skipping {image_filename}: image could not be read.")
        continue

    height, width = image.shape[:2]

    # --------------------------------------------------
    # DETECT RED BOUNDARY
    # --------------------------------------------------

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Red colour needs two HSV ranges
    lower_red_1 = np.array([0, 70, 50])
    upper_red_1 = np.array([10, 255, 255])

    lower_red_2 = np.array([170, 70, 50])
    upper_red_2 = np.array([180, 255, 255])

    mask_red_1 = cv2.inRange(hsv, lower_red_1, upper_red_1)
    mask_red_2 = cv2.inRange(hsv, lower_red_2, upper_red_2)

    red_boundary_mask = mask_red_1 + mask_red_2

    # --------------------------------------------------
    # CLEAN RED BOUNDARY
    # --------------------------------------------------

    kernel = np.ones((5, 5), np.uint8)

    # Close small gaps in the red boundary
    red_boundary_mask = cv2.morphologyEx(
        red_boundary_mask,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=2
    )

    # Slightly thicken the boundary
    red_boundary_mask = cv2.dilate(
        red_boundary_mask,
        kernel,
        iterations=1
    )

    # --------------------------------------------------
    # FIND RED CONTOURS
    # --------------------------------------------------

    contours, _ = cv2.findContours(
        red_boundary_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    valid_contours = []

    for contour in contours:
        contour_area = cv2.contourArea(contour)

        if contour_area > min_contour_area_px:
            valid_contours.append(contour)

    # --------------------------------------------------
    # HANDLE IMAGE WITH NO VALID POTHOLE
    # --------------------------------------------------

    if len(valid_contours) == 0:
        print(f"No valid pothole boundary found in {image_filename}.")

        all_results.append({
            "unique_pothole_id": f"{image_name}_none",
            "image_name": image_filename,
            "pothole_id": "none",
            "area_px2": 0,
            "area_m2": 0,
            "image_width_px": width,
            "image_height_px": height
        })

        cv2.imwrite(
            os.path.join(image_output_folder, "red_boundary_detected.png"),
            red_boundary_mask
        )

        continue

    # --------------------------------------------------
    # SORT POTHOLES BY SIZE
    # Largest pothole becomes pothole 1
    # --------------------------------------------------

    valid_contours = sorted(
        valid_contours,
        key=cv2.contourArea,
        reverse=True
    )

    # --------------------------------------------------
    # CALCULATE AREA
    # --------------------------------------------------

    m_per_px_x = real_width_m / width
    m_per_px_y = real_length_m / height

    total_area_px2 = 0
    total_area_m2 = 0

    filled_all_potholes_mask = np.zeros((height, width), dtype=np.uint8)
    check_image = image.copy()

    for i, contour in enumerate(valid_contours, start=1):
        unique_pothole_id = f"{image_name}_pothole_{i}"

        individual_mask = np.zeros((height, width), dtype=np.uint8)

        cv2.drawContours(
            individual_mask,
            [contour],
            -1,
            255,
            thickness=cv2.FILLED
        )

        area_px2 = cv2.countNonZero(individual_mask)
        area_m2 = area_px2 * m_per_px_x * m_per_px_y

        print(f"{unique_pothole_id}: {area_m2:.4f} m²")

        total_area_px2 += area_px2
        total_area_m2 += area_m2

        # Add to combined pothole mask
        cv2.drawContours(
            filled_all_potholes_mask,
            [contour],
            -1,
            255,
            thickness=cv2.FILLED
        )

        # Draw detected boundary on checking image
        cv2.drawContours(
            check_image,
            [contour],
            -1,
            (0, 255, 0),
            thickness=2
        )

        # Find label position
        M = cv2.moments(contour)

        if M["m00"] != 0:
            centre_x = int(M["m10"] / M["m00"])
            centre_y = int(M["m01"] / M["m00"])
        else:
            x, y, w, h = cv2.boundingRect(contour)
            centre_x = x + w // 2
            centre_y = y + h // 2

        # Label pothole on checking image
        cv2.putText(
            check_image,
            f"ID: {unique_pothole_id}",
            (centre_x - 80, centre_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2
        )

        cv2.putText(
            check_image,
            f"{area_m2:.3f} m2",
            (centre_x - 80, centre_y + 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2
        )

        # Save individual pothole mask
        cv2.imwrite(
            os.path.join(image_output_folder, f"{unique_pothole_id}_mask.png"),
            individual_mask
        )

        # Save individual result row
        all_results.append({
            "unique_pothole_id": unique_pothole_id,
            "image_name": image_filename,
            "pothole_id": i,
            "area_px2": area_px2,
            "area_m2": area_m2,
            "image_width_px": width,
            "image_height_px": height
        })

    # --------------------------------------------------
    # SAVE TOTAL ROW FOR THIS IMAGE
    # --------------------------------------------------

    all_results.append({
        "unique_pothole_id": f"{image_name}_total",
        "image_name": image_filename,
        "pothole_id": "total",
        "area_px2": total_area_px2,
        "area_m2": total_area_m2,
        "image_width_px": width,
        "image_height_px": height
    })

    # --------------------------------------------------
    # SAVE VISUAL CHECKING OUTPUTS
    # --------------------------------------------------

    cv2.imwrite(
        os.path.join(image_output_folder, "red_boundary_detected.png"),
        red_boundary_mask
    )

    cv2.imwrite(
        os.path.join(image_output_folder, "filled_all_potholes_mask.png"),
        filled_all_potholes_mask
    )

    cv2.imwrite(
        os.path.join(image_output_folder, "check_detected_boundaries.jpg"),
        check_image
    )

    print(f"Detected potholes: {len(valid_contours)}")
    print(f"Total area: {total_area_m2:.4f} m²")

# --------------------------------------------------
# SAVE ALL RESULTS TO ONE CSV
# --------------------------------------------------

csv_path = os.path.join(output_folder, "batch_pothole_area_results.csv")

with open(csv_path, "w", newline="") as file:
    writer = csv.writer(file)

    writer.writerow([
        "unique_pothole_id",
        "image_name",
        "pothole_id",
        "area_px2",
        "area_m2",
        "image_width_px",
        "image_height_px",
        "real_width_m",
        "real_length_m"
    ])

    for result in all_results:
        writer.writerow([
            result["unique_pothole_id"],
            result["image_name"],
            result["pothole_id"],
            result["area_px2"],
            result["area_m2"],
            result["image_width_px"],
            result["image_height_px"],
            result["real_width_m"],
            result["real_length_m"]
        ])

print("")
print("Batch processing complete.")
print("CSV saved to:", csv_path)
print("Visual checking outputs saved inside the output folder.")