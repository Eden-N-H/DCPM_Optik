"""
Step 3: Validate manifest.csv and prepare homography input.

This script:
1. Reads the manifest.csv created from frame + metadata extraction.
2. Checks whether each extracted frame image exists.
3. Reads image width and height.
4. Calculates a blur score using OpenCV.
5. Checks important metadata fields such as GPS, camera height, pitch, roll and yaw.
6. Creates:
   - validation_manifest.csv
   - homography_input_manifest.csv
   - validation_report.txt
   - validation_table.html

Example PowerShell command:

python .\\Data_pipelinine\\Validation\\validate_manifest.py --manifest ".\\Data_pipelinine\\output\\metadata\\manifest.csv" --blur-threshold 100
"""

from __future__ import annotations

import argparse
import csv
import html
from pathlib import Path
from typing import Any

try:
    import cv2
except ImportError as exc:
    raise SystemExit(
        "OpenCV is required for image validation.\n"
        "Install it with:\n"
        "pip install opencv-python"
    ) from exc


REQUIRED_INPUT_FIELDS = [
    "frame_id",
    "image_path",
    "timestamp",
    "latitude",
    "longitude",
    "camera_height_m",
    "pitch",
    "roll",
    "yaw",
    "source_video",
    "source_time_sec",
    "altitude_m",
    "status",
]


OUTPUT_FIELDS = [
    "frame_id",
    "image_path",
    "timestamp",
    "latitude",
    "longitude",
    "camera_height_m",
    "pitch",
    "roll",
    "yaw",
    "source_video",
    "source_time_sec",
    "altitude_m",
    "image_width",
    "image_height",
    "blur_score",
    "missing_metadata",
    "homography_status",
]


def find_project_root(script_path: Path) -> Path:
    """
    Find the main project folder.

    This searches upwards until it finds GPMF_Extraction or Data_pipelinine.
    """
    for parent in script_path.parents:
        if (parent / "GPMF_Extraction").exists() or (parent / "Data_pipelinine").exists():
            return parent

    return script_path.parents[2]


def read_manifest(manifest_path: Path) -> list[dict[str, str]]:
    """Read manifest.csv as a list of dictionaries."""
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest file not found: {manifest_path}")

    with manifest_path.open("r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        rows = list(reader)

    if not rows:
        raise RuntimeError("manifest.csv is empty.")

    return rows


def is_float(value: Any) -> bool:
    """Return True if the value can be converted into a float."""
    try:
        float(value)
        return True
    except (TypeError, ValueError):
        return False


def resolve_image_path(image_path_value: str, output_root: Path, project_root: Path) -> Path:
    """
    Resolve image path from manifest.

    In your manifest, image_path is usually relative to:
    Data_pipelinine/output

    Example:
    frames/hero5/hero5_frame_000001.jpg
    """
    image_path = Path(image_path_value)

    if image_path.is_absolute():
        return image_path

    candidate_from_output = output_root / image_path
    if candidate_from_output.exists():
        return candidate_from_output

    candidate_from_project = project_root / image_path
    if candidate_from_project.exists():
        return candidate_from_project

    return candidate_from_output


def calculate_blur_score(image) -> float:
    """
    Calculate blur score using variance of Laplacian.

    Higher score = sharper image.
    Lower score = blurrier image.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    score = cv2.Laplacian(gray, cv2.CV_64F).var()
    return float(score)


def validate_row(
    row: dict[str, str],
    output_root: Path,
    project_root: Path,
    blur_threshold: float,
) -> dict[str, str]:
    """Validate one manifest row and return an updated row."""
    validated = {}

    for field in REQUIRED_INPUT_FIELDS:
        validated[field] = row.get(field, "").strip()

    image_path_value = validated.get("image_path", "")
    image_file_path = resolve_image_path(image_path_value, output_root, project_root)

    missing_metadata = []

    image_width = ""
    image_height = ""
    blur_score = ""

    image_exists = image_file_path.exists()

    if not image_path_value:
        homography_status = "invalid_missing_image_path"

    elif not image_exists:
        homography_status = "invalid_image_file_not_found"

    else:
        image = cv2.imread(str(image_file_path))

        if image is None:
            homography_status = "invalid_image_unreadable"

        else:
            height, width = image.shape[:2]
            image_width = str(width)
            image_height = str(height)

            score = calculate_blur_score(image)
            blur_score = f"{score:.2f}"

            if not validated.get("latitude"):
                missing_metadata.append("latitude")

            if not validated.get("longitude"):
                missing_metadata.append("longitude")

            if not validated.get("camera_height_m"):
                missing_metadata.append("camera_height_m")

            if not validated.get("pitch"):
                missing_metadata.append("pitch")

            if not validated.get("roll"):
                missing_metadata.append("roll")

            if not validated.get("yaw"):
                missing_metadata.append("yaw")

            if not is_float(validated.get("camera_height_m")):
                homography_status = "invalid_missing_camera_height"

            elif score < blur_threshold:
                homography_status = "warning_low_quality_blurry"

            elif "pitch" in missing_metadata or "roll" in missing_metadata or "yaw" in missing_metadata:
                homography_status = "ready_for_basic_homography_needs_orientation_data"

            else:
                homography_status = "ready_for_full_homography"

    validated["image_width"] = image_width
    validated["image_height"] = image_height
    validated["blur_score"] = blur_score
    validated["missing_metadata"] = ";".join(missing_metadata)
    validated["homography_status"] = homography_status

    return validated


def write_csv(rows: list[dict[str, str]], output_path: Path, fieldnames: list[str]) -> None:
    """Write rows into a CSV file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def is_ready_for_homography(row: dict[str, str]) -> bool:
    """
    Decide whether a row should be included in homography_input_manifest.csv.

    This includes:
    - ready_for_full_homography
    - ready_for_basic_homography_needs_orientation_data

    It excludes:
    - missing images
    - unreadable images
    - missing camera height
    - blurry frames
    """
    return row.get("homography_status") in {
        "ready_for_full_homography",
        "ready_for_basic_homography_needs_orientation_data",
    }


def write_validation_report(
    validated_rows: list[dict[str, str]],
    homography_rows: list[dict[str, str]],
    report_path: Path,
) -> None:
    """Create a simple validation report text file."""
    total_rows = len(validated_rows)
    ready_rows = len(homography_rows)

    status_counts: dict[str, int] = {}

    for row in validated_rows:
        status = row.get("homography_status", "unknown")
        status_counts[status] = status_counts.get(status, 0) + 1

    lines = []
    lines.append("DCPM Data Pipeline Validation Report")
    lines.append("=" * 42)
    lines.append("")
    lines.append(f"Total frames checked: {total_rows}")
    lines.append(f"Frames ready for homography input: {ready_rows}")
    lines.append(f"Frames excluded: {total_rows - ready_rows}")
    lines.append("")
    lines.append("Status summary:")
    lines.append("-" * 20)

    for status, count in sorted(status_counts.items()):
        lines.append(f"{status}: {count}")

    lines.append("")
    lines.append("Notes:")
    lines.append("- ready_for_full_homography means image, camera height, and orientation data are available.")
    lines.append("- ready_for_basic_homography_needs_orientation_data means image and camera height are available, but pitch/roll/yaw are missing.")
    lines.append("- warning_low_quality_blurry means the blur score is below the selected threshold.")
    lines.append("- invalid rows should be checked before sending data to the homography module.")

    report_path.parent.mkdir(parents=True, exist_ok=True)

    with report_path.open("w", encoding="utf-8") as file:
        file.write("\n".join(lines))


def write_html_table(rows: list[dict[str, str]], output_path: Path) -> None:
    """Create a clean HTML table for validation results."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    header_cells = "".join(f"<th>{html.escape(field)}</th>" for field in OUTPUT_FIELDS)

    table_rows = ""

    for row in rows:
        status = row.get("homography_status", "")
        css_class = "ready" if status.startswith("ready") else "warning" if status.startswith("warning") else "invalid"

        cells = ""
        for field in OUTPUT_FIELDS:
            value = row.get(field, "")
            cells += f"<td>{html.escape(value)}</td>"

        table_rows += f'<tr class="{css_class}">{cells}</tr>\n'

    html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>DCPM Validation Manifest</title>

    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f7f9fc;
        }}

        h1 {{
            color: #1f4e79;
            margin-bottom: 5px;
        }}

        .summary {{
            margin-bottom: 15px;
            padding: 12px;
            background-color: #e8f1fb;
            border-left: 5px solid #1f4e79;
            line-height: 1.5;
        }}

        .table-container {{
            overflow-x: auto;
            border: 1px solid #ccc;
            background-color: white;
        }}

        table {{
            border-collapse: collapse;
            width: 100%;
            min-width: 1500px;
            font-size: 13px;
        }}

        th {{
            background-color: #1f4e79;
            color: white;
            padding: 8px;
            text-align: left;
            position: sticky;
            top: 0;
            z-index: 2;
        }}

        td {{
            border: 1px solid #ddd;
            padding: 7px;
            white-space: nowrap;
        }}

        tr.ready {{
            background-color: #e8f8ed;
        }}

        tr.warning {{
            background-color: #fff7df;
        }}

        tr.invalid {{
            background-color: #fde8e8;
        }}

        tr:hover {{
            background-color: #dbeafe;
        }}
    </style>
</head>

<body>
    <h1>DCPM Validation Manifest</h1>

    <div class="summary">
        <strong>Total records:</strong> {len(rows)}<br>
        <strong>Purpose:</strong> Validates extracted frames before sending them to the homography module.<br>
        <strong>Green:</strong> Ready &nbsp; | &nbsp;
        <strong>Yellow:</strong> Warning &nbsp; | &nbsp;
        <strong>Red:</strong> Invalid
    </div>

    <div class="table-container">
        <table>
            <thead>
                <tr>{header_cells}</tr>
            </thead>
            <tbody>
                {table_rows}
            </tbody>
        </table>
    </div>
</body>
</html>
"""

    with output_path.open("w", encoding="utf-8") as file:
        file.write(html_content)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate manifest.csv and prepare homography_input_manifest.csv."
    )

    parser.add_argument(
        "--manifest",
        default="Data_pipelinine/output/metadata/manifest.csv",
        help="Path to manifest.csv"
    )

    parser.add_argument(
        "--blur-threshold",
        type=float,
        default=100.0,
        help="Minimum blur score required for a frame to be accepted"
    )

    args = parser.parse_args()

    script_path = Path(__file__).resolve()
    project_root = find_project_root(script_path)

    manifest_path = Path(args.manifest)

    if not manifest_path.is_absolute():
        manifest_path = project_root / manifest_path

    # If manifest is:
    # Data_pipelinine/output/metadata/manifest.csv
    # then output_root is:
    # Data_pipelinine/output
    output_root = manifest_path.parent.parent
    metadata_dir = manifest_path.parent

    rows = read_manifest(manifest_path)

    validated_rows = [
        validate_row(
            row=row,
            output_root=output_root,
            project_root=project_root,
            blur_threshold=args.blur_threshold,
        )
        for row in rows
    ]

    homography_rows = [
        row for row in validated_rows
        if is_ready_for_homography(row)
    ]

    validation_manifest_path = metadata_dir / "validation_manifest.csv"
    homography_input_path = metadata_dir / "homography_input_manifest.csv"
    report_path = metadata_dir / "validation_report.txt"
    html_path = metadata_dir / "validation_table.html"

    write_csv(validated_rows, validation_manifest_path, OUTPUT_FIELDS)
    write_csv(homography_rows, homography_input_path, OUTPUT_FIELDS)
    write_validation_report(validated_rows, homography_rows, report_path)
    write_html_table(validated_rows, html_path)

    print()
    print("[Success] Manifest validation completed.")
    print(f"[Done] Validation manifest saved to: {validation_manifest_path}")
    print(f"[Done] Homography input manifest saved to: {homography_input_path}")
    print(f"[Done] Validation report saved to: {report_path}")
    print(f"[Done] Validation HTML table saved to: {html_path}")
    print()
    print(f"Total frames checked: {len(validated_rows)}")
    print(f"Frames ready for homography: {len(homography_rows)}")
    print(f"Frames excluded: {len(validated_rows) - len(homography_rows)}")
    print()
    print("Next step:")
    print("Open homography_input_manifest.csv and give it to the homography module/team member.")

if __name__ == "__main__":
    main()
