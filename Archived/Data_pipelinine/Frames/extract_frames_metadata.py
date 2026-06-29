"""
Step 2: Frame + metadata extraction for DCPM Optik data pipeline.

This script:
1. Accepts a GoPro video file, image file, or image folder.
2. Extracts frames from videos at a chosen time interval.
3. Creates a manifest.csv control file.
4. Creates a manifest_table.html file for cleaner viewing.
5. Attempts to extract GoPro GPMF GPS telemetry from MP4 videos.
6. Leaves pitch, roll, and yaw blank for now.

Example PowerShell command:

python .\\Data_pipelinine\\Frames\\extract_frames_metadata.py --input ".\\GPMF_Extraction\\Test_Videos\\hero5.mp4" --output ".\\Data_pipelinine\\output" --frame-interval-sec 1 --camera-height-m 1.6
"""

from __future__ import annotations

import argparse
import csv
import html
import math
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

try:
    import cv2
except ImportError as exc:
    raise SystemExit(
        "OpenCV is required for video frame extraction.\n"
        "Install it with:\n"
        "pip install opencv-python"
    ) from exc

try:
    from PIL import Image
except ImportError:
    Image = None


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv"}


@dataclass
class FrameRecord:
    frame_id: str
    image_path: str
    timestamp: str
    latitude: str = ""
    longitude: str = ""
    camera_height_m: str = ""
    pitch: str = ""
    roll: str = ""
    yaw: str = ""
    source_video: str = ""
    source_time_sec: str = ""
    altitude_m: str = ""
    status: str = "ready"


def find_project_root(script_path: Path) -> Path:
    """
    Find the main DCPM_Optik project folder.

    This searches upwards until it finds the folder containing GPMF_Extraction.
    """
    for parent in script_path.parents:
        if (parent / "GPMF_Extraction").exists():
            return parent

    return script_path.parents[2]


def format_seconds(seconds: float) -> str:
    """Convert seconds into HH:MM:SS.mmm format."""
    if seconds < 0 or math.isnan(seconds):
        seconds = 0.0

    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60

    return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"


def safe_float(value: Any) -> Optional[float]:
    """Safely convert a value to float."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def add_project_import_paths(project_root: Path) -> None:
    """Allow importing the existing GPMF extractor from the project."""
    gpmf_path = project_root / "GPMF_Extraction"

    if gpmf_path.exists():
        sys.path.insert(0, str(gpmf_path))


def extract_gpmf_streams(video_path: Path, project_root: Path) -> dict[str, list[Any]]:
    """
    Extract GoPro telemetry streams from an MP4 file.

    This function tries to reuse:
    GPMF_Extraction/extract_gpmf.py

    If extraction fails, it returns an empty dictionary so the manifest
    can still be created.
    """
    add_project_import_paths(project_root)

    try:
        from extract_gpmf import extract_streams  # type: ignore
    except Exception as exc:
        print(f"[Warning] Could not import extract_gpmf.py: {exc}")
        return {}

    try:
        return extract_streams(str(video_path))
    except Exception as exc:
        print(f"[Warning] Could not extract GPMF telemetry: {exc}")
        return {}


def nearest_stream_row(
    stream_rows: list[Any],
    time_sec: float,
    duration_sec: float
) -> Optional[Any]:
    """
    Select the nearest telemetry row for a video timestamp.

    This starter version aligns telemetry by position across the video duration.
    """
    if not stream_rows or duration_sec <= 0:
        return None

    ratio = max(0.0, min(1.0, time_sec / duration_sec))
    index = round(ratio * (len(stream_rows) - 1))

    return stream_rows[index]


def gps_from_streams(
    streams: dict[str, list[Any]],
    time_sec: float,
    duration_sec: float
) -> tuple[str, str, str]:
    """
    Return latitude, longitude, and altitude from GoPro GPS5 stream.

    GPS5 usually contains rows like:
    [latitude, longitude, altitude, speed_2d, speed_3d]
    """
    row = nearest_stream_row(streams.get("GPS5", []), time_sec, duration_sec)

    if isinstance(row, (list, tuple)) and len(row) >= 2:
        lat = safe_float(row[0])
        lon = safe_float(row[1])
        alt = safe_float(row[2]) if len(row) >= 3 else None

        return (
            f"{lat:.7f}" if lat is not None else "",
            f"{lon:.7f}" if lon is not None else "",
            f"{alt:.3f}" if alt is not None else "",
        )

    return "", "", ""


def make_relative_path(path: Path, base: Path) -> str:
    """Return a clean relative path if possible."""
    try:
        return str(path.relative_to(base))
    except ValueError:
        return str(path)


def extract_video_frames(
    video_path: Path,
    output_dir: Path,
    project_root: Path,
    frame_interval_sec: float,
    camera_height_m: float,
) -> list[FrameRecord]:
    """Extract video frames and create one manifest record per extracted frame."""
    frames_dir = output_dir / "frames" / video_path.stem
    frames_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if not fps or fps <= 0:
        fps = 30.0

    duration_sec = total_frames / fps if total_frames > 0 else 0.0
    step_frames = max(1, int(round(frame_interval_sec * fps)))

    streams = extract_gpmf_streams(video_path, project_root)

    if streams:
        print(f"[Info] GPMF streams found: {', '.join(sorted(streams.keys()))}")
    else:
        print("[Info] No GPMF streams found. Manifest will leave GPS fields blank.")

    records: list[FrameRecord] = []
    frame_number = 0
    saved_index = 1

    while True:
        success, frame = cap.read()

        if not success:
            break

        if frame_number % step_frames == 0:
            time_sec = frame_number / fps

            frame_id = f"{video_path.stem}_frame_{saved_index:06d}"
            frame_filename = f"{frame_id}.jpg"
            frame_path = frames_dir / frame_filename

            cv2.imwrite(str(frame_path), frame)

            lat, lon, alt = gps_from_streams(streams, time_sec, duration_sec)

            records.append(
                FrameRecord(
                    frame_id=frame_id,
                    image_path=make_relative_path(frame_path, output_dir),
                    timestamp=format_seconds(time_sec),
                    latitude=lat,
                    longitude=lon,
                    camera_height_m=f"{camera_height_m:.3f}",
                    pitch="",
                    roll="",
                    yaw="",
                    source_video=make_relative_path(video_path, project_root),
                    source_time_sec=f"{time_sec:.3f}",
                    altitude_m=alt,
                    status="ready",
                )
            )

            saved_index += 1

        frame_number += 1

    cap.release()

    print(f"[Done] Extracted {len(records)} frames from {video_path.name}")

    return records


def decimal_from_gps_exif(value: Any, ref: str) -> str:
    """Convert PIL EXIF GPS rational values into decimal degrees."""
    try:
        degrees = float(value[0])
        minutes = float(value[1])
        seconds = float(value[2])

        decimal = degrees + minutes / 60 + seconds / 3600

        if ref in {"S", "W"}:
            decimal *= -1

        return f"{decimal:.7f}"

    except Exception:
        return ""


def exif_gps_from_image(image_path: Path) -> tuple[str, str]:
    """Try to extract GPS latitude and longitude from image EXIF metadata."""
    if Image is None:
        return "", ""

    try:
        with Image.open(image_path) as img:
            exif = img.getexif()
            gps_ifd = exif.get_ifd(0x8825) if exif else {}

        lat = decimal_from_gps_exif(gps_ifd.get(2), gps_ifd.get(1, "")) if gps_ifd else ""
        lon = decimal_from_gps_exif(gps_ifd.get(4), gps_ifd.get(3, "")) if gps_ifd else ""

        return lat, lon

    except Exception:
        return "", ""


def collect_images(input_path: Path) -> list[Path]:
    """Collect one image file or all image files from a folder."""
    if input_path.is_file() and input_path.suffix.lower() in IMAGE_EXTENSIONS:
        return [input_path]

    if input_path.is_dir():
        return sorted(
            p for p in input_path.rglob("*")
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        )

    return []


def prepare_image_records(
    input_path: Path,
    output_dir: Path,
    camera_height_m: float
) -> list[FrameRecord]:
    """Copy still images into the pipeline image folder and create manifest records."""
    images = collect_images(input_path)

    if not images:
        return []

    images_dir = output_dir / "frames" / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    records: list[FrameRecord] = []

    for index, image_path in enumerate(images, start=1):
        frame_id = f"image_{index:06d}"
        target_path = images_dir / f"{frame_id}{image_path.suffix.lower()}"

        shutil.copy2(image_path, target_path)

        lat, lon = exif_gps_from_image(image_path)

        records.append(
            FrameRecord(
                frame_id=frame_id,
                image_path=make_relative_path(target_path, output_dir),
                timestamp="",
                latitude=lat,
                longitude=lon,
                camera_height_m=f"{camera_height_m:.3f}",
                pitch="",
                roll="",
                yaw="",
                source_video="",
                source_time_sec="",
                altitude_m="",
                status="ready",
            )
        )

    print(f"[Done] Prepared {len(records)} image records")

    return records


def write_manifest(records: list[FrameRecord], output_dir: Path) -> Path:
    """Write manifest.csv to the metadata folder."""
    metadata_dir = output_dir / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = metadata_dir / "manifest.csv"

    fieldnames = [
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

    with manifest_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for record in records:
            writer.writerow(record.__dict__)

    return manifest_path


def write_html_table(records: list[FrameRecord], output_dir: Path) -> Path:
    """Create a clean HTML table version of manifest.csv for easy viewing."""
    metadata_dir = output_dir / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)

    html_path = metadata_dir / "manifest_table.html"

    fieldnames = [
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

    header_cells = "".join(
        f"<th>{html.escape(field)}</th>"
        for field in fieldnames
    )

    table_rows = ""

    for record in records:
        record_dict = record.__dict__
        row_cells = ""

        for field in fieldnames:
            value = str(record_dict.get(field, ""))
            row_cells += f"<td>{html.escape(value)}</td>"

        table_rows += f"<tr>{row_cells}</tr>\n"

    html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>DCPM Data Pipeline Manifest</title>

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
            font-size: 13px;
            min-width: 1300px;
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

        tr:nth-child(even) {{
            background-color: #f2f6fb;
        }}

        tr:hover {{
            background-color: #dbeafe;
        }}

        .ready {{
            color: green;
            font-weight: bold;
        }}
    </style>
</head>

<body>
    <h1>DCPM Data Pipeline Manifest</h1>

    <div class="summary">
        <strong>Total records:</strong> {len(records)}<br>
        <strong>Purpose:</strong> Frame and metadata control file for homography, segmentation, and area/depth calculation.<br>
        <strong>Output files:</strong> manifest.csv and manifest_table.html
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

    with html_path.open("w", encoding="utf-8") as file:
        file.write(html_content)

    return html_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract frames and create manifest.csv for the DCPM data pipeline."
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Path to GoPro video, image file, or image folder"
    )

    parser.add_argument(
        "--output",
        default="Data_pipelinine/output",
        help="Output folder for frames and metadata"
    )

    parser.add_argument(
        "--frame-interval-sec",
        type=float,
        default=1.0,
        help="Seconds between extracted video frames"
    )

    parser.add_argument(
        "--camera-height-m",
        type=float,
        default=1.6,
        help="Camera height above road surface in metres"
    )

    args = parser.parse_args()

    script_path = Path(__file__).resolve()
    project_root = find_project_root(script_path)

    input_path = Path(args.input)
    output_dir = Path(args.output)

    if not input_path.is_absolute():
        input_path = project_root / input_path

    if not output_dir.is_absolute():
        output_dir = project_root / output_dir

    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    suffix = input_path.suffix.lower()

    if input_path.is_file() and suffix in VIDEO_EXTENSIONS:
        records = extract_video_frames(
            video_path=input_path,
            output_dir=output_dir,
            project_root=project_root,
            frame_interval_sec=args.frame_interval_sec,
            camera_height_m=args.camera_height_m,
        )
    else:
        records = prepare_image_records(
            input_path=input_path,
            output_dir=output_dir,
            camera_height_m=args.camera_height_m,
        )

    if not records:
        raise RuntimeError("No frames/images were prepared. Check your input path and file type.")

    manifest_path = write_manifest(records, output_dir)
    html_table_path = write_html_table(records, output_dir)

    print()
    print("[Success] Frame and metadata extraction completed.")
    print(f"[Done] Manifest CSV saved to: {manifest_path}")
    print(f"[Done] HTML table saved to: {html_table_path}")
    print()
    print("Next step:")
    print("1. Open manifest.csv using the CSV Excel Viewer extension.")
    print("2. Open manifest_table.html in your browser for a clean table view.")


if __name__ == "__main__":
    main()