"""Frame ingestion module for the road defect segmentation pipeline.

Accepts image files (JPEG, PNG) and video files (MP4, MOV), validates
resolutions, and yields individual frames as numpy arrays with metadata.

Validates: Requirements 1.1, 1.2, 1.3, 1.4, 1.5, 1.6
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Generator, Tuple

import cv2
import numpy as np

from src.pipeline.logger import get_logger
from src.pipeline.models import FrameMetadata, PipelineConfig


# Supported file extensions (case-insensitive)
SUPPORTED_IMAGE_EXTENSIONS = {".jpeg", ".jpg", ".png"}
SUPPORTED_VIDEO_EXTENSIONS = {".mp4", ".mov"}
SUPPORTED_EXTENSIONS = SUPPORTED_IMAGE_EXTENSIONS | SUPPORTED_VIDEO_EXTENSIONS

# Resolution constraints
MIN_WIDTH = 64
MIN_HEIGHT = 64
MAX_WIDTH = 5312
MAX_HEIGHT = 2988


class UnsupportedFormatError(Exception):
    """Raised when an input file has an unsupported format."""

    pass


class FrameIngester:
    """Ingests image and video files, yielding validated frames with metadata.

    Supports JPEG, PNG image files and MP4, MOV video files. Validates
    frame resolutions against configured bounds and skips corrupt or
    unreadable frames with logged warnings.

    Args:
        config: Pipeline configuration containing frame_extraction_rate.
    """

    def __init__(self, config: PipelineConfig) -> None:
        self._config = config
        self._logger = get_logger("FrameIngester")

    def ingest(
        self, input_path: str
    ) -> Generator[Tuple[np.ndarray, FrameMetadata], None, None]:
        """Ingest an input file and yield validated frames with metadata.

        For image files, reads and yields the single frame. For video files,
        extracts frames at the configured frame rate.

        Args:
            input_path: Path to an image or video file.

        Yields:
            Tuples of (frame_array, metadata) for each valid frame.
            Frame arrays are RGB uint8 numpy arrays of shape (H, W, 3).

        Raises:
            UnsupportedFormatError: If the file format is not supported.
        """
        if self._is_image(input_path):
            yield from self._ingest_image(input_path)
        elif self._is_video(input_path):
            yield from self._extract_video_frames(
                input_path, self._config.frame_extraction_rate
            )
        else:
            supported_formats = "JPEG, PNG for images; MP4, MOV for video"
            error_msg = (
                f"Unsupported file format: '{input_path}'. "
                f"Supported formats: {supported_formats}"
            )
            self._logger.error(error_msg)
            raise UnsupportedFormatError(error_msg)

    def _is_image(self, path: str) -> bool:
        """Check if the file path has a supported image extension.

        Args:
            path: File path to check.

        Returns:
            True if the extension is JPEG, JPG, or PNG (case-insensitive).
        """
        ext = os.path.splitext(path)[1].lower()
        return ext in SUPPORTED_IMAGE_EXTENSIONS

    def _is_video(self, path: str) -> bool:
        """Check if the file path has a supported video extension.

        Args:
            path: File path to check.

        Returns:
            True if the extension is MP4 or MOV (case-insensitive).
        """
        ext = os.path.splitext(path)[1].lower()
        return ext in SUPPORTED_VIDEO_EXTENSIONS

    def _ingest_image(
        self, path: str
    ) -> Generator[Tuple[np.ndarray, FrameMetadata], None, None]:
        """Read and yield a single image file.

        Reads the image using OpenCV, converts BGR to RGB, validates
        resolution, and yields the frame with metadata.

        Args:
            path: Path to the image file.

        Yields:
            A single (frame_array, metadata) tuple if the image is valid.
        """
        frame = cv2.imread(path)

        if frame is None:
            self._logger.warning(
                f"Corrupt or unreadable image file: '{path}'. Skipping."
            )
            return

        # Convert BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if not self._validate_resolution(frame):
            height, width = frame.shape[:2]
            self._logger.warning(
                f"Frame resolution {width}x{height} from '{path}' is outside "
                f"acceptable range ({MIN_WIDTH}x{MIN_HEIGHT} to "
                f"{MAX_WIDTH}x{MAX_HEIGHT}). Skipping."
            )
            return

        height, width = frame.shape[:2]
        frame_id = os.path.splitext(os.path.basename(path))[0]
        timestamp = datetime.now(timezone.utc).isoformat()

        metadata = FrameMetadata(
            frame_id=frame_id,
            source_path=path,
            timestamp=timestamp,
            width=width,
            height=height,
        )

        yield (frame, metadata)

    def _extract_video_frames(
        self, path: str, fps: float
    ) -> Generator[Tuple[np.ndarray, FrameMetadata], None, None]:
        """Extract frames from a video file at the specified frame rate.

        Uses OpenCV VideoCapture to read the video and extract frames
        at intervals determined by the configured extraction rate.

        Args:
            path: Path to the video file.
            fps: Desired frame extraction rate in frames per second.

        Yields:
            Tuples of (frame_array, metadata) for each valid extracted frame.
        """
        cap = cv2.VideoCapture(path)

        if not cap.isOpened():
            self._logger.error(
                f"Failed to open video file: '{path}'. Skipping."
            )
            return

        video_fps = cap.get(cv2.CAP_PROP_FPS)
        if video_fps <= 0:
            self._logger.error(
                f"Could not determine FPS for video: '{path}'. Skipping."
            )
            cap.release()
            return

        # Calculate frame interval based on extraction rate
        frame_interval = video_fps / fps
        video_name = os.path.splitext(os.path.basename(path))[0]

        frame_index = 0
        last_successful_index = -1
        extracted_count = 0

        try:
            while True:
                ret, frame = cap.read()

                if not ret:
                    break

                # Determine if this frame should be extracted
                # Extract frame when frame_index crosses the next extraction point
                target_frame = int(extracted_count * frame_interval)

                if frame_index < target_frame:
                    frame_index += 1
                    continue

                # Convert BGR to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                if not self._validate_resolution(frame):
                    height, width = frame.shape[:2]
                    self._logger.warning(
                        f"Frame {frame_index} from '{path}' has resolution "
                        f"{width}x{height} outside acceptable range. Skipping."
                    )
                    frame_index += 1
                    extracted_count += 1
                    continue

                height, width = frame.shape[:2]
                frame_id = f"{video_name}_frame_{extracted_count:04d}"
                timestamp = datetime.now(timezone.utc).isoformat()

                metadata = FrameMetadata(
                    frame_id=frame_id,
                    source_path=path,
                    timestamp=timestamp,
                    width=width,
                    height=height,
                )

                last_successful_index = extracted_count
                extracted_count += 1
                frame_index += 1

                yield (frame, metadata)

        except Exception as e:
            self._logger.error(
                f"Video extraction failed for '{path}' at frame index "
                f"{frame_index}. Last successfully extracted frame: "
                f"{last_successful_index}. Error: {e}"
            )
        finally:
            cap.release()

    def _validate_resolution(self, frame: np.ndarray) -> bool:
        """Validate that a frame's resolution is within acceptable bounds.

        Acceptable range: 64x64 to 5312x2988 pixels.

        Args:
            frame: Numpy array of shape (H, W, ...).

        Returns:
            True if the resolution is within bounds, False otherwise.
        """
        height, width = frame.shape[:2]
        return (
            MIN_WIDTH <= width <= MAX_WIDTH
            and MIN_HEIGHT <= height <= MAX_HEIGHT
        )
