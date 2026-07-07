"""Unit tests for the FrameIngester class.

Tests cover image and video ingestion, format validation, resolution
validation, corrupt file handling, and error cases.
"""

import os
import tempfile

import cv2
import numpy as np
import pytest

from src.pipeline.frame_ingester import (
    MAX_HEIGHT,
    MAX_WIDTH,
    MIN_HEIGHT,
    MIN_WIDTH,
    FrameIngester,
    UnsupportedFormatError,
)
from src.pipeline.models import PipelineConfig


@pytest.fixture
def config():
    """Default pipeline config for testing."""
    return PipelineConfig(frame_extraction_rate=1.0)


@pytest.fixture
def ingester(config):
    """FrameIngester instance with default config."""
    return FrameIngester(config)


@pytest.fixture
def temp_dir():
    """Temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


def create_test_image(path: str, width: int = 640, height: int = 480) -> str:
    """Create a test image file at the given path."""
    img = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
    cv2.imwrite(path, img)
    return path


def create_test_video(
    path: str,
    width: int = 640,
    height: int = 480,
    fps: float = 30.0,
    num_frames: int = 30,
) -> str:
    """Create a test video file at the given path."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, fps, (width, height))
    for _ in range(num_frames):
        frame = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
        writer.write(frame)
    writer.release()
    return path


class TestIsImage:
    """Tests for _is_image method."""

    def test_jpeg_extension(self, ingester):
        assert ingester._is_image("photo.jpeg") is True

    def test_jpg_extension(self, ingester):
        assert ingester._is_image("photo.jpg") is True

    def test_png_extension(self, ingester):
        assert ingester._is_image("photo.png") is True

    def test_uppercase_extension(self, ingester):
        assert ingester._is_image("photo.JPEG") is True
        assert ingester._is_image("photo.PNG") is True

    def test_video_extension_not_image(self, ingester):
        assert ingester._is_image("video.mp4") is False

    def test_unsupported_extension(self, ingester):
        assert ingester._is_image("file.bmp") is False


class TestIsVideo:
    """Tests for _is_video method."""

    def test_mp4_extension(self, ingester):
        assert ingester._is_video("video.mp4") is True

    def test_mov_extension(self, ingester):
        assert ingester._is_video("video.mov") is True

    def test_uppercase_extension(self, ingester):
        assert ingester._is_video("video.MP4") is True
        assert ingester._is_video("video.MOV") is True

    def test_image_extension_not_video(self, ingester):
        assert ingester._is_video("photo.jpeg") is False

    def test_unsupported_extension(self, ingester):
        assert ingester._is_video("file.avi") is False


class TestValidateResolution:
    """Tests for _validate_resolution method."""

    def test_valid_minimum_resolution(self, ingester):
        frame = np.zeros((MIN_HEIGHT, MIN_WIDTH, 3), dtype=np.uint8)
        assert ingester._validate_resolution(frame) is True

    def test_valid_maximum_resolution(self, ingester):
        frame = np.zeros((MAX_HEIGHT, MAX_WIDTH, 3), dtype=np.uint8)
        assert ingester._validate_resolution(frame) is True

    def test_valid_typical_resolution(self, ingester):
        frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
        assert ingester._validate_resolution(frame) is True

    def test_too_small_width(self, ingester):
        frame = np.zeros((100, 63, 3), dtype=np.uint8)
        assert ingester._validate_resolution(frame) is False

    def test_too_small_height(self, ingester):
        frame = np.zeros((63, 100, 3), dtype=np.uint8)
        assert ingester._validate_resolution(frame) is False

    def test_too_large_width(self, ingester):
        frame = np.zeros((100, MAX_WIDTH + 1, 3), dtype=np.uint8)
        assert ingester._validate_resolution(frame) is False

    def test_too_large_height(self, ingester):
        frame = np.zeros((MAX_HEIGHT + 1, 100, 3), dtype=np.uint8)
        assert ingester._validate_resolution(frame) is False


class TestIngestImage:
    """Tests for image ingestion via ingest()."""

    def test_ingest_jpeg(self, ingester, temp_dir):
        path = create_test_image(os.path.join(temp_dir, "test.jpeg"))
        results = list(ingester.ingest(path))
        assert len(results) == 1
        frame, metadata = results[0]
        assert frame.shape == (480, 640, 3)
        assert metadata.frame_id == "test"
        assert metadata.source_path == path
        assert metadata.width == 640
        assert metadata.height == 480

    def test_ingest_png(self, ingester, temp_dir):
        path = create_test_image(os.path.join(temp_dir, "test.png"))
        results = list(ingester.ingest(path))
        assert len(results) == 1
        frame, metadata = results[0]
        assert frame.shape == (480, 640, 3)

    def test_ingest_image_returns_rgb(self, ingester, temp_dir):
        """Verify that ingested images are converted from BGR to RGB."""
        path = os.path.join(temp_dir, "blue.png")
        # Create a pure blue image (BGR: 255,0,0)
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        img[:, :, 0] = 255  # Blue channel in BGR
        cv2.imwrite(path, img)

        results = list(ingester.ingest(path))
        frame, _ = results[0]
        # After BGR->RGB conversion, blue should be in channel 2
        assert frame[0, 0, 2] == 255
        assert frame[0, 0, 0] == 0

    def test_ingest_corrupt_image_skipped(self, ingester, temp_dir):
        path = os.path.join(temp_dir, "corrupt.jpeg")
        with open(path, "wb") as f:
            f.write(b"not a valid image")
        results = list(ingester.ingest(path))
        assert len(results) == 0

    def test_ingest_image_invalid_resolution_skipped(self, ingester, temp_dir):
        path = create_test_image(
            os.path.join(temp_dir, "tiny.png"), width=32, height=32
        )
        results = list(ingester.ingest(path))
        assert len(results) == 0

    def test_metadata_has_iso_timestamp(self, ingester, temp_dir):
        path = create_test_image(os.path.join(temp_dir, "test.png"))
        results = list(ingester.ingest(path))
        _, metadata = results[0]
        # ISO 8601 format check - should contain 'T' and timezone info
        assert "T" in metadata.timestamp


class TestIngestVideo:
    """Tests for video ingestion via ingest()."""

    def test_ingest_video_extracts_frames(self, ingester, temp_dir):
        # 30 frames at 30fps = 1 second of video, at 1fps extraction = 1 frame
        path = create_test_video(
            os.path.join(temp_dir, "test.mp4"),
            fps=30.0,
            num_frames=30,
        )
        results = list(ingester.ingest(path))
        assert len(results) == 1

    def test_ingest_video_frame_id_format(self, ingester, temp_dir):
        path = create_test_video(
            os.path.join(temp_dir, "myvideo.mp4"),
            fps=30.0,
            num_frames=60,
        )
        results = list(ingester.ingest(path))
        assert len(results) >= 1
        _, metadata = results[0]
        assert metadata.frame_id == "myvideo_frame_0000"

    def test_ingest_video_configurable_fps(self, temp_dir):
        # Create a 2 second video at 30fps = 60 frames
        path = create_test_video(
            os.path.join(temp_dir, "test.mp4"),
            fps=30.0,
            num_frames=60,
        )
        config = PipelineConfig(frame_extraction_rate=2.0)
        ingester = FrameIngester(config)
        results = list(ingester.ingest(path))
        # 2 seconds * 2 fps = 4 frames expected
        assert len(results) == 4

    def test_ingest_nonexistent_video(self, ingester, temp_dir):
        path = os.path.join(temp_dir, "nonexistent.mp4")
        results = list(ingester.ingest(path))
        assert len(results) == 0

    def test_ingest_video_frames_are_rgb(self, ingester, temp_dir):
        path = create_test_video(
            os.path.join(temp_dir, "test.mp4"),
            fps=30.0,
            num_frames=30,
        )
        results = list(ingester.ingest(path))
        if results:
            frame, _ = results[0]
            assert frame.shape[2] == 3  # 3 channels (RGB)
            assert frame.dtype == np.uint8


class TestUnsupportedFormat:
    """Tests for unsupported file format rejection."""

    def test_unsupported_format_raises_error(self, ingester, temp_dir):
        path = os.path.join(temp_dir, "file.bmp")
        with open(path, "wb") as f:
            f.write(b"data")
        with pytest.raises(UnsupportedFormatError) as exc_info:
            list(ingester.ingest(path))
        error_msg = str(exc_info.value)
        assert "JPEG" in error_msg
        assert "PNG" in error_msg
        assert "MP4" in error_msg
        assert "MOV" in error_msg

    def test_unsupported_format_avi(self, ingester, temp_dir):
        path = os.path.join(temp_dir, "video.avi")
        with open(path, "wb") as f:
            f.write(b"data")
        with pytest.raises(UnsupportedFormatError):
            list(ingester.ingest(path))

    def test_unsupported_format_gif(self, ingester, temp_dir):
        path = os.path.join(temp_dir, "anim.gif")
        with open(path, "wb") as f:
            f.write(b"data")
        with pytest.raises(UnsupportedFormatError):
            list(ingester.ingest(path))
