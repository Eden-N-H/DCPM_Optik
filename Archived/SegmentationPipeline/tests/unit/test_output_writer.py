"""Unit tests for the OutputWriter class.

Tests JSON serialization of per-frame results and batch summaries,
COCO RLE mask encoding, and output directory validation.
"""

import json
import os
import sys
import tempfile

import numpy as np
import pytest

from src.pipeline.models import BatchSummary, DefectOutput, FrameMetadata, PipelineConfig
from src.pipeline.output_writer import OutputWriter


@pytest.fixture
def output_dir(tmp_path):
    """Create a writable temporary output directory."""
    return str(tmp_path)


@pytest.fixture
def config(output_dir):
    """Create a PipelineConfig with the temporary output directory."""
    return PipelineConfig(output_directory=output_dir)


@pytest.fixture
def writer(config):
    """Create an OutputWriter instance with valid config."""
    return OutputWriter(config)


@pytest.fixture
def sample_frame_metadata():
    """Create sample FrameMetadata for testing."""
    return FrameMetadata(
        frame_id="video001_frame_0042",
        source_path="/data/video001.mp4",
        timestamp="2024-03-15T10:30:42Z",
        width=5312,
        height=2988,
    )


@pytest.fixture
def sample_defects():
    """Create sample DefectOutput list for testing."""
    return [
        DefectOutput(
            class_label="pothole",
            confidence=0.87,
            bounding_box={"x": 120, "y": 340, "width": 200, "height": 150},
            segmentation={"size": [2988, 5312], "counts": "encoded_rle_string"},
            measurements={
                "area_pixels": 18500,
                "area_cm2": 2450.3,
                "width_cm": 45.2,
                "length_cm": 62.1,
                "width_pixels": 180,
                "length_pixels": 140,
                "severity": "severe",
            },
            review_flag=False,
        ),
    ]


class TestOutputWriterInit:
    """Tests for OutputWriter initialization and directory validation."""

    def test_init_with_valid_directory(self, config):
        """OutputWriter initializes successfully with a valid, writable directory."""
        writer = OutputWriter(config)
        assert writer._output_directory == config.output_directory

    def test_init_with_nonexistent_directory(self, tmp_path):
        """OutputWriter exits with error when directory does not exist."""
        config = PipelineConfig(output_directory=str(tmp_path / "nonexistent"))
        with pytest.raises(SystemExit) as exc_info:
            OutputWriter(config)
        assert exc_info.value.code == 1

    def test_init_with_non_writable_directory(self, tmp_path):
        """OutputWriter exits with error when directory is not writable."""
        read_only_dir = tmp_path / "readonly"
        read_only_dir.mkdir()
        os.chmod(str(read_only_dir), 0o444)

        config = PipelineConfig(output_directory=str(read_only_dir))
        try:
            with pytest.raises(SystemExit) as exc_info:
                OutputWriter(config)
            assert exc_info.value.code == 1
        finally:
            # Restore permissions for cleanup
            os.chmod(str(read_only_dir), 0o755)


class TestWriteFrameResult:
    """Tests for write_frame_result method."""

    def test_writes_frame_json_file(self, writer, output_dir, sample_frame_metadata, sample_defects):
        """write_frame_result creates a JSON file named {frame_id}.json."""
        writer.write_frame_result(sample_frame_metadata, sample_defects)

        expected_path = os.path.join(output_dir, "video001_frame_0042.json")
        assert os.path.exists(expected_path)

    def test_frame_json_contains_required_fields(
        self, writer, output_dir, sample_frame_metadata, sample_defects
    ):
        """Per-frame JSON contains frame_id, timestamp, source_file, and defects."""
        writer.write_frame_result(sample_frame_metadata, sample_defects)

        output_path = os.path.join(output_dir, "video001_frame_0042.json")
        with open(output_path, "r") as f:
            data = json.load(f)

        assert data["frame_id"] == "video001_frame_0042"
        assert data["timestamp"] == "2024-03-15T10:30:42Z"
        assert data["source_file"] == "video001.mp4"
        assert isinstance(data["defects"], list)
        assert len(data["defects"]) == 1

    def test_defect_serialization(
        self, writer, output_dir, sample_frame_metadata, sample_defects
    ):
        """Each defect in output contains class, confidence, bounding_box, segmentation, measurements, review_flag."""
        writer.write_frame_result(sample_frame_metadata, sample_defects)

        output_path = os.path.join(output_dir, "video001_frame_0042.json")
        with open(output_path, "r") as f:
            data = json.load(f)

        defect = data["defects"][0]
        assert defect["class"] == "pothole"
        assert defect["confidence"] == 0.87
        assert defect["bounding_box"] == {"x": 120, "y": 340, "width": 200, "height": 150}
        assert defect["segmentation"] == {"size": [2988, 5312], "counts": "encoded_rle_string"}
        assert defect["measurements"]["area_pixels"] == 18500
        assert defect["review_flag"] is False

    def test_empty_defects_list(self, writer, output_dir, sample_frame_metadata):
        """Frames with zero detections produce JSON with empty defects list."""
        writer.write_frame_result(sample_frame_metadata, [])

        output_path = os.path.join(output_dir, "video001_frame_0042.json")
        with open(output_path, "r") as f:
            data = json.load(f)

        assert data["frame_id"] == "video001_frame_0042"
        assert data["defects"] == []

    def test_json_is_indented(self, writer, output_dir, sample_frame_metadata, sample_defects):
        """Output JSON is formatted with indent=2 for readability."""
        writer.write_frame_result(sample_frame_metadata, sample_defects)

        output_path = os.path.join(output_dir, "video001_frame_0042.json")
        with open(output_path, "r") as f:
            content = f.read()

        # Indented JSON will have lines starting with spaces
        lines = content.strip().split("\n")
        assert any(line.startswith("  ") for line in lines)


class TestWriteBatchSummary:
    """Tests for write_batch_summary method."""

    def test_writes_batch_summary_file(self, writer, output_dir):
        """write_batch_summary creates batch_summary.json in output directory."""
        summary = BatchSummary(
            total_frames_processed=100,
            total_defects_detected=25,
            defects_by_class={"pothole": 15, "longitudinal_crack": 10},
            defects_by_severity={"minor": 10, "moderate": 8, "severe": 7},
            processing_time_seconds=50.5,
            average_time_per_frame_seconds=0.505,
        )

        writer.write_batch_summary(summary)

        expected_path = os.path.join(output_dir, "batch_summary.json")
        assert os.path.exists(expected_path)

    def test_batch_summary_content(self, writer, output_dir):
        """Batch summary JSON contains all required fields."""
        summary = BatchSummary(
            total_frames_processed=1500,
            total_defects_detected=237,
            defects_by_class={
                "pothole": 89,
                "longitudinal_crack": 52,
                "transverse_crack": 41,
                "alligator_cracking": 33,
                "patch_deterioration": 22,
            },
            defects_by_severity={"minor": 112, "moderate": 78, "severe": 47},
            processing_time_seconds=842.5,
            average_time_per_frame_seconds=0.562,
        )

        writer.write_batch_summary(summary)

        output_path = os.path.join(output_dir, "batch_summary.json")
        with open(output_path, "r") as f:
            data = json.load(f)

        assert data["total_frames_processed"] == 1500
        assert data["total_defects_detected"] == 237
        assert data["defects_by_class"]["pothole"] == 89
        assert data["defects_by_severity"]["severe"] == 47
        assert data["processing_time_seconds"] == 842.5
        assert data["average_time_per_frame_seconds"] == 0.562


class TestEncodeMaskRle:
    """Tests for _encode_mask_rle method."""

    def test_rle_encoding_basic(self, writer):
        """_encode_mask_rle produces dict with size and counts fields."""
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[20:40, 30:60] = 1

        result = writer._encode_mask_rle(mask)

        assert "size" in result
        assert "counts" in result
        assert result["size"] == [100, 100]
        assert isinstance(result["counts"], str)

    def test_rle_encoding_roundtrip(self, writer):
        """Encoding a mask to RLE and decoding produces the original mask."""
        from pycocotools import mask as mask_utils

        mask = np.zeros((50, 80), dtype=np.uint8)
        mask[10:30, 20:50] = 1

        rle_dict = writer._encode_mask_rle(mask)

        # Decode back
        rle_for_decode = {
            "size": rle_dict["size"],
            "counts": rle_dict["counts"].encode("utf-8"),
        }
        decoded_mask = mask_utils.decode(rle_for_decode)

        np.testing.assert_array_equal(decoded_mask, mask)

    def test_rle_encoding_empty_mask(self, writer):
        """_encode_mask_rle handles all-zero mask."""
        mask = np.zeros((64, 64), dtype=np.uint8)

        result = writer._encode_mask_rle(mask)

        assert result["size"] == [64, 64]
        assert isinstance(result["counts"], str)

    def test_rle_encoding_full_mask(self, writer):
        """_encode_mask_rle handles all-ones mask."""
        mask = np.ones((64, 64), dtype=np.uint8)

        result = writer._encode_mask_rle(mask)

        assert result["size"] == [64, 64]
        assert isinstance(result["counts"], str)

    def test_rle_counts_is_utf8_string(self, writer):
        """The counts field is a UTF-8 string, not bytes."""
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[25:75, 25:75] = 1

        result = writer._encode_mask_rle(mask)

        assert isinstance(result["counts"], str)
        # Verify it's valid UTF-8 by encoding it
        result["counts"].encode("utf-8")
