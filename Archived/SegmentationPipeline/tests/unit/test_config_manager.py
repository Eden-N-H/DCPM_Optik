"""Unit tests for the ConfigManager class."""

import os
import tempfile
from pathlib import Path

import pytest
import yaml

from src.pipeline.config_manager import ConfigManager
from src.pipeline.models import PipelineConfig


@pytest.fixture
def config_manager():
    """Create a ConfigManager instance."""
    return ConfigManager()


@pytest.fixture
def default_config_path():
    """Path to the project's default config file."""
    return str(
        Path(__file__).parent.parent.parent / "config" / "default_config.yaml"
    )


@pytest.fixture
def tmp_config(tmp_path):
    """Helper to create temporary config files."""

    def _write(content: dict) -> str:
        config_file = tmp_path / "config.yaml"
        with open(config_file, "w") as f:
            yaml.dump(content, f)
        return str(config_file)

    return _write


class TestConfigManagerLoad:
    """Tests for ConfigManager.load() method."""

    def test_load_default_config(self, config_manager, default_config_path):
        """Loading the project's default config should produce a valid PipelineConfig."""
        config = config_manager.load(default_config_path)
        assert isinstance(config, PipelineConfig)
        assert config.frame_extraction_rate == 1.0
        assert config.confidence_threshold == 0.5
        assert config.iou_threshold == 0.45

    def test_load_missing_file_exits(self, config_manager):
        """Loading a nonexistent file should exit with code 1."""
        with pytest.raises(SystemExit) as exc_info:
            config_manager.load("/nonexistent/path/config.yaml")
        assert exc_info.value.code == 1

    def test_load_invalid_yaml_exits(self, config_manager, tmp_path):
        """Loading a file with invalid YAML syntax should exit with code 1."""
        bad_file = tmp_path / "bad.yaml"
        bad_file.write_text("key: [unclosed bracket")
        with pytest.raises(SystemExit) as exc_info:
            config_manager.load(str(bad_file))
        assert exc_info.value.code == 1

    def test_load_empty_yaml_uses_defaults(self, config_manager, tmp_path):
        """An empty YAML file should produce a PipelineConfig with all defaults."""
        empty_file = tmp_path / "empty.yaml"
        empty_file.write_text("")
        config = config_manager.load(str(empty_file))
        assert config.frame_extraction_rate == 1.0
        assert config.confidence_threshold == 0.5
        assert config.iou_threshold == 0.45
        assert config.max_detections == 50
        assert config.log_level == "INFO"
        assert config.output_directory == "output"

    def test_load_partial_config_applies_defaults(self, config_manager, tmp_config):
        """A config with only some fields should use defaults for unspecified ones."""
        partial = {
            "detection": {"confidence_threshold": 0.8},
            "output": {"directory": "/custom/output"},
        }
        config = config_manager.load(tmp_config(partial))
        assert config.confidence_threshold == 0.8
        assert config.output_directory == "/custom/output"
        # Defaults for unspecified
        assert config.frame_extraction_rate == 1.0
        assert config.iou_threshold == 0.45
        assert config.max_detections == 50

    def test_load_maps_nested_yaml_to_flat_config(self, config_manager, tmp_config):
        """Nested YAML keys should map correctly to flat PipelineConfig fields."""
        full = {
            "pipeline": {"frame_extraction_rate": 5.0},
            "preprocessing": {
                "clip_limit": 3.0,
                "tile_grid_size": [16, 16],
                "distortion_coefficients": [0.1, 0.2, 0.01, 0.02, 0.001],
                "camera_matrix": [[1000, 0, 500], [0, 1000, 400], [0, 0, 1]],
            },
            "detection": {
                "model_path": "custom/model.pt",
                "confidence_threshold": 0.7,
                "iou_threshold": 0.5,
                "max_detections": 100,
            },
            "segmentation": {
                "checkpoint_path": "custom/sam2.pth",
                "model_cfg": "custom_cfg.yaml",
            },
            "verification": {"min_area_ratio": 0.1, "max_area_ratio": 0.9},
            "measurement": {"camera_height_cm": 150.0, "focal_length_px": 2000.0},
            "output": {"directory": "/results"},
            "logging": {"level": "DEBUG", "max_consecutive_failures": 5},
        }
        config = config_manager.load(tmp_config(full))
        assert config.frame_extraction_rate == 5.0
        assert config.clip_limit == 3.0
        assert config.tile_grid_size == (16, 16)
        assert config.distortion_coefficients == [0.1, 0.2, 0.01, 0.02, 0.001]
        assert config.camera_matrix == [[1000, 0, 500], [0, 1000, 400], [0, 0, 1]]
        assert config.yolo_model_path == "custom/model.pt"
        assert config.confidence_threshold == 0.7
        assert config.iou_threshold == 0.5
        assert config.max_detections == 100
        assert config.sam2_checkpoint_path == "custom/sam2.pth"
        assert config.sam2_model_cfg == "custom_cfg.yaml"
        assert config.min_area_ratio == 0.1
        assert config.max_area_ratio == 0.9
        assert config.camera_height_cm == 150.0
        assert config.focal_length_px == 2000.0
        assert config.output_directory == "/results"
        assert config.log_level == "DEBUG"
        assert config.max_consecutive_failures == 5

    def test_load_null_values_map_to_none(self, config_manager, tmp_config):
        """Explicit null values in YAML should map to None in PipelineConfig."""
        data = {
            "preprocessing": {
                "distortion_coefficients": None,
                "camera_matrix": None,
            },
            "measurement": {
                "camera_height_cm": None,
                "focal_length_px": None,
            },
        }
        config = config_manager.load(tmp_config(data))
        assert config.distortion_coefficients is None
        assert config.camera_matrix is None
        assert config.camera_height_cm is None
        assert config.focal_length_px is None


class TestConfigManagerValidate:
    """Tests for ConfigManager.validate() method."""

    def test_validate_default_config_no_errors(self, config_manager):
        """A default PipelineConfig should have no validation errors."""
        config = PipelineConfig()
        errors = config_manager.validate(config)
        assert errors == []

    def test_validate_reports_all_errors(self, config_manager):
        """Validation should report ALL out-of-range parameters, not just the first."""
        config = PipelineConfig(
            frame_extraction_rate=50.0,  # out of range [0.1, 30]
            confidence_threshold=1.5,  # out of range [0.0, 1.0]
            iou_threshold=-0.1,  # out of range [0.0, 1.0]
            log_level="INVALID",  # not a valid level
        )
        errors = config_manager.validate(config)
        assert len(errors) == 4
        assert any("frame_extraction_rate" in e for e in errors)
        assert any("confidence_threshold" in e for e in errors)
        assert any("iou_threshold" in e for e in errors)
        assert any("log_level" in e for e in errors)

    def test_validate_confidence_threshold_boundaries(self, config_manager):
        """Boundary values for confidence_threshold should be valid."""
        # Valid boundaries
        config_low = PipelineConfig(confidence_threshold=0.0)
        config_high = PipelineConfig(confidence_threshold=1.0)
        assert config_manager.validate(config_low) == []
        assert config_manager.validate(config_high) == []

        # Invalid
        config_over = PipelineConfig(confidence_threshold=1.01)
        config_under = PipelineConfig(confidence_threshold=-0.01)
        assert len(config_manager.validate(config_over)) == 1
        assert len(config_manager.validate(config_under)) == 1

    def test_validate_frame_extraction_rate_boundaries(self, config_manager):
        """Boundary values for frame_extraction_rate should be valid."""
        config_low = PipelineConfig(frame_extraction_rate=0.1)
        config_high = PipelineConfig(frame_extraction_rate=30.0)
        assert config_manager.validate(config_low) == []
        assert config_manager.validate(config_high) == []

        config_under = PipelineConfig(frame_extraction_rate=0.05)
        config_over = PipelineConfig(frame_extraction_rate=31.0)
        assert len(config_manager.validate(config_under)) == 1
        assert len(config_manager.validate(config_over)) == 1

    def test_validate_max_detections_must_be_positive(self, config_manager):
        """max_detections must be >= 1."""
        config = PipelineConfig(max_detections=0)
        errors = config_manager.validate(config)
        assert any("max_detections" in e for e in errors)

    def test_validate_max_consecutive_failures_must_be_positive(self, config_manager):
        """max_consecutive_failures must be >= 1."""
        config = PipelineConfig(max_consecutive_failures=0)
        errors = config_manager.validate(config)
        assert any("max_consecutive_failures" in e for e in errors)

    def test_validate_camera_height_must_be_positive(self, config_manager):
        """camera_height_cm when provided must be positive."""
        config = PipelineConfig(camera_height_cm=-10.0)
        errors = config_manager.validate(config)
        assert any("camera_height_cm" in e for e in errors)

    def test_validate_focal_length_must_be_positive(self, config_manager):
        """focal_length_px when provided must be positive."""
        config = PipelineConfig(focal_length_px=0.0)
        errors = config_manager.validate(config)
        assert any("focal_length_px" in e for e in errors)

    def test_validate_distortion_coefficients_length(self, config_manager):
        """distortion_coefficients must have exactly 5 values."""
        config = PipelineConfig(distortion_coefficients=[0.1, 0.2, 0.01])
        errors = config_manager.validate(config)
        assert any("distortion_coefficients" in e for e in errors)

    def test_validate_camera_matrix_shape(self, config_manager):
        """camera_matrix must be a 3x3 matrix."""
        config = PipelineConfig(camera_matrix=[[1, 0], [0, 1]])
        errors = config_manager.validate(config)
        assert any("camera_matrix" in e for e in errors)

    def test_validate_invalid_log_level(self, config_manager):
        """Invalid log level should produce an error."""
        config = PipelineConfig(log_level="TRACE")
        errors = config_manager.validate(config)
        assert any("log_level" in e for e in errors)


class TestConfigManagerParseArgs:
    """Tests for ConfigManager.parse_args() method (CLI argument parsing)."""

    def test_parse_args_with_config_flag(self, config_manager):
        """--config flag should be parsed correctly."""
        result = config_manager.parse_args(["--config", "path/to/config.yaml"])
        assert result == "path/to/config.yaml"

    def test_parse_args_missing_config_exits(self, config_manager):
        """Missing --config argument should exit."""
        with pytest.raises(SystemExit) as exc_info:
            config_manager.parse_args([])
        assert exc_info.value.code == 2  # argparse exits with code 2

    def test_parse_args_config_without_value_exits(self, config_manager):
        """--config without a value should exit."""
        with pytest.raises(SystemExit) as exc_info:
            config_manager.parse_args(["--config"])
        assert exc_info.value.code == 2
