"""Configuration manager for the road defect segmentation pipeline.

Handles loading YAML configuration files, mapping nested YAML keys to
flat PipelineConfig fields, applying defaults for missing parameters,
validating all values, and CLI argument parsing for the --config flag.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

from src.pipeline.models import PipelineConfig


class ConfigManager:
    """Load, validate, and provide access to pipeline configuration.

    Responsible for:
    - Parsing CLI arguments for --config flag
    - Loading YAML config files and mapping nested keys to PipelineConfig
    - Applying defaults for missing fields
    - Validating all configuration values against documented ranges
    """

    def load(self, config_path: str) -> PipelineConfig:
        """Read a YAML config file, apply defaults for missing fields.

        Args:
            config_path: Path to the YAML configuration file.

        Returns:
            A PipelineConfig instance with all fields populated.

        Raises:
            SystemExit: If the file is not found or contains invalid YAML.
        """
        path = Path(config_path)

        if not path.exists():
            print(
                f"Error: Configuration file not found: {config_path}",
                file=sys.stderr,
            )
            sys.exit(1)

        try:
            with open(path, "r") as f:
                raw = yaml.safe_load(f)
        except yaml.YAMLError as e:
            print(
                f"Error: Invalid YAML syntax in configuration file '{config_path}': {e}",
                file=sys.stderr,
            )
            sys.exit(1)

        if raw is None:
            raw = {}

        return self._map_yaml_to_config(raw)

    def validate(self, config: PipelineConfig) -> List[str]:
        """Check all configuration values against documented ranges.

        Delegates to PipelineConfig.validate() to check ranges and returns
        all errors found (not just the first).

        Args:
            config: The PipelineConfig instance to validate.

        Returns:
            A list of error messages for invalid parameters. Empty if all valid.
        """
        return config.validate()

    def parse_args(self, args: Optional[List[str]] = None) -> str:
        """Parse CLI arguments for the --config flag.

        Args:
            args: Optional list of arguments (defaults to sys.argv[1:]).

        Returns:
            The config file path string.

        Raises:
            SystemExit: If --config is not provided.
        """
        parser = argparse.ArgumentParser(
            description="Road Defect Segmentation Pipeline",
        )
        parser.add_argument(
            "--config",
            type=str,
            required=True,
            help="Path to the YAML configuration file",
        )

        parsed = parser.parse_args(args)
        return parsed.config

    def _map_yaml_to_config(self, raw: Dict[str, Any]) -> PipelineConfig:
        """Map nested YAML structure to flat PipelineConfig fields.

        The YAML config uses a nested structure (e.g. pipeline.frame_extraction_rate,
        detection.confidence_threshold) which must be mapped to the flat dataclass.

        Args:
            raw: Parsed YAML dictionary.

        Returns:
            A PipelineConfig with values from the YAML file, defaults for missing.
        """
        kwargs: Dict[str, Any] = {}

        # pipeline section
        pipeline_section = raw.get("pipeline", {}) or {}
        if "frame_extraction_rate" in pipeline_section:
            kwargs["frame_extraction_rate"] = pipeline_section["frame_extraction_rate"]

        # preprocessing section
        preprocessing_section = raw.get("preprocessing", {}) or {}
        if "clip_limit" in preprocessing_section:
            kwargs["clip_limit"] = preprocessing_section["clip_limit"]
        if "tile_grid_size" in preprocessing_section:
            value = preprocessing_section["tile_grid_size"]
            if isinstance(value, list) and len(value) == 2:
                kwargs["tile_grid_size"] = tuple(value)
        if "distortion_coefficients" in preprocessing_section:
            kwargs["distortion_coefficients"] = preprocessing_section[
                "distortion_coefficients"
            ]
        if "camera_matrix" in preprocessing_section:
            kwargs["camera_matrix"] = preprocessing_section["camera_matrix"]

        # detection section
        detection_section = raw.get("detection", {}) or {}
        if "model_path" in detection_section:
            kwargs["yolo_model_path"] = detection_section["model_path"]
        if "confidence_threshold" in detection_section:
            kwargs["confidence_threshold"] = detection_section["confidence_threshold"]
        if "iou_threshold" in detection_section:
            kwargs["iou_threshold"] = detection_section["iou_threshold"]
        if "max_detections" in detection_section:
            kwargs["max_detections"] = detection_section["max_detections"]

        # segmentation section
        segmentation_section = raw.get("segmentation", {}) or {}
        if "checkpoint_path" in segmentation_section:
            kwargs["sam2_checkpoint_path"] = segmentation_section["checkpoint_path"]
        if "model_cfg" in segmentation_section:
            kwargs["sam2_model_cfg"] = segmentation_section["model_cfg"]

        # verification section
        verification_section = raw.get("verification", {}) or {}
        if "min_area_ratio" in verification_section:
            kwargs["min_area_ratio"] = verification_section["min_area_ratio"]
        if "max_area_ratio" in verification_section:
            kwargs["max_area_ratio"] = verification_section["max_area_ratio"]

        # measurement section
        measurement_section = raw.get("measurement", {}) or {}
        if "camera_height_cm" in measurement_section:
            kwargs["camera_height_cm"] = measurement_section["camera_height_cm"]
        if "focal_length_px" in measurement_section:
            kwargs["focal_length_px"] = measurement_section["focal_length_px"]

        # output section
        output_section = raw.get("output", {}) or {}
        if "directory" in output_section:
            kwargs["output_directory"] = output_section["directory"]

        # logging section
        logging_section = raw.get("logging", {}) or {}
        if "level" in logging_section:
            kwargs["log_level"] = logging_section["level"]
        if "max_consecutive_failures" in logging_section:
            kwargs["max_consecutive_failures"] = logging_section[
                "max_consecutive_failures"
            ]

        return PipelineConfig(**kwargs)
