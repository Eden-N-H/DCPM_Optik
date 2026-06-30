"""Measurement engine for computing defect area, dimensions, and severity.

Computes pixel-based measurements from verified segmentation masks, and
optionally converts to metric units (cm², cm) when camera parameters are
provided. Assigns severity classifications based on area thresholds.

Validates: Requirements 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from src.pipeline.logger import get_logger
from src.pipeline.models import DefectMeasurement, PipelineConfig, VerifiedResult

logger = get_logger("MeasurementEngine")


class MeasurementEngine:
    """Compute defect area, dimensions, and severity from verified masks.

    Always computes pixel-based measurements. Computes metric measurements
    only when camera_height_cm and focal_length_px are both provided and
    positive. Returns None for metric fields and severity when camera
    parameters are unavailable or invalid.

    Args:
        config: Pipeline configuration containing camera parameters.
    """

    def __init__(self, config: PipelineConfig) -> None:
        self._camera_height_cm = config.camera_height_cm
        self._focal_length_px = config.focal_length_px
        self._has_valid_camera_params = self._validate_camera_params()

    def _validate_camera_params(self) -> bool:
        """Check if camera parameters are available and valid (positive).

        Returns:
            True if both camera_height_cm and focal_length_px are provided
            and positive, False otherwise.
        """
        if self._camera_height_cm is None or self._focal_length_px is None:
            return False

        if self._camera_height_cm <= 0 or self._focal_length_px <= 0:
            logger.warning(
                "Camera parameters are zero or negative "
                "(camera_height_cm=%s, focal_length_px=%s). "
                "Falling back to pixel-only measurements.",
                self._camera_height_cm,
                self._focal_length_px,
            )
            return False

        return True

    def measure(self, verified_result: VerifiedResult) -> DefectMeasurement:
        """Compute measurements from a verified segmentation mask.

        Computes area in pixels and bounding dimensions in pixels. If camera
        parameters are valid, also computes metric measurements and severity.

        Args:
            verified_result: A verified segmentation result containing a
                binary mask.

        Returns:
            DefectMeasurement with pixel-based measurements always populated,
            and metric fields/severity set to None when camera parameters
            are unavailable.
        """
        mask = verified_result.mask

        # Always compute pixel-based measurements
        area_pixels = int(np.count_nonzero(mask))
        length_pixels, width_pixels = self._compute_bounding_dimensions(mask)

        # Compute metric measurements if camera params are valid
        area_cm2: Optional[float] = None
        width_cm: Optional[float] = None
        length_cm: Optional[float] = None
        severity: Optional[str] = None

        if self._has_valid_camera_params:
            area_cm2 = self._pixel_to_cm2(area_pixels)
            pixel_size_cm = self._camera_height_cm / self._focal_length_px  # type: ignore[operator]
            width_cm = round(width_pixels * pixel_size_cm, 1)
            length_cm = round(length_pixels * pixel_size_cm, 1)
            severity = self._compute_severity(area_cm2)

        return DefectMeasurement(
            area_pixels=area_pixels,
            area_cm2=area_cm2,
            width_cm=width_cm,
            length_cm=length_cm,
            width_pixels=width_pixels,
            length_pixels=length_pixels,
            severity=severity,
        )

    def _pixel_to_cm2(self, pixel_area: int) -> float:
        """Convert pixel area to square centimeters using ground-plane projection.

        Formula: area_cm2 = pixel_area × (camera_height_cm / focal_length_px)²

        Args:
            pixel_area: Defect area in pixels.

        Returns:
            Area in square centimeters, rounded to 1 decimal place.
        """
        pixel_size_cm = self._camera_height_cm / self._focal_length_px  # type: ignore[operator]
        area_cm2 = pixel_area * (pixel_size_cm ** 2)
        return round(area_cm2, 1)

    def _compute_bounding_dimensions(self, mask: np.ndarray) -> Tuple[int, int]:
        """Compute bounding dimensions from the mask's non-zero pixel extent.

        Length is computed as max_row - min_row, width as max_col - min_col.

        Args:
            mask: Binary mask array of shape (H, W).

        Returns:
            Tuple of (length_pixels, width_pixels). Returns (0, 0) if the
            mask contains no foreground pixels.
        """
        nonzero_coords = np.nonzero(mask)

        if len(nonzero_coords[0]) == 0:
            return (0, 0)

        rows = nonzero_coords[0]
        cols = nonzero_coords[1]

        length_pixels = int(rows.max() - rows.min())
        width_pixels = int(cols.max() - cols.min())

        return (length_pixels, width_pixels)

    def _compute_severity(self, area_cm2: float) -> str:
        """Assign severity classification based on area thresholds.

        Severity levels:
            - minor: area < 500 cm²
            - moderate: 500 ≤ area ≤ 2000 cm²
            - severe: area > 2000 cm²

        Args:
            area_cm2: Defect area in square centimeters.

        Returns:
            Severity string: "minor", "moderate", or "severe".
        """
        if area_cm2 < 500:
            return "minor"
        elif area_cm2 <= 2000:
            return "moderate"
        else:
            return "severe"
