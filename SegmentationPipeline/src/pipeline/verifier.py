"""Post-segmentation verification for the road defect segmentation pipeline.

Validates segmentation quality through area ratio filtering and connected
component cleanup. Masks with area ratio below the minimum threshold are
discarded; masks above the maximum threshold are flagged for review.
Disconnected mask regions are cleaned by retaining only the largest
connected component (8-connectivity).

Validates: Requirements 5.1, 5.2, 5.3, 5.4, 5.5
"""

from __future__ import annotations

from typing import List, Tuple

import cv2
import numpy as np

from src.pipeline.logger import get_logger
from src.pipeline.models import (
    Detection,
    PipelineConfig,
    SegmentationResult,
    VerifiedResult,
)

logger = get_logger("PostSegmentationVerifier")


class PostSegmentationVerifier:
    """Validate segmentation quality and filter spurious masks.

    Performs two verification steps on each segmentation result:
    1. Connected component cleanup: retains only the largest connected
       component (8-connectivity) in the binary mask.
    2. Area ratio verification: computes the ratio of foreground pixels
       within the bounding box region to the total bounding box area,
       then discards or flags results based on configured thresholds.

    Args:
        config: Pipeline configuration containing min_area_ratio and
            max_area_ratio thresholds.
    """

    def __init__(self, config: PipelineConfig) -> None:
        self._min_area_ratio = config.min_area_ratio
        self._max_area_ratio = config.max_area_ratio

    def verify(self, results: List[SegmentationResult]) -> List[VerifiedResult]:
        """Verify a list of segmentation results.

        For each result:
        1. Clean the mask by retaining only the largest connected component.
        2. Compute the area ratio (foreground pixels in bbox / bbox area).
        3. Discard if ratio < min_area_ratio (log WARNING).
        4. Flag for review if ratio > max_area_ratio (log WARNING).
        5. Otherwise, pass through with review_flag=False.

        Args:
            results: List of segmentation results to verify.

        Returns:
            List of verified results that passed the minimum area ratio check.
        """
        verified: List[VerifiedResult] = []

        for result in results:
            # Step 1: Clean connected components
            cleaned_mask = self._clean_connected_components(result.mask)

            # Step 2: Compute area ratio
            area_ratio = self._compute_area_ratio(cleaned_mask, result.bbox)

            # Step 3: Check minimum threshold - discard if too small
            if area_ratio < self._min_area_ratio:
                logger.warning(
                    "Discarding mask: area ratio %.4f below minimum %.4f "
                    "(bbox: x=%d, y=%d, w=%d, h=%d)",
                    area_ratio,
                    self._min_area_ratio,
                    result.bbox[0],
                    result.bbox[1],
                    result.bbox[2],
                    result.bbox[3],
                )
                continue

            # Step 4: Check maximum threshold - flag for review if too large
            review_flag = False
            if area_ratio > self._max_area_ratio:
                review_flag = True
                logger.warning(
                    "Flagging mask for review: area ratio %.4f exceeds "
                    "maximum %.4f (bbox: x=%d, y=%d, w=%d, h=%d)",
                    area_ratio,
                    self._max_area_ratio,
                    result.bbox[0],
                    result.bbox[1],
                    result.bbox[2],
                    result.bbox[3],
                )

            # Step 5: Create verified result
            verified.append(
                VerifiedResult(
                    mask=cleaned_mask,
                    detection=result.detection,
                    bbox=result.bbox,
                    area_ratio=area_ratio,
                    review_flag=review_flag,
                )
            )

        return verified

    def _compute_area_ratio(
        self, mask: np.ndarray, bbox: Tuple[int, int, int, int]
    ) -> float:
        """Compute the mask-to-bounding-box area ratio.

        Counts the number of foreground pixels within the bounding box region
        of the mask and divides by the total bounding box pixel area.

        Args:
            mask: Binary mask at full frame resolution, shape (H, W),
                dtype uint8, values 0 or 1.
            bbox: Bounding box as (x, y, width, height) in frame coordinates.

        Returns:
            The area ratio as a float in [0.0, 1.0]. Returns 0.0 if the
            bounding box has zero area.
        """
        x, y, w, h = bbox
        bbox_area = w * h

        if bbox_area <= 0:
            return 0.0

        # Extract the bbox region from the full-frame mask
        roi = mask[y : y + h, x : x + w]
        foreground_pixels = int(np.count_nonzero(roi))

        return foreground_pixels / bbox_area

    def _clean_connected_components(self, mask: np.ndarray) -> np.ndarray:
        """Retain only the largest connected component in the mask.

        Uses 8-connectivity to identify connected components and keeps
        only the largest one by pixel count, discarding all smaller
        components.

        Args:
            mask: Binary mask, shape (H, W), dtype uint8, values 0 or 1.

        Returns:
            Cleaned binary mask with only the largest connected component,
            same shape and dtype as input.
        """
        # Handle edge case: empty mask
        if np.count_nonzero(mask) == 0:
            return mask.copy()

        # Ensure mask is in uint8 format for connectedComponentsWithStats
        mask_uint8 = mask.astype(np.uint8)

        # Find connected components with 8-connectivity
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            mask_uint8, connectivity=8
        )

        # If only background (label 0) exists, return empty mask
        if num_labels <= 1:
            return np.zeros_like(mask)

        # Find the largest component (excluding background at label 0)
        # stats[:, cv2.CC_STAT_AREA] gives the area of each component
        # Label 0 is background, so we look at labels 1 onwards
        component_areas = stats[1:, cv2.CC_STAT_AREA]
        largest_label = np.argmax(component_areas) + 1  # +1 to offset for background

        # Create output mask with only the largest component
        cleaned = np.zeros_like(mask)
        cleaned[labels == largest_label] = 1

        return cleaned
