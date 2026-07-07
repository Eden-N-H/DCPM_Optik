"""Output writer for serializing pipeline results to JSON files.

Writes per-frame JSON records and a batch summary report to the
configured output directory. Masks are encoded in COCO compressed RLE format.

Validates: Requirements 7.1, 7.2, 7.3, 7.4, 7.5, 7.6
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict, List

import numpy as np
from pycocotools import mask as mask_utils

from src.pipeline.logger import get_logger
from src.pipeline.models import BatchSummary, DefectOutput, FrameMetadata, PipelineConfig


class OutputWriter:
    """Serialize pipeline results to JSON files in the output directory.

    Writes one JSON file per processed frame (named {frame_id}.json) and
    a batch_summary.json at the end of a processing run.

    Validates output directory accessibility at initialization and exits
    with an error if the directory does not exist or is not writable.
    """

    def __init__(self, config: PipelineConfig) -> None:
        """Initialize the OutputWriter.

        Validates that the configured output directory exists and is writable.
        Exits with an error message if validation fails.

        Args:
            config: Pipeline configuration containing output_directory.
        """
        self._logger = get_logger("OutputWriter")
        self._output_directory = config.output_directory

        # Validate output directory exists and is writable
        if not os.path.isdir(self._output_directory):
            print(
                f"Error: Output directory does not exist: {self._output_directory}",
                file=sys.stderr,
            )
            sys.exit(1)

        if not os.access(self._output_directory, os.W_OK):
            print(
                f"Error: Output directory is not writable: {self._output_directory}",
                file=sys.stderr,
            )
            sys.exit(1)

        self._logger.info(
            "OutputWriter initialized with output directory: %s",
            self._output_directory,
        )

    def write_frame_result(
        self, frame_metadata: FrameMetadata, defects: List[DefectOutput]
    ) -> None:
        """Write a per-frame JSON result file.

        Serializes frame metadata and defect list to a JSON file named
        {frame_id}.json in the output directory. Produces a record with
        an empty defects list when no defects are detected.

        Args:
            frame_metadata: Metadata for the processed frame.
            defects: List of defect outputs for this frame (may be empty).
        """
        frame_record: Dict[str, Any] = {
            "frame_id": frame_metadata.frame_id,
            "timestamp": frame_metadata.timestamp,
            "source_file": os.path.basename(frame_metadata.source_path),
            "defects": [defect.to_dict() for defect in defects],
        }

        output_path = os.path.join(
            self._output_directory, f"{frame_metadata.frame_id}.json"
        )

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(frame_record, f, indent=2)

        self._logger.debug(
            "Wrote frame result: %s (%d defects)",
            output_path,
            len(defects),
        )

    def write_batch_summary(self, summary: BatchSummary) -> None:
        """Write the batch summary JSON file.

        Serializes batch processing statistics to batch_summary.json
        in the output directory.

        Args:
            summary: Batch summary statistics for the completed run.
        """
        output_path = os.path.join(self._output_directory, "batch_summary.json")

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(summary.to_dict(), f, indent=2)

        self._logger.info("Wrote batch summary: %s", output_path)

    def _encode_mask_rle(self, mask: np.ndarray) -> Dict[str, Any]:
        """Encode a binary mask to COCO compressed RLE format.

        Uses pycocotools.mask.encode() which requires a Fortran-ordered
        uint8 array. The resulting counts bytes are decoded to a UTF-8
        string for JSON serialization.

        Args:
            mask: Binary mask of shape (H, W), dtype uint8, values 0 or 1.

        Returns:
            Dictionary with 'size' (list [H, W]) and 'counts' (RLE string).
        """
        # pycocotools requires Fortran-ordered uint8 array
        fortran_mask = np.asfortranarray(mask.astype(np.uint8))
        rle = mask_utils.encode(fortran_mask)

        return {
            "size": list(rle["size"]),
            "counts": rle["counts"].decode("utf-8"),
        }
