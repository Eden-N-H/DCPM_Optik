"""Frame preprocessing for the road defect segmentation pipeline.

Corrects barrel distortion, reduces motion blur, and normalizes exposure
using CLAHE in a fixed processing order. On failure, returns the original
frame unchanged and logs a warning.

Validates: Requirements 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7
"""

from __future__ import annotations

import numpy as np

from src.pipeline.logger import get_logger
from src.pipeline.models import PipelineConfig

try:
    import cv2
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "OpenCV is required for preprocessing. Install with: pip install opencv-python"
    ) from e

logger = get_logger("Preprocessor")

# Default GoPro wide-angle calibration coefficients (approximate)
_DEFAULT_DISTORTION_COEFFICIENTS = [-0.1, 0.01, 0.0, 0.0, 0.0]


class Preprocessor:
    """Correct barrel distortion, reduce motion blur, and normalize exposure.

    Processing order is fixed: distortion correction → motion blur reduction → CLAHE.
    On failure, returns the original frame unchanged and logs a warning.

    Attributes:
        config: Pipeline configuration containing preprocessing parameters.
    """

    def __init__(self, config: PipelineConfig) -> None:
        """Initialize preprocessor with pipeline configuration.

        Args:
            config: Pipeline configuration with preprocessing parameters
                including clip_limit, tile_grid_size, distortion_coefficients,
                and camera_matrix.
        """
        self._config = config
        self._clip_limit = config.clip_limit
        self._tile_grid_size = config.tile_grid_size

        # Set up distortion coefficients
        if config.distortion_coefficients is not None:
            self._dist_coeffs = np.array(
                config.distortion_coefficients, dtype=np.float64
            )
        else:
            self._dist_coeffs = np.array(
                _DEFAULT_DISTORTION_COEFFICIENTS, dtype=np.float64
            )
            logger.info(
                "No distortion coefficients provided; using default GoPro "
                "wide-angle calibration coefficients"
            )

        # Camera matrix will be computed per-frame if not provided
        if config.camera_matrix is not None:
            self._camera_matrix = np.array(
                config.camera_matrix, dtype=np.float64
            )
        else:
            self._camera_matrix = None

    def process(self, frame: np.ndarray) -> np.ndarray:
        """Apply all preprocessing steps in fixed order.

        Processing order: distortion correction → motion blur reduction → CLAHE.

        Args:
            frame: 8-bit RGB numpy array of shape (H, W, 3).

        Returns:
            Preprocessed 8-bit RGB numpy array of same shape (H, W, 3).
            On failure, returns the original frame unchanged.
        """
        try:
            result = self._correct_barrel_distortion(frame)
            result = self._reduce_motion_blur(result)
            result = self._apply_clahe(result)
            return result
        except Exception as exc:
            logger.warning(
                "Preprocessing failed, returning original frame: %s", str(exc)
            )
            return frame

    def _correct_barrel_distortion(self, frame: np.ndarray) -> np.ndarray:
        """Correct barrel distortion using cv2.undistort().

        Uses a 5-parameter distortion model (k1, k2, p1, p2, k3) and a 3x3
        camera intrinsic matrix. If no camera matrix was provided in config,
        one is constructed assuming the frame center as principal point.

        Args:
            frame: 8-bit RGB numpy array of shape (H, W, 3).

        Returns:
            Distortion-corrected frame with same shape and dtype.
        """
        h, w = frame.shape[:2]

        # Build camera matrix if not provided: assume frame center as principal point
        if self._camera_matrix is not None:
            camera_matrix = self._camera_matrix
        else:
            fx = float(w)
            fy = float(w)
            cx = w / 2.0
            cy = h / 2.0
            camera_matrix = np.array(
                [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
                dtype=np.float64,
            )

        corrected = cv2.undistort(frame, camera_matrix, self._dist_coeffs)
        return corrected

    def _reduce_motion_blur(self, frame: np.ndarray) -> np.ndarray:
        """Reduce motion blur using unsharp masking.

        Applies a Gaussian blur and subtracts it from the original to
        enhance sharpness.

        Args:
            frame: 8-bit RGB numpy array of shape (H, W, 3).

        Returns:
            Sharpened frame with same shape and dtype.
        """
        # Gaussian blur with moderate kernel
        blurred = cv2.GaussianBlur(frame, (5, 5), sigmaX=1.0)

        # Unsharp mask: original + alpha * (original - blurred)
        alpha = 1.5
        sharpened = cv2.addWeighted(frame, 1.0 + alpha, blurred, -alpha, 0)

        return sharpened

    def _apply_clahe(self, frame: np.ndarray) -> np.ndarray:
        """Apply CLAHE on the L channel in LAB color space.

        Converts to LAB, applies CLAHE to the L channel with configurable
        clip_limit and tile_grid_size, then converts back to RGB.

        Args:
            frame: 8-bit RGB numpy array of shape (H, W, 3).

        Returns:
            Exposure-normalized frame with same shape and dtype.
        """
        # Convert RGB to LAB
        lab = cv2.cvtColor(frame, cv2.COLOR_RGB2LAB)

        # Split into channels
        l_channel, a_channel, b_channel = cv2.split(lab)

        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(
            clipLimit=self._clip_limit, tileGridSize=self._tile_grid_size
        )
        l_enhanced = clahe.apply(l_channel)

        # Merge channels back
        lab_enhanced = cv2.merge([l_enhanced, a_channel, b_channel])

        # Convert back to RGB
        result = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2RGB)

        return result
