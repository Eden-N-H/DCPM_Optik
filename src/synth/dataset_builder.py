"""Dataset builder for orchestrating full synthetic dataset generation.

Generates the complete dataset of images split into train/val/test sets with
balanced dashcam/drone viewpoints. Organizes output into the required directory
structure and produces a dataset manifest JSON.

Requirements: 1.7, 2.1, 2.2, 2.3, 2.4, 2.5
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np

from src.synth.scene_generator import SceneConfig, SceneGenerator
from src.synth.renderer import (
    DomainRandomizationConfig,
    RenderConfig,
    SceneRenderer,
)
from src.utils.data_types import CameraConfig, DatasetManifest, DefectInstance, RenderOutputs


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Target dataset size (Req 1.7)
DEFAULT_DATASET_SIZE = 16036

# Split ratios (Req 2.1)
DEFAULT_SPLIT_RATIOS: Dict[str, float] = {
    "train": 0.80,
    "val": 0.10,
    "test": 0.10,
}

# View types for balanced generation (Req 1.7)
VIEW_TYPES: List[str] = ["dashcam", "drone"]


@dataclass
class DatasetConfig:
    """Configuration for dataset generation.

    Attributes:
        total_samples: Target total number of images to generate.
        split_ratios: Dictionary mapping split names to their proportions.
        scene_config: Configuration for scene generation.
        dr_config: Domain randomization configuration.
        render_config: Rendering configuration.
        seed: Random seed for reproducibility.
    """

    total_samples: int = DEFAULT_DATASET_SIZE
    split_ratios: Dict[str, float] = field(
        default_factory=lambda: dict(DEFAULT_SPLIT_RATIOS)
    )
    scene_config: SceneConfig = field(default_factory=SceneConfig)
    dr_config: DomainRandomizationConfig = field(
        default_factory=DomainRandomizationConfig
    )
    render_config: RenderConfig = field(default_factory=RenderConfig)
    seed: Optional[int] = None


# ---------------------------------------------------------------------------
# Split computation helpers
# ---------------------------------------------------------------------------


def compute_split_counts(
    total: int, ratios: Dict[str, float]
) -> Dict[str, int]:
    """Compute the number of samples per split from ratios.

    Uses largest-remainder method to ensure counts sum exactly to total
    while maintaining ratios as closely as possible.

    Args:
        total: Total number of samples.
        ratios: Dictionary mapping split names to their proportions (must sum to ~1.0).

    Returns:
        Dictionary mapping split names to integer sample counts.

    Raises:
        ValueError: If ratios don't sum approximately to 1.0 or if total <= 0.
    """
    if total <= 0:
        raise ValueError(f"total must be positive, got {total}")

    ratio_sum = sum(ratios.values())
    if not math.isclose(ratio_sum, 1.0, abs_tol=1e-6):
        raise ValueError(
            f"Split ratios must sum to 1.0, got {ratio_sum}"
        )

    # Compute exact (fractional) counts
    exact_counts = {name: total * ratio for name, ratio in ratios.items()}

    # Floor each count
    floor_counts = {name: int(math.floor(count)) for name, count in exact_counts.items()}

    # Distribute remaining samples by largest fractional remainder
    remainder = total - sum(floor_counts.values())
    remainders = {
        name: exact_counts[name] - floor_counts[name] for name in ratios
    }

    # Sort by remainder descending, then by name for determinism
    sorted_names = sorted(
        remainders.keys(), key=lambda n: (-remainders[n], n)
    )

    for i in range(remainder):
        floor_counts[sorted_names[i]] += 1

    return floor_counts


def compute_view_counts(split_count: int) -> Dict[str, int]:
    """Compute balanced dashcam/drone counts for a given split size.

    Ensures 50% ±2% balance between viewpoints.

    Args:
        split_count: Total number of samples in this split.

    Returns:
        Dictionary mapping view type to count. Sums to split_count.
    """
    dashcam_count = split_count // 2
    drone_count = split_count - dashcam_count
    return {"dashcam": dashcam_count, "drone": drone_count}


# ---------------------------------------------------------------------------
# DatasetBuilder class
# ---------------------------------------------------------------------------


class DatasetBuilder:
    """Orchestrates full dataset generation with proper splits and viewpoint balance.

    Generates a complete dataset by repeatedly invoking SceneGenerator and
    SceneRenderer, organizing the output into the required directory structure:
    {root}/{split}/{view_type}/{scene_id}/

    Produces a dataset manifest JSON with per-sample metadata.

    Attributes:
        config: Dataset generation configuration.
        scene_generator: SceneGenerator instance for creating scenes.
        renderer: SceneRenderer instance for rendering output modalities.
        rng: Random number generator for reproducibility.
    """

    def __init__(
        self,
        config: Optional[DatasetConfig] = None,
        scene_generator: Optional[SceneGenerator] = None,
        renderer: Optional[SceneRenderer] = None,
    ) -> None:
        """Initialize the DatasetBuilder.

        Args:
            config: Dataset generation configuration. Uses defaults if None.
            scene_generator: SceneGenerator to use. Creates one if None.
            renderer: SceneRenderer to use. Creates one if None.
        """
        self.config = config or DatasetConfig()
        seed = self.config.seed

        self.scene_generator = scene_generator or SceneGenerator(
            config=self.config.scene_config, seed=seed
        )
        self.renderer = renderer or SceneRenderer(
            dr_config=self.config.dr_config,
            render_config=self.config.render_config,
            seed=seed,
        )
        self.rng = random.Random(seed)

    def generate_dataset(
        self,
        output_root: Path,
        total_samples: Optional[int] = None,
    ) -> DatasetManifest:
        """Generate the full dataset with train/val/test splits and viewpoint balance.

        Produces images organized into:
            {output_root}/{split}/{view_type}/{scene_id}/

        Each scene directory contains RGB, depth, segmentation, severity, and
        camera parameter files.

        Args:
            output_root: Root directory for dataset output.
            total_samples: Override for total samples. Uses config value if None.

        Returns:
            DatasetManifest with per-sample metadata and split counts.
        """
        output_root = Path(output_root)
        total = total_samples if total_samples is not None else self.config.total_samples

        # Compute split counts
        split_counts = compute_split_counts(total, self.config.split_ratios)

        # Generate samples for each split
        all_samples: List[Dict[str, Any]] = []
        scene_counter = 0

        for split_name, split_count in split_counts.items():
            view_counts = compute_view_counts(split_count)

            for view_type, view_count in view_counts.items():
                for i in range(view_count):
                    scene_id = f"scene_{scene_counter:06d}"
                    scene_counter += 1

                    sample_metadata = self._generate_single_sample(
                        output_root=output_root,
                        split=split_name,
                        view_type=view_type,
                        scene_id=scene_id,
                    )
                    all_samples.append(sample_metadata)

        # Build manifest
        manifest = DatasetManifest(
            root=output_root,
            total_samples=len(all_samples),
            splits=split_counts,
            samples=all_samples,
        )

        # Write manifest JSON
        self._write_manifest(output_root, manifest)

        return manifest

    def validate_split_balance(
        self, manifest: DatasetManifest
    ) -> Dict[str, Any]:
        """Validate that the dataset meets split and viewpoint balance requirements.

        Checks:
        - Total samples within ±1% of 16,036 (15,876 to 16,196)
        - Each viewpoint is 50% ±2% of total
        - Splits are approximately 80/10/10

        Args:
            manifest: The dataset manifest to validate.

        Returns:
            Dictionary with validation results including pass/fail and details.
        """
        total = manifest.total_samples
        results: Dict[str, Any] = {"valid": True, "issues": []}

        # Check total count (±1% of target)
        target = self.config.total_samples
        lower = int(target * 0.99)
        upper = int(target * 1.01)
        if not (lower <= total <= upper):
            results["valid"] = False
            results["issues"].append(
                f"Total samples {total} outside ±1% of {target} "
                f"(expected {lower}-{upper})"
            )

        # Check viewpoint balance (50% ±2%)
        view_counts: Dict[str, int] = {"dashcam": 0, "drone": 0}
        for sample in manifest.samples:
            vt = sample.get("view_type", "")
            if vt in view_counts:
                view_counts[vt] += 1

        for view_type, count in view_counts.items():
            proportion = count / total if total > 0 else 0
            if not (0.48 <= proportion <= 0.52):
                results["valid"] = False
                results["issues"].append(
                    f"View type '{view_type}' proportion {proportion:.3f} "
                    f"outside 50% ±2% range"
                )

        # Check split ratios
        for split_name, expected_ratio in self.config.split_ratios.items():
            split_count = manifest.splits.get(split_name, 0)
            actual_ratio = split_count / total if total > 0 else 0
            tolerance = 0.01  # ±1% tolerance
            if not (expected_ratio - tolerance <= actual_ratio <= expected_ratio + tolerance):
                results["valid"] = False
                results["issues"].append(
                    f"Split '{split_name}' ratio {actual_ratio:.3f} "
                    f"outside {expected_ratio} ±{tolerance}"
                )

        results["total_samples"] = total
        results["view_counts"] = view_counts
        results["split_counts"] = dict(manifest.splits)

        return results

    # -----------------------------------------------------------------------
    # Private helpers
    # -----------------------------------------------------------------------

    def _generate_single_sample(
        self,
        output_root: Path,
        split: str,
        view_type: str,
        scene_id: str,
    ) -> Dict[str, Any]:
        """Generate a single scene sample and return its metadata.

        Args:
            output_root: Dataset root directory.
            split: Split name (train/val/test).
            view_type: View type (dashcam/drone).
            scene_id: Unique scene identifier.

        Returns:
            Dictionary with sample metadata.
        """
        # Create output directory: {root}/{split}/{view_type}/{scene_id}/
        scene_dir = output_root / split / view_type / scene_id
        scene_dir.mkdir(parents=True, exist_ok=True)

        # Generate road mesh
        road_mesh, road_width, road_length = self.scene_generator.generate_road_mesh()

        # Place defects
        defect_instances = self.scene_generator.place_defects(
            road_width=road_width,
            road_length=road_length,
        )

        # Set up camera
        camera_config = self.scene_generator.setup_camera(
            view_type=view_type,  # type: ignore[arg-type]
            road_length=road_length,
        )

        # Apply domain randomization
        dr_state = self.renderer.apply_domain_randomization(
            road_width=road_width,
            road_length=road_length,
        )

        # Render all output modalities
        render_outputs = self.renderer.render(
            output_dir=scene_dir,
            camera_config=camera_config,
            scene_id=scene_id,
        )

        # Collect defect types present
        defect_types_present = list(set(
            inst.spec.defect_type for inst in defect_instances
        ))

        # Build sample metadata (Req 2.5)
        metadata: Dict[str, Any] = {
            "scene_id": scene_id,
            "split": split,
            "view_type": view_type,
            "defect_types_present": sorted(defect_types_present),
            "num_defects": len(defect_instances),
            "camera_config": {
                "height": camera_config.height,
                "pitch": camera_config.pitch,
                "view_type": camera_config.view_type,
            },
            "paths": {
                "rgb": str(render_outputs.rgb),
                "depth": str(render_outputs.depth),
                "segmentation": str(render_outputs.segmentation),
                "severity": str(render_outputs.severity),
                "camera_params": str(render_outputs.camera_params),
            },
            "domain_randomization": {
                "hdri_map": dr_state.hdri_map,
                "vehicle_count": dr_state.vehicle_count,
                "weather": dr_state.weather,
            },
        }

        return metadata

    def _write_manifest(
        self, output_root: Path, manifest: DatasetManifest
    ) -> None:
        """Write the dataset manifest as a JSON file.

        Args:
            output_root: Dataset root directory.
            manifest: The manifest to serialize.
        """
        manifest_path = output_root / "manifest.json"
        manifest_path.parent.mkdir(parents=True, exist_ok=True)

        manifest_dict = {
            "root": str(manifest.root),
            "total_samples": manifest.total_samples,
            "splits": manifest.splits,
            "samples": manifest.samples,
        }

        with open(manifest_path, "w") as f:
            json.dump(manifest_dict, f, indent=2)

