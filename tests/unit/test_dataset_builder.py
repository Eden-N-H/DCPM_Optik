"""Unit tests for DatasetBuilder split computation and balance validation.

Tests cover:
- Split count computation (largest-remainder method)
- View count balancing (50% ±2% dashcam/drone)
- DatasetBuilder end-to-end generation (small scale)
- Manifest writing and structure
- Validation of split/viewpoint balance
"""

import json
import math
from pathlib import Path

import pytest

from src.synth.dataset_builder import (
    DatasetBuilder,
    DatasetConfig,
    compute_split_counts,
    compute_view_counts,
    DEFAULT_DATASET_SIZE,
    DEFAULT_SPLIT_RATIOS,
)
from src.utils.data_types import DatasetManifest


# ---------------------------------------------------------------------------
# Tests for compute_split_counts
# ---------------------------------------------------------------------------


class TestComputeSplitCounts:
    """Tests for the split count computation helper."""

    def test_default_split_target_size(self):
        """Test 80/10/10 split for the default target of 16,036."""
        counts = compute_split_counts(16036, DEFAULT_SPLIT_RATIOS)

        assert sum(counts.values()) == 16036
        assert set(counts.keys()) == {"train", "val", "test"}

        # Check each split is within ±1% of expected
        assert math.isclose(counts["train"] / 16036, 0.80, abs_tol=0.01)
        assert math.isclose(counts["val"] / 16036, 0.10, abs_tol=0.01)
        assert math.isclose(counts["test"] / 16036, 0.10, abs_tol=0.01)

    def test_counts_sum_to_total(self):
        """Test that split counts always sum exactly to the total."""
        for total in [100, 1000, 16036, 15876, 16196, 7, 13]:
            counts = compute_split_counts(total, DEFAULT_SPLIT_RATIOS)
            assert sum(counts.values()) == total

    def test_specific_known_values(self):
        """Test specific expected values for small totals."""
        counts = compute_split_counts(100, DEFAULT_SPLIT_RATIOS)
        assert counts["train"] == 80
        assert counts["val"] == 10
        assert counts["test"] == 10

    def test_uneven_split(self):
        """Test a split where remainders matter."""
        counts = compute_split_counts(10, DEFAULT_SPLIT_RATIOS)
        assert sum(counts.values()) == 10
        assert counts["train"] == 8
        assert counts["val"] == 1
        assert counts["test"] == 1

    def test_custom_ratios(self):
        """Test with non-standard split ratios."""
        ratios = {"train": 0.7, "val": 0.15, "test": 0.15}
        counts = compute_split_counts(1000, ratios)
        assert sum(counts.values()) == 1000
        assert counts["train"] == 700
        assert counts["val"] == 150
        assert counts["test"] == 150

    def test_invalid_total_raises(self):
        """Test that zero or negative total raises ValueError."""
        with pytest.raises(ValueError, match="total must be positive"):
            compute_split_counts(0, DEFAULT_SPLIT_RATIOS)
        with pytest.raises(ValueError, match="total must be positive"):
            compute_split_counts(-5, DEFAULT_SPLIT_RATIOS)

    def test_ratios_not_summing_to_one_raises(self):
        """Test that ratios not summing to 1.0 raise ValueError."""
        with pytest.raises(ValueError, match="must sum to 1.0"):
            compute_split_counts(100, {"train": 0.5, "val": 0.2, "test": 0.1})

    def test_all_counts_within_tolerance_of_16036(self):
        """Test the exact target produces values within ±1% of expected ratios."""
        counts = compute_split_counts(16036, DEFAULT_SPLIT_RATIOS)

        # Train should be ~12829 (80%)
        assert 12749 <= counts["train"] <= 12909
        # Val should be ~1604 (10%)
        assert 1563 <= counts["val"] <= 1644
        # Test should be ~1604 (10%)
        assert 1563 <= counts["test"] <= 1644


# ---------------------------------------------------------------------------
# Tests for compute_view_counts
# ---------------------------------------------------------------------------


class TestComputeViewCounts:
    """Tests for the viewpoint balance computation."""

    def test_even_split(self):
        """Test that an even number splits evenly."""
        counts = compute_view_counts(100)
        assert counts["dashcam"] == 50
        assert counts["drone"] == 50

    def test_odd_split(self):
        """Test that an odd number gives one view one extra sample."""
        counts = compute_view_counts(101)
        assert counts["dashcam"] + counts["drone"] == 101
        assert counts["dashcam"] == 50
        assert counts["drone"] == 51

    def test_total_preserved(self):
        """Test that view counts always sum to the input."""
        for n in [1, 2, 5, 10, 100, 1000, 12829, 1604]:
            counts = compute_view_counts(n)
            assert counts["dashcam"] + counts["drone"] == n

    def test_balance_within_two_percent(self):
        """Test that each viewpoint is within 50% ±2% for various sizes."""
        for n in [100, 1000, 12829, 1604, 16036]:
            counts = compute_view_counts(n)
            for view_type, count in counts.items():
                proportion = count / n
                assert 0.48 <= proportion <= 0.52, (
                    f"{view_type} proportion {proportion:.4f} outside 50%±2% "
                    f"for n={n}"
                )


# ---------------------------------------------------------------------------
# Tests for DatasetBuilder (small scale end-to-end)
# ---------------------------------------------------------------------------


class TestDatasetBuilder:
    """Integration-style tests for DatasetBuilder with small dataset sizes."""

    def test_generate_small_dataset(self, tmp_path):
        """Test generating a small dataset (20 samples) with correct structure."""
        config = DatasetConfig(
            total_samples=20,
            seed=42,
        )
        builder = DatasetBuilder(config=config)
        manifest = builder.generate_dataset(output_root=tmp_path, total_samples=20)

        # Check total count
        assert manifest.total_samples == 20

        # Check splits sum
        assert sum(manifest.splits.values()) == 20

        # Check all samples have required metadata fields
        for sample in manifest.samples:
            assert "scene_id" in sample
            assert "split" in sample
            assert "view_type" in sample
            assert "defect_types_present" in sample
            assert "camera_config" in sample
            assert "paths" in sample
            assert sample["view_type"] in ("dashcam", "drone")
            assert sample["split"] in ("train", "val", "test")

    def test_directory_structure(self, tmp_path):
        """Test that output directory structure follows spec."""
        config = DatasetConfig(total_samples=10, seed=123)
        builder = DatasetBuilder(config=config)
        manifest = builder.generate_dataset(output_root=tmp_path)

        # Check that directories exist for each sample
        for sample in manifest.samples:
            scene_dir = tmp_path / sample["split"] / sample["view_type"] / sample["scene_id"]
            assert scene_dir.exists(), f"Missing scene dir: {scene_dir}"

    def test_manifest_json_written(self, tmp_path):
        """Test that manifest.json is written correctly."""
        config = DatasetConfig(total_samples=10, seed=42)
        builder = DatasetBuilder(config=config)
        builder.generate_dataset(output_root=tmp_path)

        manifest_path = tmp_path / "manifest.json"
        assert manifest_path.exists()

        with open(manifest_path) as f:
            data = json.load(f)

        assert data["total_samples"] == 10
        assert "splits" in data
        assert "samples" in data
        assert len(data["samples"]) == 10

    def test_viewpoint_balance_small(self, tmp_path):
        """Test viewpoint balance for a small dataset."""
        config = DatasetConfig(total_samples=100, seed=42)
        builder = DatasetBuilder(config=config)
        manifest = builder.generate_dataset(output_root=tmp_path)

        dashcam_count = sum(
            1 for s in manifest.samples if s["view_type"] == "dashcam"
        )
        drone_count = sum(
            1 for s in manifest.samples if s["view_type"] == "drone"
        )

        total = manifest.total_samples
        assert dashcam_count + drone_count == total
        # 50% ±2%
        assert 0.48 <= dashcam_count / total <= 0.52
        assert 0.48 <= drone_count / total <= 0.52

    def test_validate_split_balance_passes(self, tmp_path):
        """Test that validation passes for a correctly generated dataset."""
        config = DatasetConfig(total_samples=16036, seed=42)
        builder = DatasetBuilder(config=config)

        # We won't actually generate 16k images, but test validation logic
        # with a manifest that has correct proportions
        split_counts = compute_split_counts(16036, DEFAULT_SPLIT_RATIOS)
        samples = []
        scene_counter = 0
        for split_name, split_count in split_counts.items():
            view_counts = compute_view_counts(split_count)
            for vt, vc in view_counts.items():
                for _ in range(vc):
                    samples.append({
                        "scene_id": f"scene_{scene_counter:06d}",
                        "split": split_name,
                        "view_type": vt,
                    })
                    scene_counter += 1

        manifest = DatasetManifest(
            root=tmp_path,
            total_samples=len(samples),
            splits=split_counts,
            samples=samples,
        )

        result = builder.validate_split_balance(manifest)
        assert result["valid"], f"Validation failed: {result['issues']}"

    def test_validate_split_balance_fails_bad_total(self, tmp_path):
        """Test that validation fails when total is way off target."""
        config = DatasetConfig(total_samples=16036, seed=42)
        builder = DatasetBuilder(config=config)

        manifest = DatasetManifest(
            root=tmp_path,
            total_samples=10000,
            splits={"train": 8000, "val": 1000, "test": 1000},
            samples=[{"view_type": "dashcam", "split": "train"}] * 5000
            + [{"view_type": "drone", "split": "train"}] * 5000,
        )

        result = builder.validate_split_balance(manifest)
        assert not result["valid"]

    def test_render_outputs_exist(self, tmp_path):
        """Test that rendered files exist for each sample."""
        config = DatasetConfig(total_samples=4, seed=42)
        builder = DatasetBuilder(config=config)
        manifest = builder.generate_dataset(output_root=tmp_path)

        for sample in manifest.samples:
            paths = sample["paths"]
            assert Path(paths["rgb"]).exists()
            assert Path(paths["depth"]).exists()
            assert Path(paths["segmentation"]).exists()
            assert Path(paths["severity"]).exists()
            assert Path(paths["camera_params"]).exists()

    def test_reproducibility_with_seed(self, tmp_path):
        """Test that the same seed produces the same manifest metadata."""
        config1 = DatasetConfig(total_samples=10, seed=99)
        builder1 = DatasetBuilder(config=config1)

        out1 = tmp_path / "run1"
        manifest1 = builder1.generate_dataset(output_root=out1)

        config2 = DatasetConfig(total_samples=10, seed=99)
        builder2 = DatasetBuilder(config=config2)

        out2 = tmp_path / "run2"
        manifest2 = builder2.generate_dataset(output_root=out2)

        # Same scene IDs, splits, view types, defect counts
        for s1, s2 in zip(manifest1.samples, manifest2.samples):
            assert s1["scene_id"] == s2["scene_id"]
            assert s1["split"] == s2["split"]
            assert s1["view_type"] == s2["view_type"]
            assert s1["num_defects"] == s2["num_defects"]
            assert s1["defect_types_present"] == s2["defect_types_present"]

