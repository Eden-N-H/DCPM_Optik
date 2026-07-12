"""Unit tests for core data structures in utils/data_types.py."""
import numpy as np
import pytest
from pathlib import Path

from src.utils.data_types import (
    DefectSpec,
    DefectInstance,
    CameraConfig,
    RenderOutputs,
    DatasetManifest,
    ModelOutput,
    PointCloudData,
    BEVMap,
    Checkpoint,
)


class TestDefectSpec:
    """Tests for DefectSpec dataclass."""

    def test_create_crack(self):
        spec = DefectSpec(
            defect_type="crack",
            position=(1.0, 2.0),
            orientation=45.0,
            scale=(1.5, 0.02),
        )
        assert spec.defect_type == "crack"
        assert spec.position == (1.0, 2.0)
        assert spec.orientation == 45.0
        assert spec.scale == (1.5, 0.02)

    def test_create_pothole(self):
        spec = DefectSpec(
            defect_type="pothole",
            position=(5.0, 3.0),
            orientation=0.0,
            scale=(0.5, 0.1),
        )
        assert spec.defect_type == "pothole"

    def test_all_defect_types(self):
        """All 5 defect types from Req 1.2 can be represented."""
        for dtype in ["crack", "pothole", "puddle", "patch", "manhole"]:
            spec = DefectSpec(defect_type=dtype, position=(0.0, 0.0), orientation=0.0, scale=(1.0,))
            assert spec.defect_type == dtype


class TestDefectInstance:
    """Tests for DefectInstance dataclass."""

    def test_create_instance(self):
        spec = DefectSpec(defect_type="crack", position=(1.0, 2.0), orientation=0.0, scale=(1.0, 0.01))
        instance = DefectInstance(
            spec=spec,
            mesh_object=None,
            bounding_box_2d=(10.0, 20.0, 50.0, 60.0),
            area=0.015,
        )
        assert instance.spec is spec
        assert instance.bounding_box_2d == (10.0, 20.0, 50.0, 60.0)
        assert instance.area == 0.015


class TestCameraConfig:
    """Tests for CameraConfig dataclass — validates Req 1.4."""

    def test_dashcam_config(self):
        K = np.eye(3, dtype=np.float64)
        K[0, 0] = 500.0  # fx
        K[1, 1] = 500.0  # fy
        K[0, 2] = 256.0  # cx
        K[1, 2] = 256.0  # cy

        Rt = np.zeros((3, 4), dtype=np.float64)
        Rt[:3, :3] = np.eye(3)

        config = CameraConfig(
            view_type="dashcam",
            height=1.3,
            pitch=-10.0,
            intrinsics=K,
            extrinsics=Rt,
        )
        assert config.view_type == "dashcam"
        assert config.height == 1.3
        assert config.pitch == -10.0
        assert config.intrinsics.shape == (3, 3)
        assert config.extrinsics.shape == (3, 4)

    def test_drone_config(self):
        K = np.eye(3, dtype=np.float64)
        Rt = np.hstack([np.eye(3), np.zeros((3, 1))])

        config = CameraConfig(
            view_type="drone",
            height=12.0,
            pitch=-75.0,
            intrinsics=K,
            extrinsics=Rt,
        )
        assert config.view_type == "drone"
        assert config.intrinsics.shape == (3, 3)
        assert config.extrinsics.shape == (3, 4)


class TestRenderOutputs:
    """Tests for RenderOutputs — validates Req 1.3."""

    def test_all_paths(self):
        outputs = RenderOutputs(
            rgb=Path("/data/scene_001/rgb.png"),
            depth=Path("/data/scene_001/depth.png"),
            segmentation=Path("/data/scene_001/segmentation.png"),
            severity=Path("/data/scene_001/severity.npy"),
            camera_params=Path("/data/scene_001/camera.json"),
        )
        assert outputs.rgb.suffix == ".png"
        assert outputs.depth.suffix == ".png"
        assert outputs.segmentation.suffix == ".png"
        assert outputs.severity.suffix == ".npy"
        assert outputs.camera_params.suffix == ".json"


class TestDatasetManifest:
    """Tests for DatasetManifest — validates Req 2.5."""

    def test_manifest_structure(self):
        manifest = DatasetManifest(
            root=Path("/data/road_quality"),
            total_samples=16036,
            splits={"train": 12829, "val": 1604, "test": 1603},
            samples=[
                {"scene_id": "001", "view_type": "dashcam", "defect_types_present": ["crack"]},
                {"scene_id": "002", "view_type": "drone", "defect_types_present": ["pothole", "puddle"]},
            ],
        )
        assert manifest.total_samples == 16036
        assert sum(manifest.splits.values()) == 16036
        assert len(manifest.samples) == 2
        assert "scene_id" in manifest.samples[0]
        assert "view_type" in manifest.samples[0]
        assert "defect_types_present" in manifest.samples[0]


class TestModelOutput:
    """Tests for ModelOutput dataclass."""

    def test_model_output_shapes(self):
        """ModelOutput can hold tensor-like objects with correct semantics."""
        # Using numpy arrays as stand-in for torch tensors
        B = 4
        output = ModelOutput(
            segmentation=np.zeros((B, 3, 512, 512)),
            severity=np.zeros((B, 1, 512, 512)),
            depth=np.zeros((B, 1, 512, 512)),
            intrinsics=np.zeros((B, 4)),
            extrinsics=np.zeros((B, 6)),
        )
        assert output.segmentation.shape == (B, 3, 512, 512)
        assert output.severity.shape == (B, 1, 512, 512)
        assert output.depth.shape == (B, 1, 512, 512)
        assert output.intrinsics.shape == (B, 4)
        assert output.extrinsics.shape == (B, 6)


class TestPointCloudData:
    """Tests for PointCloudData — validates Req 13.3."""

    def test_point_cloud_attributes(self):
        N = 1000
        pcd = PointCloudData(
            positions=np.random.randn(N, 3).astype(np.float32),
            colors=np.random.randint(0, 256, (N, 3), dtype=np.uint8),
            classes=np.random.randint(0, 6, N, dtype=np.int32),
            severities=np.random.rand(N).astype(np.float32),
        )
        assert pcd.positions.shape == (N, 3)
        assert pcd.colors.shape == (N, 3)
        assert pcd.colors.dtype == np.uint8
        assert pcd.classes.shape == (N,)
        assert pcd.severities.shape == (N,)

    def test_empty_point_cloud(self):
        pcd = PointCloudData(
            positions=np.zeros((0, 3)),
            colors=np.zeros((0, 3), dtype=np.uint8),
            classes=np.zeros(0, dtype=np.int32),
            severities=np.zeros(0, dtype=np.float64),
        )
        assert pcd.positions.shape == (0, 3)
        assert len(pcd.classes) == 0


class TestBEVMap:
    """Tests for BEVMap — validates Req 14.1."""

    def test_bev_map_structure(self):
        H, W = 500, 300
        bev = BEVMap(
            image=np.zeros((H, W, 3), dtype=np.uint8),
            class_grid=np.zeros((H, W), dtype=np.int32),
            severity_grid=np.zeros((H, W), dtype=np.float64),
            origin=(0.0, 0.0),
            resolution=0.02,
        )
        assert bev.image.shape == (H, W, 3)
        assert bev.class_grid.shape == (H, W)
        assert bev.severity_grid.shape == (H, W)
        assert bev.resolution == 0.02
        assert bev.origin == (0.0, 0.0)

    def test_configurable_resolution(self):
        """BEV resolution is configurable per Req 14.1."""
        for res in [0.01, 0.02, 0.05, 0.1]:
            bev = BEVMap(
                image=np.zeros((10, 10, 3), dtype=np.uint8),
                class_grid=np.zeros((10, 10), dtype=np.int32),
                severity_grid=np.zeros((10, 10), dtype=np.float64),
                origin=(0.0, 0.0),
                resolution=res,
            )
            assert bev.resolution == res


class TestCheckpoint:
    """Tests for Checkpoint dataclass — validates Req 18.1."""

    def test_checkpoint_structure(self):
        ckpt = Checkpoint(
            epoch=42,
            model_state_dict={"layer.weight": np.zeros((64, 64))},
            optimizer_state_dict={"state": {}, "param_groups": []},
            scheduler_state_dict={"last_epoch": 42},
            best_metric=0.85,
            rng_states={
                "python": 12345,
                "numpy": np.random.get_state(),
                "torch": None,
                "cuda": None,
            },
        )
        assert ckpt.epoch == 42
        assert ckpt.best_metric == 0.85
        assert "python" in ckpt.rng_states
        assert "numpy" in ckpt.rng_states
        assert "torch" in ckpt.rng_states
        assert "cuda" in ckpt.rng_states
