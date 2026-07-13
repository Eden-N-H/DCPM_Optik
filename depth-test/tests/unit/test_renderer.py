"""Unit tests for domain randomization and multi-pass rendering.

Tests the SceneRenderer class including:
- Domain randomization (HDRI selection, vehicle placement, weather)
- Render output structure (correct file types, dimensions, value ranges)
- Camera parameters JSON serialization
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from src.synth.renderer import (
    DomainRandomizationConfig,
    DomainRandomizationState,
    MockRenderBackend,
    RenderConfig,
    SceneRenderer,
    HDRI_LIBRARY,
    WEATHER_TYPES,
    VEHICLE_COUNT_RANGE,
    save_camera_params_json,
    load_camera_params_json,
)
from src.synth.scene_generator import SceneGenerator, SceneConfig, generate_intrinsics, generate_extrinsics
from src.utils.data_types import CameraConfig, RenderOutputs


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def renderer():
    """Create a SceneRenderer with mock backend and fixed seed."""
    return SceneRenderer(seed=42)


@pytest.fixture
def scene_gen():
    """Create a SceneGenerator with fixed seed."""
    return SceneGenerator(seed=42)


@pytest.fixture
def camera_config():
    """Create a sample CameraConfig for testing."""
    intrinsics = generate_intrinsics(focal_length=443.0, image_size=512)
    extrinsics = generate_extrinsics(height=1.3, pitch_deg=-10.0, road_length=100.0)
    return CameraConfig(
        view_type="dashcam",
        height=1.3,
        pitch=-10.0,
        intrinsics=intrinsics,
        extrinsics=extrinsics,
    )


@pytest.fixture
def output_dir(tmp_path):
    """Create a temporary output directory."""
    return tmp_path / "render_output"


# ---------------------------------------------------------------------------
# Domain Randomization Tests
# ---------------------------------------------------------------------------


class TestDomainRandomization:
    """Tests for apply_domain_randomization()."""

    def test_hdri_library_has_at_least_20_maps(self):
        """HDRI library must contain at least 20 maps (Req 1.5)."""
        assert len(HDRI_LIBRARY) >= 20

    def test_hdri_selection(self, renderer):
        """Domain randomization selects a valid HDRI from the library."""
        state = renderer.apply_domain_randomization(
            road_width=7.5, road_length=100.0
        )
        assert state.hdri_map in HDRI_LIBRARY

    def test_vehicle_count_range(self, renderer):
        """Vehicle count is between 0 and 5 (Req 1.5)."""
        # Run many times to get variety
        counts = set()
        for seed in range(100):
            r = SceneRenderer(seed=seed)
            state = r.apply_domain_randomization(
                road_width=7.5, road_length=100.0
            )
            counts.add(state.vehicle_count)
            assert 0 <= state.vehicle_count <= 5

        # Verify we see at least some variation
        assert len(counts) > 1

    def test_vehicle_positions_within_road(self, renderer):
        """Vehicle positions are within road bounds."""
        road_width = 7.5
        road_length = 100.0
        state = renderer.apply_domain_randomization(
            road_width=road_width, road_length=road_length
        )
        for vx, vy, vr in state.vehicle_positions:
            assert 0.0 <= vx <= road_width
            assert 0.0 <= vy <= road_length
            assert 0.0 <= vr <= 360.0

    def test_weather_selection(self, renderer):
        """Weather is selected from valid set (Req 1.5)."""
        state = renderer.apply_domain_randomization(
            road_width=7.5, road_length=100.0
        )
        assert state.weather in WEATHER_TYPES

    def test_weather_types_include_required(self):
        """Weather types include clear, overcast, rain."""
        assert "clear" in WEATHER_TYPES
        assert "overcast" in WEATHER_TYPES
        assert "rain" in WEATHER_TYPES

    def test_returns_domain_randomization_state(self, renderer):
        """apply_domain_randomization returns proper state object."""
        state = renderer.apply_domain_randomization(
            road_width=7.5, road_length=100.0
        )
        assert isinstance(state, DomainRandomizationState)
        assert isinstance(state.hdri_map, str)
        assert isinstance(state.vehicle_count, int)
        assert isinstance(state.vehicle_positions, list)
        assert isinstance(state.weather, str)
        assert len(state.vehicle_positions) == state.vehicle_count

    def test_insufficient_hdri_library_raises_error(self):
        """Raises ValueError if HDRI library has fewer than 20 maps."""
        small_config = DomainRandomizationConfig(
            hdri_library=["map_1", "map_2", "map_3"]
        )
        renderer = SceneRenderer(dr_config=small_config, seed=42)
        with pytest.raises(ValueError, match="at least 20 maps"):
            renderer.apply_domain_randomization(
                road_width=7.5, road_length=100.0
            )

    def test_reproducibility_with_same_seed(self):
        """Same seed produces same randomization results."""
        r1 = SceneRenderer(seed=123)
        r2 = SceneRenderer(seed=123)

        state1 = r1.apply_domain_randomization(road_width=7.5, road_length=100.0)
        state2 = r2.apply_domain_randomization(road_width=7.5, road_length=100.0)

        assert state1.hdri_map == state2.hdri_map
        assert state1.vehicle_count == state2.vehicle_count
        assert state1.vehicle_positions == state2.vehicle_positions
        assert state1.weather == state2.weather


# ---------------------------------------------------------------------------
# Render Output Tests
# ---------------------------------------------------------------------------


class TestRender:
    """Tests for render() method output structure."""

    def test_render_returns_render_outputs(self, renderer, output_dir, camera_config):
        """render() returns a RenderOutputs dataclass."""
        outputs = renderer.render(output_dir, camera_config)
        assert isinstance(outputs, RenderOutputs)

    def test_render_creates_all_files(self, renderer, output_dir, camera_config):
        """render() creates all expected output files."""
        outputs = renderer.render(output_dir, camera_config)

        assert outputs.rgb.exists()
        assert outputs.depth.exists()
        assert outputs.segmentation.exists()
        assert outputs.severity.exists()
        assert outputs.camera_params.exists()

    def test_rgb_is_512x512_png(self, renderer, output_dir, camera_config):
        """RGB output is a 512x512x3 PNG (Req 1.3)."""
        import cv2

        outputs = renderer.render(output_dir, camera_config)

        img = cv2.imread(str(outputs.rgb), cv2.IMREAD_COLOR)
        assert img is not None
        assert img.shape == (512, 512, 3)
        assert img.dtype == np.uint8

    def test_depth_is_512x512_16bit_png(self, renderer, output_dir, camera_config):
        """Depth output is a 512x512 16-bit PNG in mm (Req 1.3)."""
        import cv2

        outputs = renderer.render(output_dir, camera_config)

        img = cv2.imread(str(outputs.depth), cv2.IMREAD_UNCHANGED)
        assert img is not None
        assert img.shape == (512, 512)
        assert img.dtype == np.uint16
        # Values should be in valid 16-bit range for millimeter depth
        assert img.max() <= 65535

    def test_segmentation_is_512x512_integer_mask(self, renderer, output_dir, camera_config):
        """Segmentation output is 512x512 with integer class IDs (Req 1.3)."""
        import cv2

        outputs = renderer.render(output_dir, camera_config)

        img = cv2.imread(str(outputs.segmentation), cv2.IMREAD_UNCHANGED)
        assert img is not None
        assert img.shape == (512, 512)
        # All values should be valid class IDs (0-7)
        assert img.min() >= 0
        assert img.max() <= 7
        assert np.issubdtype(img.dtype, np.integer)

    def test_severity_is_512x512_float32(self, renderer, output_dir, camera_config):
        """Severity output is 512x512 float32 in [0, 1] (Req 1.3)."""
        outputs = renderer.render(output_dir, camera_config)

        severity = np.load(str(outputs.severity))
        assert severity.shape == (512, 512)
        assert severity.dtype == np.float32
        assert severity.min() >= 0.0
        assert severity.max() <= 1.0

    def test_camera_params_json_structure(self, renderer, output_dir, camera_config):
        """Camera params JSON has intrinsics K and extrinsics [R|t] (Req 1.4)."""
        outputs = renderer.render(output_dir, camera_config)

        with open(outputs.camera_params, "r") as f:
            params = json.load(f)

        assert "intrinsics_K" in params
        assert "extrinsics_Rt" in params

        K = np.array(params["intrinsics_K"])
        Rt = np.array(params["extrinsics_Rt"])

        assert K.shape == (3, 3)
        assert Rt.shape == (3, 4)

    def test_camera_params_serialization_roundtrip(self, output_dir, camera_config):
        """Camera params round-trip preserves values within tolerance (Req 1.4)."""
        path = output_dir / "test_camera.json"
        path.parent.mkdir(parents=True, exist_ok=True)

        save_camera_params_json(path, camera_config.intrinsics, camera_config.extrinsics)
        loaded_K, loaded_Rt = load_camera_params_json(path)

        np.testing.assert_allclose(loaded_K, camera_config.intrinsics, atol=1e-6)
        np.testing.assert_allclose(loaded_Rt, camera_config.extrinsics, atol=1e-6)

    def test_render_with_custom_scene_id(self, renderer, output_dir, camera_config):
        """render() uses scene_id in file names."""
        outputs = renderer.render(output_dir, camera_config, scene_id="test_scene_001")

        assert "test_scene_001" in outputs.rgb.name
        assert "test_scene_001" in outputs.depth.name
        assert "test_scene_001" in outputs.segmentation.name
        assert "test_scene_001" in outputs.severity.name
        assert "test_scene_001" in outputs.camera_params.name

    def test_render_output_directory_created(self, tmp_path, renderer, camera_config):
        """render() creates output directory if it doesn't exist."""
        nested_dir = tmp_path / "a" / "b" / "c"
        assert not nested_dir.exists()

        renderer.render(nested_dir, camera_config)
        assert nested_dir.exists()


# ---------------------------------------------------------------------------
# Integration: Domain Randomization + Render
# ---------------------------------------------------------------------------


class TestDomainRandomizationAndRender:
    """Tests combining domain randomization with rendering."""

    def test_full_pipeline(self, output_dir):
        """Test the full domain randomization + render pipeline."""
        renderer = SceneRenderer(seed=42)
        gen = SceneGenerator(seed=42)

        # Generate road
        road_mesh, road_width, road_length = gen.generate_road_mesh(
            lanes=2, lane_width=3.5, length=100.0
        )

        # Apply domain randomization
        dr_state = renderer.apply_domain_randomization(
            road_width=road_width, road_length=road_length
        )

        # Set up camera
        camera = gen.setup_camera("dashcam", road_length=road_length)

        # Render
        outputs = renderer.render(output_dir, camera)

        # Verify outputs
        assert outputs.rgb.exists()
        assert outputs.depth.exists()
        assert outputs.segmentation.exists()
        assert outputs.severity.exists()
        assert outputs.camera_params.exists()

        # Verify domain randomization was applied
        assert dr_state.hdri_map in HDRI_LIBRARY
        assert dr_state.weather in WEATHER_TYPES
        assert 0 <= dr_state.vehicle_count <= 5

    def test_different_seeds_produce_different_randomization(self, output_dir):
        """Different seeds produce different domain randomization states."""
        states = []
        for seed in [1, 2, 3, 4, 5]:
            r = SceneRenderer(seed=seed)
            state = r.apply_domain_randomization(road_width=7.5, road_length=100.0)
            states.append(state)

        # At least some should differ
        hdris = [s.hdri_map for s in states]
        weathers = [s.weather for s in states]
        # With 5 different seeds, we expect some variation
        assert len(set(hdris)) > 1 or len(set(weathers)) > 1


# ---------------------------------------------------------------------------
# Camera Params Serialization Tests
# ---------------------------------------------------------------------------


class TestCameraParamsSerialization:
    """Tests for camera parameter JSON serialization/deserialization."""

    def test_save_and_load_identity_matrices(self, tmp_path):
        """Save/load with identity-like matrices."""
        K = np.eye(3, dtype=np.float64)
        Rt = np.hstack([np.eye(3), np.zeros((3, 1))]).astype(np.float64)
        path = tmp_path / "cam.json"

        save_camera_params_json(path, K, Rt)
        K_loaded, Rt_loaded = load_camera_params_json(path)

        np.testing.assert_allclose(K_loaded, K, atol=1e-10)
        np.testing.assert_allclose(Rt_loaded, Rt, atol=1e-10)

    def test_save_and_load_realistic_params(self, tmp_path):
        """Save/load with realistic camera parameters."""
        K = generate_intrinsics(focal_length=443.4, image_size=512)
        Rt = generate_extrinsics(height=1.4, pitch_deg=-12.0, road_length=150.0)
        path = tmp_path / "cam_realistic.json"

        save_camera_params_json(path, K, Rt)
        K_loaded, Rt_loaded = load_camera_params_json(path)

        np.testing.assert_allclose(K_loaded, K, atol=1e-6)
        np.testing.assert_allclose(Rt_loaded, Rt, atol=1e-6)

    def test_json_is_human_readable(self, tmp_path):
        """JSON output is formatted for readability."""
        K = np.eye(3, dtype=np.float64)
        Rt = np.hstack([np.eye(3), np.zeros((3, 1))]).astype(np.float64)
        path = tmp_path / "cam_readable.json"

        save_camera_params_json(path, K, Rt)

        content = path.read_text()
        # Should contain newlines (indented format)
        assert "\n" in content
        # Should parse as valid JSON
        parsed = json.loads(content)
        assert "intrinsics_K" in parsed
        assert "extrinsics_Rt" in parsed
