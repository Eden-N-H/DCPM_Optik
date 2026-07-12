"""Property-based tests for ConfigLoader (Properties 29, 30, 31, 32).

Validates: Requirements 19.1, 19.2, 19.3, 19.4
"""

import copy
import tempfile
from pathlib import Path

import pytest
import yaml
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from src.utils.config import ConfigLoader, ConfigValidationError


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# Strategy for generating simple YAML-serializable scalar values
yaml_scalars = st.one_of(
    st.integers(min_value=-1000, max_value=1000),
    st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
    st.text(min_size=1, max_size=20, alphabet=st.characters(whitelist_categories=("L", "N", "P"))),
    st.booleans(),
)

# Strategy for generating nested dicts that are YAML-round-trip safe
yaml_leaf = st.one_of(
    st.integers(min_value=-1000, max_value=1000),
    st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
    st.booleans(),
    st.text(min_size=1, max_size=15, alphabet=st.characters(whitelist_categories=("L", "N"))),
    st.lists(st.integers(min_value=-100, max_value=100), min_size=0, max_size=5),
)


@st.composite
def yaml_dicts(draw, max_depth=2):
    """Generate nested dicts that survive YAML round-trip."""
    if max_depth <= 0:
        return draw(yaml_leaf)
    keys = draw(st.lists(
        st.text(min_size=1, max_size=10, alphabet=st.characters(whitelist_categories=("L",))),
        min_size=1,
        max_size=5,
        unique=True,
    ))
    values = [draw(st.one_of(yaml_leaf, yaml_dicts(max_depth=max_depth - 1))) for _ in keys]
    return dict(zip(keys, values))


# Strategy for valid partial configs that pass validation when merged with defaults
@st.composite
def valid_partial_configs(draw):
    """Generate partial configs that, when merged with DEFAULTS, pass validation."""
    partial = {}

    # Optionally override some training params with valid values
    if draw(st.booleans()):
        partial["training"] = {}
        if draw(st.booleans()):
            lr = draw(st.floats(min_value=1e-6, max_value=1.0))
            partial["training"]["optimizer"] = {"lr": lr}
        if draw(st.booleans()):
            batch_size = draw(st.integers(min_value=1, max_value=64))
            partial["training"]["batch_size"] = batch_size
        if draw(st.booleans()):
            epochs = draw(st.integers(min_value=1, max_value=500))
            partial["training"]["epochs"] = epochs

    # Optionally override data params with valid values
    if draw(st.booleans()):
        # Generate valid splits that sum to 1.0
        train_split = draw(st.floats(min_value=0.5, max_value=0.9))
        val_split = draw(st.floats(min_value=0.05, max_value=min(0.4, 1.0 - train_split - 0.01)))
        test_split = round(1.0 - train_split - val_split, 10)
        # Ensure test_split is valid
        assume(0.0 <= test_split <= 1.0)
        assume(abs(train_split + val_split + test_split - 1.0) < 0.01)
        partial["data"] = {
            "train_split": train_split,
            "val_split": val_split,
            "test_split": test_split,
        }
        if draw(st.booleans()):
            partial["data"]["num_workers"] = draw(st.integers(min_value=0, max_value=16))

    # Optionally add a seed
    if draw(st.booleans()):
        partial["seed"] = draw(st.integers(min_value=0, max_value=9999))

    return partial


# Strategy for dot-notation override keys that exist in defaults
@st.composite
def valid_override_pairs(draw):
    """Generate (dot_key, value) pairs that target real keys in DEFAULTS."""
    # Pick from known valid paths
    valid_paths_with_values = [
        ("training.optimizer.lr", st.floats(min_value=1e-6, max_value=1.0)),
        ("training.batch_size", st.integers(min_value=1, max_value=128)),
        ("training.epochs", st.integers(min_value=1, max_value=1000)),
        ("data.num_workers", st.integers(min_value=0, max_value=32)),
        ("training.amp", st.booleans()),
        ("training.grad_clip_norm", st.floats(min_value=0.01, max_value=100.0)),
        ("seed", st.integers(min_value=0, max_value=99999)),
        ("model.encoder.backbone", st.just("resnet50")),
        ("model.view_embedding.embed_dim", st.integers(min_value=1, max_value=128)),
    ]
    path, value_strategy = draw(st.sampled_from(valid_paths_with_values))
    value = draw(value_strategy)
    return path, value


# ---------------------------------------------------------------------------
# Property 29: YAML configuration round-trip
# ---------------------------------------------------------------------------


class TestProperty29YAMLRoundTrip:
    """**Validates: Requirements 19.1**

    Property 29: For any valid configuration dictionary, writing to YAML
    and reading back SHALL produce an equivalent dictionary with all values preserved.
    """

    @given(data=yaml_dicts(max_depth=2))
    @settings(max_examples=50, deadline=5000)
    def test_yaml_round_trip_arbitrary_dict(self, data, tmp_path_factory):
        """Write arbitrary nested dict to YAML and read back produces equivalent dict."""
        tmp_dir = tmp_path_factory.mktemp("yaml_rt")
        yaml_path = tmp_dir / "test_config.yaml"

        # Write and read back
        with open(yaml_path, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)
        with open(yaml_path, "r") as f:
            loaded = yaml.safe_load(f)

        assert loaded == data

    @given(partial=valid_partial_configs())
    @settings(max_examples=30, deadline=10000)
    def test_config_loader_yaml_round_trip(self, partial, tmp_path_factory):
        """ConfigLoader write to YAML and read back produces equivalent config."""
        tmp_dir = tmp_path_factory.mktemp("config_rt")

        # Create a config from partial (merged with defaults)
        yaml_path = tmp_dir / "input.yaml"
        with open(yaml_path, "w") as f:
            yaml.dump(partial, f, default_flow_style=False, sort_keys=False)

        config1 = ConfigLoader(config_path=yaml_path)

        # Write the full config to YAML
        output_path = tmp_dir / "output.yaml"
        config1.to_yaml(output_path)

        # Read it back
        config2 = ConfigLoader(config_path=output_path)

        assert config1.config == config2.config


# ---------------------------------------------------------------------------
# Property 30: Configuration defaults for partial configs
# ---------------------------------------------------------------------------


class TestProperty30ConfigDefaults:
    """**Validates: Requirements 19.2**

    Property 30: For any valid partial configuration (subset of keys), loading
    SHALL produce a complete configuration where all unspecified parameters have
    their documented default values and all specified parameters retain their
    provided values.
    """

    @given(partial=valid_partial_configs())
    @settings(max_examples=50, deadline=10000)
    def test_defaults_applied_for_unspecified_keys(self, partial, tmp_path_factory):
        """Unspecified params get documented defaults, specified params retained."""
        tmp_dir = tmp_path_factory.mktemp("defaults")
        yaml_path = tmp_dir / "partial.yaml"

        with open(yaml_path, "w") as f:
            yaml.dump(partial, f, default_flow_style=False, sort_keys=False)

        config = ConfigLoader(config_path=yaml_path)
        full_config = config.config
        defaults = ConfigLoader.DEFAULTS

        # Check: specified params are retained
        self._assert_specified_retained(partial, full_config)

        # Check: unspecified top-level sections get defaults
        for key in defaults:
            if key not in partial:
                assert full_config[key] == defaults[key], (
                    f"Default not applied for unspecified key '{key}'"
                )

    def _assert_specified_retained(self, partial, full_config):
        """Recursively check that specified values are retained."""
        for key, value in partial.items():
            assert key in full_config, f"Specified key '{key}' missing from config"
            if isinstance(value, dict) and isinstance(full_config[key], dict):
                self._assert_specified_retained(value, full_config[key])
            else:
                assert full_config[key] == value, (
                    f"Specified value for '{key}' not retained: "
                    f"expected {value}, got {full_config[key]}"
                )

    def test_defaults_only_config(self):
        """ConfigLoader with no YAML file produces exact DEFAULTS."""
        config = ConfigLoader()
        assert config.config == ConfigLoader.DEFAULTS


# ---------------------------------------------------------------------------
# Property 31: Configuration validation rejects invalid values
# ---------------------------------------------------------------------------


class TestProperty31ValidationRejectsInvalid:
    """**Validates: Requirements 19.3**

    Property 31: For any configuration containing values outside valid ranges
    (negative learning rates, empty required paths, dilation rates <= 0,
    loss weights < 0), validation SHALL raise a clear error identifying
    the invalid parameter.
    """

    @given(lr=st.floats(max_value=0, allow_nan=False, allow_infinity=False))
    @settings(max_examples=30, deadline=10000)
    def test_negative_learning_rate_rejected(self, lr, tmp_path_factory):
        """Negative or zero learning rates raise ConfigValidationError."""
        tmp_dir = tmp_path_factory.mktemp("invalid_lr")
        yaml_path = tmp_dir / "bad_lr.yaml"
        config_data = {"training": {"optimizer": {"lr": lr}}}
        with open(yaml_path, "w") as f:
            yaml.dump(config_data, f)

        with pytest.raises(ConfigValidationError) as exc_info:
            ConfigLoader(config_path=yaml_path)
        assert "lr" in str(exc_info.value).lower() or "positive" in str(exc_info.value).lower()

    @given(batch_size=st.integers(max_value=0))
    @settings(max_examples=20, deadline=10000)
    def test_invalid_batch_size_rejected(self, batch_size, tmp_path_factory):
        """Zero or negative batch sizes raise ConfigValidationError."""
        tmp_dir = tmp_path_factory.mktemp("invalid_bs")
        yaml_path = tmp_dir / "bad_bs.yaml"
        config_data = {"training": {"batch_size": batch_size}}
        with open(yaml_path, "w") as f:
            yaml.dump(config_data, f)

        with pytest.raises(ConfigValidationError) as exc_info:
            ConfigLoader(config_path=yaml_path)
        assert "batch_size" in str(exc_info.value)

    @given(weight=st.floats(max_value=-0.01, allow_nan=False, allow_infinity=False))
    @settings(max_examples=20, deadline=10000)
    def test_negative_loss_weight_rejected(self, weight, tmp_path_factory):
        """Negative loss weights raise ConfigValidationError."""
        tmp_dir = tmp_path_factory.mktemp("invalid_lw")
        yaml_path = tmp_dir / "bad_weight.yaml"
        config_data = {"training": {"loss_weights": {"segmentation": weight}}}
        with open(yaml_path, "w") as f:
            yaml.dump(config_data, f)

        with pytest.raises(ConfigValidationError) as exc_info:
            ConfigLoader(config_path=yaml_path)
        assert "loss_weights" in str(exc_info.value) or "non-negative" in str(exc_info.value).lower()

    @given(epochs=st.integers(max_value=0))
    @settings(max_examples=20, deadline=10000)
    def test_invalid_epochs_rejected(self, epochs, tmp_path_factory):
        """Zero or negative epochs raise ConfigValidationError."""
        tmp_dir = tmp_path_factory.mktemp("invalid_ep")
        yaml_path = tmp_dir / "bad_epochs.yaml"
        config_data = {"training": {"epochs": epochs}}
        with open(yaml_path, "w") as f:
            yaml.dump(config_data, f)

        with pytest.raises(ConfigValidationError) as exc_info:
            ConfigLoader(config_path=yaml_path)
        assert "epochs" in str(exc_info.value)

    @given(
        train_split=st.floats(min_value=0.1, max_value=0.5),
        val_split=st.floats(min_value=0.1, max_value=0.5),
    )
    @settings(max_examples=30, deadline=10000)
    def test_invalid_split_sum_rejected(self, train_split, val_split, tmp_path_factory):
        """Data splits that don't sum to ~1.0 raise ConfigValidationError."""
        test_split = 0.01  # Force sum to be far from 1.0
        split_sum = train_split + val_split + test_split
        assume(abs(split_sum - 1.0) > 0.01)

        tmp_dir = tmp_path_factory.mktemp("invalid_split")
        yaml_path = tmp_dir / "bad_split.yaml"
        config_data = {
            "data": {
                "train_split": train_split,
                "val_split": val_split,
                "test_split": test_split,
            }
        }
        with open(yaml_path, "w") as f:
            yaml.dump(config_data, f)

        with pytest.raises(ConfigValidationError) as exc_info:
            ConfigLoader(config_path=yaml_path)
        assert "split" in str(exc_info.value).lower()

    @given(num_workers=st.integers(max_value=-1))
    @settings(max_examples=20, deadline=10000)
    def test_negative_num_workers_rejected(self, num_workers, tmp_path_factory):
        """Negative num_workers raises ConfigValidationError."""
        tmp_dir = tmp_path_factory.mktemp("invalid_nw")
        yaml_path = tmp_dir / "bad_nw.yaml"
        config_data = {"data": {"num_workers": num_workers}}
        with open(yaml_path, "w") as f:
            yaml.dump(config_data, f)

        with pytest.raises(ConfigValidationError) as exc_info:
            ConfigLoader(config_path=yaml_path)
        assert "num_workers" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Property 32: CLI dot-notation override application
# ---------------------------------------------------------------------------


class TestProperty32CLIOverrides:
    """**Validates: Requirements 19.4**

    Property 32: For any valid configuration and a set of dot-notation overrides
    (e.g., "training.lr=1e-3"), after applying overrides the configuration SHALL
    have the specified nested key set to the override value while all other keys
    remain unchanged.
    """

    @given(override_pair=valid_override_pairs())
    @settings(max_examples=50, deadline=10000)
    def test_override_sets_nested_key(self, override_pair):
        """CLI override sets the specified nested key to the override value."""
        dot_key, value = override_pair

        # Format the override string
        override_str = f"--{dot_key}={value}"
        config = ConfigLoader(overrides=[override_str])

        # The overridden value should match
        actual = config.get(dot_key)
        if isinstance(value, float):
            assert abs(actual - value) < 1e-9, (
                f"Override for '{dot_key}' not applied: expected {value}, got {actual}"
            )
        else:
            assert actual == value, (
                f"Override for '{dot_key}' not applied: expected {value}, got {actual}"
            )

    @given(override_pair=valid_override_pairs())
    @settings(max_examples=30, deadline=10000)
    def test_override_preserves_other_keys(self, override_pair):
        """CLI override does not modify other unrelated keys."""
        dot_key, value = override_pair

        # Get baseline config (defaults only)
        baseline = ConfigLoader()
        baseline_config = baseline.config

        # Apply override
        override_str = f"--{dot_key}={value}"
        config = ConfigLoader(overrides=[override_str])
        new_config = config.config

        # All keys except the overridden path should remain unchanged
        self._assert_unchanged_except(baseline_config, new_config, dot_key.split("."))

    def _assert_unchanged_except(self, original, modified, path_parts, current_path=""):
        """Recursively check that all keys except the override path are unchanged."""
        if not isinstance(original, dict) or not isinstance(modified, dict):
            return

        for key in original:
            full_path = f"{current_path}.{key}" if current_path else key
            if path_parts and key == path_parts[0]:
                # This key is on the override path; check deeper
                if len(path_parts) == 1:
                    # This is the overridden leaf — skip comparison
                    continue
                else:
                    self._assert_unchanged_except(
                        original[key], modified[key], path_parts[1:], full_path
                    )
            else:
                # This key is NOT on the override path — must be unchanged
                assert modified.get(key) == original[key], (
                    f"Key '{full_path}' was modified by override but should not have been"
                )

    @given(
        overrides=st.lists(valid_override_pairs(), min_size=1, max_size=3, unique_by=lambda x: x[0])
    )
    @settings(max_examples=30, deadline=10000)
    def test_multiple_overrides_all_applied(self, overrides):
        """Multiple CLI overrides all get applied correctly."""
        override_strs = [f"--{key}={value}" for key, value in overrides]
        config = ConfigLoader(overrides=override_strs)

        for dot_key, expected_value in overrides:
            actual = config.get(dot_key)
            if isinstance(expected_value, float):
                assert abs(actual - expected_value) < 1e-9, (
                    f"Override for '{dot_key}' not applied"
                )
            else:
                assert actual == expected_value, (
                    f"Override for '{dot_key}' not applied: expected {expected_value}, got {actual}"
                )

    def test_malformed_override_rejected(self):
        """Malformed override without '=' raises ConfigValidationError."""
        with pytest.raises(ConfigValidationError):
            ConfigLoader(overrides=["--training.lr"])
