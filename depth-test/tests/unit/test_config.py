"""Unit tests for ConfigLoader - YAML loading, defaults, validation, and CLI overrides.

Tests Requirements 19.1, 19.2, 19.3, 19.4.
"""

import pytest
import yaml
from pathlib import Path
from src.utils.config import ConfigLoader, ConfigValidationError


class TestConfigLoaderDefaults:
    """Test that ConfigLoader provides documented defaults (Req 19.2)."""

    def test_loads_with_no_args(self):
        """ConfigLoader with no arguments produces a valid config from defaults."""
        config = ConfigLoader()
        assert config.get("training.optimizer.lr") == 0.0001
        assert config.get("training.epochs") == 200
        assert config.get("training.batch_size") == 8
        assert config.get("seed") == 42

    def test_all_required_sections_have_defaults(self):
        """All required sections are present in defaults."""
        config = ConfigLoader()
        assert config.get("data") is not None
        assert config.get("model") is not None
        assert config.get("training") is not None
        assert config.get("cyclegan") is not None
        assert config.get("reconstruction") is not None
        assert config.get("logging") is not None

    def test_model_architecture_defaults(self):
        """Model architecture defaults are correct."""
        config = ConfigLoader()
        assert config.get("model.encoder.backbone") == "resnet50"
        assert config.get("model.encoder.pretrained") is True
        assert config.get("model.encoder.dsc_stages") == [3, 4]
        assert config.get("model.easpp.dilations") == [3, 6, 12, 18]
        assert config.get("model.easpp.out_channels") == 256
        assert config.get("model.soa.reduction") == 16
        assert config.get("model.decoder.channels") == [256, 128, 64]

    def test_training_hyperparameter_defaults(self):
        """Training hyperparameter defaults are correct."""
        config = ConfigLoader()
        assert config.get("training.optimizer.type") == "adam"
        assert config.get("training.optimizer.weight_decay") == 0.00001
        assert config.get("training.scheduler.type") == "reduce_on_plateau"
        assert config.get("training.scheduler.patience") == 10
        assert config.get("training.scheduler.factor") == 0.5
        assert config.get("training.amp") is True
        assert config.get("training.grad_clip_norm") == 1.0

    def test_loss_weight_defaults(self):
        """Loss weight defaults match requirements."""
        config = ConfigLoader()
        assert config.get("training.loss_weights.segmentation") == 1.5
        assert config.get("training.loss_weights.depth") == 1.0
        assert config.get("training.loss_weights.camera") == 0.3
        assert config.get("training.loss_weights.adversarial") == 0.1
        assert config.get("training.loss_weights.view") == 0.1

    def test_data_path_defaults(self):
        """Data path defaults are present."""
        config = ConfigLoader()
        assert config.get("data.root") == "./data/road_quality"
        assert config.get("data.train_split") == 0.8
        assert config.get("data.val_split") == 0.1
        assert config.get("data.test_split") == 0.1

    def test_augmentation_defaults(self):
        """Augmentation parameter defaults are documented."""
        config = ConfigLoader()
        assert config.get("augmentation.horizontal_flip") is True
        assert config.get("augmentation.rotation_range") == 10
        assert config.get("augmentation.crop_size") == 480
        assert config.get("augmentation.color_jitter.brightness") == 0.2
        assert config.get("augmentation.color_jitter.contrast") == 0.2
        assert config.get("augmentation.color_jitter.saturation") == 0.1
        assert config.get("augmentation.color_jitter.hue") == 0.05


class TestConfigLoaderYAML:
    """Test YAML file loading (Req 19.1)."""

    def test_loads_from_yaml_file(self, tmp_path):
        """ConfigLoader loads configuration from a YAML file."""
        yaml_content = {
            "training": {"epochs": 100, "batch_size": 16},
            "seed": 123,
        }
        config_file = tmp_path / "test_config.yaml"
        with open(config_file, "w") as f:
            yaml.dump(yaml_content, f)

        config = ConfigLoader(config_path=config_file)
        assert config.get("training.epochs") == 100
        assert config.get("training.batch_size") == 16
        assert config.get("seed") == 123

    def test_yaml_merges_with_defaults(self, tmp_path):
        """YAML values override defaults while defaults fill gaps."""
        yaml_content = {
            "training": {"epochs": 50},
        }
        config_file = tmp_path / "partial.yaml"
        with open(config_file, "w") as f:
            yaml.dump(yaml_content, f)

        config = ConfigLoader(config_path=config_file)
        # Overridden value
        assert config.get("training.epochs") == 50
        # Default value preserved
        assert config.get("training.batch_size") == 8
        assert config.get("training.optimizer.lr") == 0.0001

    def test_deep_nested_yaml_merge(self, tmp_path):
        """Deep nested YAML values merge correctly."""
        yaml_content = {
            "model": {"encoder": {"backbone": "resnet101"}},
        }
        config_file = tmp_path / "nested.yaml"
        with open(config_file, "w") as f:
            yaml.dump(yaml_content, f)

        config = ConfigLoader(config_path=config_file)
        assert config.get("model.encoder.backbone") == "resnet101"
        # Other encoder defaults preserved
        assert config.get("model.encoder.pretrained") is True
        assert config.get("model.encoder.dsc_stages") == [3, 4]

    def test_file_not_found_raises(self):
        """Raises FileNotFoundError for non-existent YAML file."""
        with pytest.raises(FileNotFoundError, match="Configuration file not found"):
            ConfigLoader(config_path=Path("/nonexistent/config.yaml"))

    def test_empty_yaml_file_uses_defaults(self, tmp_path):
        """An empty YAML file results in all defaults."""
        config_file = tmp_path / "empty.yaml"
        config_file.write_text("")

        config = ConfigLoader(config_path=config_file)
        assert config.get("training.epochs") == 200
        assert config.get("seed") == 42

    def test_loads_default_yaml(self):
        """Loads the actual default.yaml from the project."""
        config_path = Path(__file__).parent.parent.parent / "configs" / "default.yaml"
        if config_path.exists():
            config = ConfigLoader(config_path=config_path)
            assert config.get("training.epochs") == 200
            assert config.get("model.encoder.backbone") == "resnet50"


class TestConfigLoaderCLIOverrides:
    """Test CLI dot-notation overrides (Req 19.4)."""

    def test_simple_override(self):
        """Simple dot-notation override works."""
        config = ConfigLoader(overrides=["training.epochs=100"])
        assert config.get("training.epochs") == 100

    def test_override_with_dashes(self):
        """Leading dashes are stripped from override keys."""
        config = ConfigLoader(overrides=["--training.epochs=100"])
        assert config.get("training.epochs") == 100

    def test_nested_override(self):
        """Deeply nested dot-notation override works."""
        config = ConfigLoader(overrides=["--training.optimizer.lr=1e-3"])
        assert config.get("training.optimizer.lr") == 0.001

    def test_float_scientific_notation(self):
        """Scientific notation floats are parsed correctly."""
        config = ConfigLoader(overrides=["--training.optimizer.lr=2e-4"])
        assert config.get("training.optimizer.lr") == 0.0002

    def test_boolean_override(self):
        """Boolean overrides parse correctly."""
        config = ConfigLoader(overrides=["--training.amp=false"])
        assert config.get("training.amp") is False

    def test_string_override(self):
        """String values are preserved."""
        config = ConfigLoader(overrides=["--model.encoder.backbone=resnet101"])
        assert config.get("model.encoder.backbone") == "resnet101"

    def test_list_override(self):
        """List values can be overridden."""
        config = ConfigLoader(overrides=["--model.easpp.dilations=[2,4,8,16]"])
        assert config.get("model.easpp.dilations") == [2, 4, 8, 16]

    def test_none_override(self):
        """None/null values are parsed correctly."""
        config = ConfigLoader(overrides=["--data.root=none"])
        assert config.get("data.root") is None

    def test_multiple_overrides(self):
        """Multiple overrides can be applied."""
        config = ConfigLoader(overrides=[
            "--training.epochs=50",
            "--training.batch_size=16",
            "--seed=0",
        ])
        assert config.get("training.epochs") == 50
        assert config.get("training.batch_size") == 16
        assert config.get("seed") == 0

    def test_override_takes_precedence_over_yaml(self, tmp_path):
        """CLI overrides take precedence over YAML file values."""
        yaml_content = {"training": {"epochs": 100}}
        config_file = tmp_path / "config.yaml"
        with open(config_file, "w") as f:
            yaml.dump(yaml_content, f)

        config = ConfigLoader(config_path=config_file, overrides=["--training.epochs=50"])
        assert config.get("training.epochs") == 50

    def test_creates_new_nested_keys(self):
        """Overrides can create new nested keys."""
        config = ConfigLoader(overrides=["--custom.new_param=42"])
        assert config.get("custom.new_param") == 42

    def test_malformed_override_raises(self):
        """Malformed override without '=' raises ConfigValidationError."""
        with pytest.raises(ConfigValidationError, match="Invalid override format"):
            ConfigLoader(overrides=["training.epochs"])

    def test_value_with_equals_sign(self):
        """Values containing '=' are handled correctly (split on first '=' only)."""
        config = ConfigLoader(overrides=["--data.root=path/with=equals"])
        assert config.get("data.root") == "path/with=equals"


class TestConfigLoaderValidation:
    """Test config validation with clear error messages (Req 19.3)."""

    def test_valid_defaults_pass(self):
        """Default configuration passes validation."""
        config = ConfigLoader()
        # Should not raise
        config.validate()

    def test_invalid_splits_detected(self, tmp_path):
        """Data splits not summing to 1.0 are caught."""
        yaml_content = {"data": {"train_split": 0.5, "val_split": 0.1, "test_split": 0.1}}
        config_file = tmp_path / "bad_splits.yaml"
        with open(config_file, "w") as f:
            yaml.dump(yaml_content, f)

        with pytest.raises(ConfigValidationError, match="Data splits must sum to 1.0"):
            ConfigLoader(config_path=config_file)

    def test_negative_lr_detected(self):
        """Negative learning rate is caught."""
        with pytest.raises(ConfigValidationError, match="training.optimizer.lr must be positive"):
            ConfigLoader(overrides=["--training.optimizer.lr=-0.001"])

    def test_zero_lr_detected(self):
        """Zero learning rate is caught."""
        with pytest.raises(ConfigValidationError, match="training.optimizer.lr must be positive"):
            ConfigLoader(overrides=["--training.optimizer.lr=0"])

    def test_negative_batch_size_detected(self):
        """Negative batch size is caught."""
        with pytest.raises(ConfigValidationError, match="training.batch_size must be a positive integer"):
            ConfigLoader(overrides=["--training.batch_size=-1"])

    def test_zero_epochs_detected(self):
        """Zero epochs is caught."""
        with pytest.raises(ConfigValidationError, match="training.epochs must be a positive integer"):
            ConfigLoader(overrides=["--training.epochs=0"])

    def test_negative_loss_weight_detected(self):
        """Negative loss weight is caught."""
        with pytest.raises(ConfigValidationError, match="training.loss_weights.segmentation must be non-negative"):
            ConfigLoader(overrides=["--training.loss_weights.segmentation=-1.0"])

    def test_negative_num_workers_detected(self):
        """Negative num_workers is caught."""
        with pytest.raises(ConfigValidationError, match="data.num_workers must be a non-negative integer"):
            ConfigLoader(overrides=["--data.num_workers=-1"])

    def test_missing_required_section_detected(self):
        """Missing required section is caught with clear message."""
        config = ConfigLoader()
        # Manually remove a required section to test validation
        del config._config["model"]
        with pytest.raises(ConfigValidationError, match="Missing required config section: 'model'"):
            config.validate()

    def test_missing_model_subsection_detected(self):
        """Missing required model subsection is caught."""
        config = ConfigLoader()
        del config._config["model"]["encoder"]
        with pytest.raises(ConfigValidationError, match="Missing required model subsection: 'model.encoder'"):
            config.validate()

    def test_multiple_errors_reported(self):
        """Multiple validation errors are collected and reported together."""
        config = ConfigLoader()
        config._config["training"]["epochs"] = -1
        config._config["training"]["batch_size"] = 0
        config._config["training"]["optimizer"]["lr"] = -0.1
        with pytest.raises(ConfigValidationError) as exc_info:
            config.validate()
        error_msg = str(exc_info.value)
        assert "training.epochs" in error_msg
        assert "training.batch_size" in error_msg
        assert "training.optimizer.lr" in error_msg

    def test_invalid_split_range_detected(self):
        """Split values outside [0, 1] are caught."""
        config = ConfigLoader()
        config._config["data"]["train_split"] = 1.5
        config._config["data"]["val_split"] = -0.3
        config._config["data"]["test_split"] = -0.2
        with pytest.raises(ConfigValidationError, match="data.train_split must be between 0 and 1"):
            config.validate()

    def test_cyclegan_negative_lr_detected(self):
        """CycleGAN negative learning rate is caught."""
        config = ConfigLoader()
        config._config["cyclegan"]["training"]["lr"] = -0.0002
        with pytest.raises(ConfigValidationError, match="cyclegan.training.lr must be positive"):
            config.validate()


class TestConfigLoaderDotNotationAccess:
    """Test nested dot-notation access."""

    def test_top_level_access(self):
        """Top-level key access works."""
        config = ConfigLoader()
        assert config.get("seed") == 42

    def test_nested_access(self):
        """Nested key access works."""
        config = ConfigLoader()
        assert config.get("model.encoder.backbone") == "resnet50"

    def test_deep_nested_access(self):
        """Deep nested key access works."""
        config = ConfigLoader()
        assert config.get("training.loss_weights.segmentation") == 1.5

    def test_missing_key_returns_default(self):
        """Missing key returns the specified default."""
        config = ConfigLoader()
        assert config.get("nonexistent.key") is None
        assert config.get("nonexistent.key", "fallback") == "fallback"

    def test_partial_path_missing_returns_default(self):
        """Partial path failure returns default."""
        config = ConfigLoader()
        assert config.get("model.nonexistent.key") is None

    def test_get_returns_dict_for_section(self):
        """Getting a section key returns the nested dictionary."""
        config = ConfigLoader()
        optimizer = config.get("training.optimizer")
        assert isinstance(optimizer, dict)
        assert optimizer["lr"] == 0.0001

    def test_set_and_get_roundtrip(self):
        """set() followed by get() retrieves the value."""
        config = ConfigLoader()
        config.set("custom.nested.value", 99)
        assert config.get("custom.nested.value") == 99


class TestConfigLoaderYAMLRoundTrip:
    """Test YAML serialization round-trip."""

    def test_write_and_read_yaml(self, tmp_path):
        """Writing config to YAML and reading back produces equivalent config."""
        original = ConfigLoader()
        yaml_file = tmp_path / "output.yaml"
        original.to_yaml(yaml_file)

        loaded = ConfigLoader(config_path=yaml_file)
        assert loaded.get("training.epochs") == original.get("training.epochs")
        assert loaded.get("model.encoder.backbone") == original.get("model.encoder.backbone")
        assert loaded.get("seed") == original.get("seed")
        assert loaded.get("training.loss_weights") == original.get("training.loss_weights")


class TestConfigLoaderParseValue:
    """Test the value parser for CLI overrides."""

    def test_parses_integers(self):
        """Integers are parsed correctly."""
        config = ConfigLoader()
        assert config._parse_value("42") == 42
        assert config._parse_value("-1") == -1
        assert config._parse_value("0") == 0

    def test_parses_floats(self):
        """Floats are parsed correctly."""
        config = ConfigLoader()
        assert config._parse_value("3.14") == 3.14
        assert config._parse_value("1e-3") == 0.001
        assert config._parse_value("2.5e2") == 250.0

    def test_parses_booleans(self):
        """Booleans are parsed correctly."""
        config = ConfigLoader()
        assert config._parse_value("true") is True
        assert config._parse_value("True") is True
        assert config._parse_value("yes") is True
        assert config._parse_value("false") is False
        assert config._parse_value("False") is False
        assert config._parse_value("no") is False

    def test_parses_none(self):
        """None values are parsed correctly."""
        config = ConfigLoader()
        assert config._parse_value("none") is None
        assert config._parse_value("null") is None
        assert config._parse_value("~") is None

    def test_parses_lists(self):
        """Lists are parsed correctly."""
        config = ConfigLoader()
        assert config._parse_value("[1, 2, 3]") == [1, 2, 3]
        assert config._parse_value("[3.0, 6.0]") == [3.0, 6.0]
        assert config._parse_value("[]") == []

    def test_parses_quoted_strings(self):
        """Quoted strings have quotes stripped."""
        config = ConfigLoader()
        assert config._parse_value('"hello"') == "hello"
        assert config._parse_value("'world'") == "world"

    def test_parses_plain_strings(self):
        """Unquoted strings are preserved."""
        config = ConfigLoader()
        assert config._parse_value("resnet101") == "resnet101"
        assert config._parse_value("./path/to/data") == "./path/to/data"
