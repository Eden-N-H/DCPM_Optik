"""Configuration management with YAML loading, defaults, validation, and CLI overrides.

This module provides the ConfigLoader class for centralized configuration management.
It supports:
- Loading configuration from YAML files (Req 19.1)
- Documented default values for all optional parameters (Req 19.2)
- Startup validation with clear error messages (Req 19.3)
- CLI dot-notation overrides e.g. --training.lr=1e-3 (Req 19.4)
"""

from pathlib import Path
from typing import Any, Dict, Optional, List
import yaml
import copy


class ConfigValidationError(Exception):
    """Raised when config validation fails with detailed error messages."""
    pass


class ConfigLoader:
    """YAML config loading with defaults, validation, and CLI overrides.

    The ConfigLoader manages the full pipeline configuration including model architecture,
    training hyperparameters, data paths, augmentation parameters, and loss weights.
    Configuration is loaded in three layers (each overriding the previous):
    1. Built-in defaults (DEFAULTS class variable)
    2. YAML file values
    3. CLI dot-notation overrides

    Usage:
        # Load from YAML with CLI overrides
        config = ConfigLoader(
            config_path=Path("configs/default.yaml"),
            overrides=["--training.lr=1e-3", "--training.batch_size=16"]
        )
        lr = config.get("training.optimizer.lr")

        # Defaults only
        config = ConfigLoader()
    """

    # Documented default values for all optional parameters.
    # Required parameters (like data.root) still have sensible defaults but
    # validation will flag issues with obviously invalid values.
    DEFAULTS: Dict[str, Any] = {
        # Data loading configuration
        "data": {
            "root": "./data/road_quality",  # Root directory for dataset
            "train_split": 0.8,             # Fraction of data for training
            "val_split": 0.1,               # Fraction of data for validation
            "test_split": 0.1,              # Fraction of data for testing
            "num_workers": 4,               # DataLoader worker processes
            "pin_memory": True,             # Pin memory for faster GPU transfer
            "prefetch_factor": 2,           # Batches to prefetch per worker
        },
        # Scene generation parameters (Blender synthetic data)
        "scene_generation": {
            "road": {
                "lanes": [1, 4],            # Min/max lane count
                "lane_width": [3.0, 3.75],  # Lane width range in meters
                "road_length": [50, 200],   # Road length range in meters
            },
            "defects": {
                "count": [1, 10],           # Min/max defects per scene
                "types": ["crack", "pothole", "puddle", "patch", "manhole"],
                "overlap_threshold": 0.25,  # Max allowed overlap fraction
            },
            "camera": {
                "dashcam": {
                    "height": [1.2, 1.5],   # Height range in meters
                    "pitch": [-15, -5],     # Pitch range in degrees
                },
                "drone": {
                    "height": [8, 15],      # Height range in meters
                    "pitch": [-90, -60],    # Pitch range in degrees
                },
            },
            "domain_randomization": {
                "hdri_count": 20,           # Number of HDRI environment maps
                "vehicles": [0, 5],         # Min/max vehicle count
                "weather": ["clear", "overcast", "rain"],
            },
            "dataset_size": 16036,          # Target total images
            "render_size": 512,             # Render resolution (pixels)
        },
        # CycleGAN domain adaptation configuration
        "cyclegan": {
            "input_size": 256,              # Input spatial resolution
            "input_nc": 4,                  # Input channels (RGB + mask)
            "output_nc": 3,                 # Output channels (RGB)
            "ngf": 64,                      # Generator base filters
            "ndf": 64,                      # Discriminator base filters
            "n_blocks": 9,                  # Residual blocks in generator
            "training": {
                "epochs": 200,              # Total training epochs
                "lr": 0.0002,               # Learning rate
                "beta1": 0.5,               # Adam beta1
                "beta2": 0.999,             # Adam beta2
                "decay_start_epoch": 100,   # Epoch to start LR decay
                "pool_size": 50,            # Image history buffer size
                "lambda_cycle": 10.0,       # Cycle consistency loss weight
                "lambda_identity": 0.5,     # Identity loss weight
                "lambda_defect": 5.0,       # Defect preservation loss weight
            },
        },
        # Multi-task model architecture
        "model": {
            "encoder": {
                "backbone": "resnet50",     # Backbone architecture
                "pretrained": True,         # Use ImageNet pretrained weights
                "dsc_stages": [3, 4],       # Stages using depthwise separable convs
            },
            "view_embedding": {
                "num_views": 2,             # Number of viewpoint types
                "embed_dim": 32,            # Embedding dimension
            },
            "easpp": {
                "dilations": [3, 6, 12, 18],  # ASPP dilation rates
                "out_channels": 256,        # Output channel count
            },
            "soa": {
                "reduction": 16,            # Channel attention reduction ratio
                "alpha": 0.3,               # High-pass enhancement scale
                "gaussian_kernel": 7,       # Gaussian kernel size for high-pass
                "gaussian_sigma": 1.0,      # Gaussian sigma for high-pass
            },
            "decoder": {
                "channels": [256, 128, 64],  # Decoder block channel counts
            },
            "heads": {
                "segmentation": {
                    "num_classes": 8,        # Number of segmentation classes
                    "hidden_channels": 128,  # Hidden layer channels
                },
                "severity": {
                    "hidden_channels": 128,  # Hidden layer channels
                },
                "depth": {
                    "hidden_channels": 128,  # Hidden layer channels
                },
                "camera": {
                    "hidden_dim": [512, 256],  # FC layer dimensions
                    "intrinsic_params": 4,   # Number of intrinsic parameters
                    "extrinsic_params": 6,   # Number of extrinsic parameters
                },
            },
        },
        # Training hyperparameters
        "training": {
            "epochs": 200,                  # Maximum training epochs
            "batch_size": 8,                # Training batch size
            "optimizer": {
                "type": "adam",              # Optimizer type
                "lr": 0.0001,               # Learning rate
                "beta1": 0.9,               # Adam beta1
                "beta2": 0.999,             # Adam beta2
                "weight_decay": 0.00001,    # L2 regularization
            },
            "scheduler": {
                "type": "reduce_on_plateau",  # LR scheduler type
                "patience": 10,             # Epochs before LR reduction
                "factor": 0.5,              # LR reduction factor
            },
            "early_stopping_patience": 30,  # Epochs before early stopping
            "loss_weights": {
                "segmentation": 1.5,        # Segmentation loss weight
                "depth": 1.0,               # Depth loss weight
                "camera": 0.3,              # Camera loss weight
                "adversarial": 0.1,         # Adversarial loss weight
                "view": 0.1,                # View consistency loss weight
            },
            "amp": True,                    # Enable Automatic Mixed Precision
            "grad_clip_norm": 1.0,          # Maximum gradient norm
        },
        # Augmentation parameters for training data
        "augmentation": {
            "horizontal_flip": True,        # Random horizontal flip
            "rotation_range": 10,           # Random rotation ±degrees
            "crop_size": 480,               # Random crop size (training)
            "color_jitter": {
                "brightness": 0.2,          # Brightness jitter factor
                "contrast": 0.2,            # Contrast jitter factor
                "saturation": 0.1,          # Saturation jitter factor
                "hue": 0.05,                # Hue jitter factor
            },
        },
        # Domain adaptation configuration
        "domain_adaptation": {
            "lambda_adv": 0.1,              # Adversarial loss scaling factor
            "feature_disc": {
                "channels": [256, 128, 1],  # Feature discriminator channels
                "kernel_size": 3,           # Conv kernel size
                "stride": 2,                # Conv stride
            },
            "logit_disc": {
                "channels": [256, 128, 1],  # Logit discriminator channels
                "kernel_size": 3,           # Conv kernel size
                "stride": 2,                # Conv stride
            },
        },
        # 3D reconstruction parameters
        "reconstruction": {
            "depth_confidence_threshold": 0.5,  # Min depth confidence
            "height_range": [-0.5, 0.5],    # Valid height range (meters)
            "bev_resolution": 0.02,         # BEV map resolution (m/pixel)
        },
        # Evaluation metric configuration
        "evaluation": {
            "segmentation": ["miou", "per_class_iou", "pixel_accuracy", "mean_class_accuracy"],
            "depth": ["rmse", "abs_rel", "delta_1", "delta_2", "delta_3"],
            "camera": ["intrinsic_mae", "rotation_geodesic", "translation_error"],
            "severity": ["mae", "pearson_correlation"],
        },
        # Logging and experiment tracking
        "logging": {
            "tb_log_interval": 100,         # TensorBoard log interval (steps)
            "console_log_interval": 10,     # Console log interval (steps)
            "checkpoint_interval": 5,       # Checkpoint save interval (epochs)
        },
        # Random seed for reproducibility
        "seed": 42,
    }

    def __init__(self, config_path: Optional[Path] = None, overrides: Optional[List[str]] = None):
        """Initialize ConfigLoader with optional YAML file and CLI overrides.

        Configuration is assembled in priority order:
        1. DEFAULTS (lowest priority)
        2. YAML file values
        3. CLI overrides (highest priority)

        After merging, validate() is called to check for invalid values.

        Args:
            config_path: Path to a YAML configuration file. If None, only defaults are used.
            overrides: List of CLI override strings in dot-notation format,
                       e.g. ["--training.lr=1e-3", "training.batch_size=16"].
                       Leading dashes are stripped automatically.

        Raises:
            FileNotFoundError: If config_path is specified but file doesn't exist.
            ConfigValidationError: If the final configuration has invalid values.
        """
        # Start with a deep copy of defaults
        self._config = copy.deepcopy(self.DEFAULTS)

        # Load and merge YAML file if provided
        if config_path is not None:
            config_path = Path(config_path)
            if not config_path.exists():
                raise FileNotFoundError(f"Configuration file not found: {config_path}")
            with open(config_path, "r") as f:
                file_config = yaml.safe_load(f)
            if file_config is not None:
                self._config = self._deep_merge(self._config, file_config)

        # Apply CLI overrides
        if overrides:
            self._apply_overrides(overrides)

        # Validate the final configuration
        self.validate()

    def _deep_merge(self, base: Dict, override: Dict) -> Dict:
        """Recursively merge override dict into base dict.

        Values in override take precedence. Nested dicts are merged recursively;
        non-dict values in override replace values in base.

        Args:
            base: The base dictionary (defaults or accumulated config).
            override: The dictionary with values to merge in.

        Returns:
            A new dictionary with merged values.
        """
        result = copy.deepcopy(base)
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = copy.deepcopy(value)
        return result

    def _apply_overrides(self, overrides: List[str]) -> None:
        """Apply CLI dot-notation overrides like --training.lr=1e-3.

        Supports formats:
        - "key.path=value"
        - "--key.path=value"
        Leading dashes are stripped automatically.

        Args:
            overrides: List of strings in "key.path=value" format.

        Raises:
            ConfigValidationError: If an override string is malformed (missing '=').
        """
        for override in overrides:
            if "=" not in override:
                raise ConfigValidationError(
                    f"Invalid override format '{override}': expected 'key.path=value'"
                )
            # Split on first '=' only to allow '=' in values
            key_path, value_str = override.split("=", 1)
            key_path = key_path.strip().lstrip("-")  # Strip leading -- from CLI args
            value = self._parse_value(value_str.strip())

            # Navigate to the parent dict and set the value
            keys = key_path.split(".")
            current = self._config
            for key in keys[:-1]:
                if key not in current:
                    current[key] = {}
                elif not isinstance(current[key], dict):
                    current[key] = {}
                current = current[key]
            current[keys[-1]] = value

    def _parse_value(self, value_str: str) -> Any:
        """Parse a string value to the appropriate Python type.

        Handles the following types (in order of precedence):
        - None/null: "none", "null", "~"
        - Booleans: "true", "yes", "false", "no"
        - Integers: whole numbers like "42", "-1"
        - Floats: decimal or scientific notation like "1e-3", "0.5"
        - Lists: JSON-style brackets like "[1, 2, 3]"
        - Strings: everything else, with optional quote stripping

        Args:
            value_str: The string representation of the value.

        Returns:
            The parsed Python value.
        """
        # None
        if value_str.lower() in ("none", "null", "~"):
            return None

        # Booleans
        if value_str.lower() in ("true", "yes"):
            return True
        if value_str.lower() in ("false", "no"):
            return False

        # Try integer
        try:
            return int(value_str)
        except ValueError:
            pass

        # Try float
        try:
            return float(value_str)
        except ValueError:
            pass

        # Try list (JSON-style brackets)
        if value_str.startswith("[") and value_str.endswith("]"):
            inner = value_str[1:-1].strip()
            if not inner:
                return []
            items = [self._parse_value(item.strip()) for item in inner.split(",")]
            return items

        # Return as string (strip quotes if present)
        if (value_str.startswith('"') and value_str.endswith('"')) or \
           (value_str.startswith("'") and value_str.endswith("'")):
            return value_str[1:-1]

        return value_str

    def validate(self) -> None:
        """Validate configuration values at startup.

        Checks for invalid or missing required parameters and raises
        ConfigValidationError with clear, specific error messages identifying
        each invalid parameter.

        Validates:
            - Required top-level sections exist (data, model, training)
            - Data splits sum to 1.0 (within tolerance)
            - Data split values are in [0, 1]
            - Learning rates are positive
            - Batch size is positive integer
            - Number of workers is non-negative integer
            - Epoch counts are positive integers
            - Loss weights are non-negative
            - Required model subsections exist (encoder, decoder, heads)
            - CycleGAN learning rate is positive (if section present)
            - Gradient clip norm is positive (if specified)
            - Scheduler patience is positive (if specified)

        Raises:
            ConfigValidationError: With detailed message listing all validation failures.
        """
        errors = []

        # Check required top-level sections
        required_sections = ["data", "model", "training"]
        for section in required_sections:
            if section not in self._config:
                errors.append(f"Missing required config section: '{section}'")

        # If required sections are missing, report immediately
        if errors:
            raise ConfigValidationError(
                "Configuration validation failed:\n  - " + "\n  - ".join(errors)
            )

        # Validate data splits sum to ~1.0
        data = self._config.get("data", {})
        train_split = data.get("train_split", 0)
        val_split = data.get("val_split", 0)
        test_split = data.get("test_split", 0)
        split_sum = train_split + val_split + test_split
        if abs(split_sum - 1.0) > 0.01:
            errors.append(
                f"Data splits must sum to 1.0 (got {split_sum:.3f}: "
                f"train={train_split}, val={val_split}, test={test_split})"
            )

        # Validate splits are in valid range
        for name, val in [("train_split", train_split), ("val_split", val_split), ("test_split", test_split)]:
            if not isinstance(val, (int, float)) or not (0.0 <= val <= 1.0):
                errors.append(f"data.{name} must be between 0 and 1, got {val}")

        # Validate num_workers
        num_workers = data.get("num_workers", 0)
        if not isinstance(num_workers, int) or num_workers < 0:
            errors.append(f"data.num_workers must be a non-negative integer, got {num_workers}")

        # Validate training parameters
        training = self._config.get("training", {})
        epochs = training.get("epochs", 0)
        if not isinstance(epochs, int) or epochs <= 0:
            errors.append(f"training.epochs must be a positive integer, got {epochs}")

        batch_size = training.get("batch_size", 0)
        if not isinstance(batch_size, int) or batch_size <= 0:
            errors.append(f"training.batch_size must be a positive integer, got {batch_size}")

        # Validate optimizer learning rate
        optimizer = training.get("optimizer", {})
        if isinstance(optimizer, dict):
            lr = optimizer.get("lr", 0)
            if not isinstance(lr, (int, float)) or lr <= 0:
                errors.append(f"training.optimizer.lr must be positive, got {lr}")

        # Validate loss weights are non-negative
        loss_weights = training.get("loss_weights", {})
        if isinstance(loss_weights, dict):
            for name, weight in loss_weights.items():
                if not isinstance(weight, (int, float)) or weight < 0:
                    errors.append(f"training.loss_weights.{name} must be non-negative, got {weight}")

        # Validate gradient clip norm is positive (if specified)
        grad_clip = training.get("grad_clip_norm")
        if grad_clip is not None:
            if not isinstance(grad_clip, (int, float)) or grad_clip <= 0:
                errors.append(f"training.grad_clip_norm must be positive, got {grad_clip}")

        # Validate scheduler patience is positive (if specified)
        scheduler = training.get("scheduler", {})
        if isinstance(scheduler, dict):
            patience = scheduler.get("patience")
            if patience is not None and (not isinstance(patience, int) or patience <= 0):
                errors.append(f"training.scheduler.patience must be a positive integer, got {patience}")
            factor = scheduler.get("factor")
            if factor is not None and (not isinstance(factor, (int, float)) or not (0 < factor < 1)):
                errors.append(f"training.scheduler.factor must be between 0 and 1 (exclusive), got {factor}")

        # Validate cyclegan learning rate if section exists
        cyclegan = self._config.get("cyclegan", {})
        if isinstance(cyclegan, dict):
            cg_training = cyclegan.get("training", {})
            if isinstance(cg_training, dict):
                cg_lr = cg_training.get("lr", 0)
                if isinstance(cg_lr, (int, float)) and cg_lr <= 0:
                    errors.append(f"cyclegan.training.lr must be positive, got {cg_lr}")

        # Validate model section has required subsections
        model = self._config.get("model", {})
        required_model_keys = ["encoder", "decoder", "heads"]
        for key in required_model_keys:
            if key not in model:
                errors.append(f"Missing required model subsection: 'model.{key}'")

        # Validate early stopping patience
        es_patience = training.get("early_stopping_patience")
        if es_patience is not None and (not isinstance(es_patience, int) or es_patience <= 0):
            errors.append(f"training.early_stopping_patience must be a positive integer, got {es_patience}")

        if errors:
            raise ConfigValidationError(
                "Configuration validation failed:\n  - " + "\n  - ".join(errors)
            )

    def get(self, key: str, default: Any = None) -> Any:
        """Get a config value using dot-notation key path.

        Supports nested access like "model.encoder.backbone" which traverses
        the config dictionary hierarchy.

        Args:
            key: Dot-separated path, e.g. "model.encoder.backbone".
            default: Value to return if the key path doesn't exist.

        Returns:
            The config value at the key path, or default if not found.

        Examples:
            >>> config.get("training.optimizer.lr")
            0.0001
            >>> config.get("model.encoder.backbone")
            'resnet50'
            >>> config.get("nonexistent.key", "fallback")
            'fallback'
        """
        keys = key.split(".")
        current = self._config
        for k in keys:
            if isinstance(current, dict) and k in current:
                current = current[k]
            else:
                return default
        return current

    def set(self, key: str, value: Any) -> None:
        """Set a config value using dot-notation key path.

        Creates intermediate dictionaries as needed.

        Args:
            key: Dot-separated path, e.g. "training.optimizer.lr".
            value: The value to set.
        """
        keys = key.split(".")
        current = self._config
        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            elif not isinstance(current[k], dict):
                current[k] = {}
            current = current[k]
        current[keys[-1]] = value

    @property
    def config(self) -> Dict:
        """Return the full configuration dictionary (read-only copy)."""
        return copy.deepcopy(self._config)

    def to_yaml(self, path: Path) -> None:
        """Write the current configuration to a YAML file.

        Args:
            path: Path where the YAML file will be written.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(self._config, f, default_flow_style=False, sort_keys=False)

    def __repr__(self) -> str:
        """Return a string representation showing the config source."""
        return f"ConfigLoader(keys={list(self._config.keys())})"
