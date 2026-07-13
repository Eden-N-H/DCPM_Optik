"""Pytest configuration and hypothesis profiles for the road quality pipeline tests."""

import pytest
from hypothesis import settings, HealthCheck, Phase

# Hypothesis profiles
settings.register_profile(
    "ci",
    max_examples=200,
    suppress_health_check=[HealthCheck.too_slow],
    deadline=None,
)

settings.register_profile(
    "dev",
    max_examples=50,
    suppress_health_check=[HealthCheck.too_slow],
    deadline=None,
)

settings.register_profile(
    "debug",
    max_examples=10,
    suppress_health_check=[HealthCheck.too_slow],
    deadline=None,
    phases=[Phase.explicit, Phase.generate],
)

settings.load_profile("dev")


@pytest.fixture
def device():
    """Provide the appropriate torch device for testing."""
    import torch

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def default_config():
    """Load and return the default configuration dictionary."""
    import yaml
    from pathlib import Path

    config_path = Path(__file__).parent.parent / "configs" / "default.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)
