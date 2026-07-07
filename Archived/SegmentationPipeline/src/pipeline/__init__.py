"""Pipeline components for road defect segmentation.

Public API:
    PipelineOrchestrator - Coordinates all pipeline components and manages the processing loop.
    PipelineConfig - Dataclass holding all pipeline configuration parameters.
    ConfigManager - Loads, validates, and provides access to pipeline configuration.
"""

from src.pipeline.config_manager import ConfigManager
from src.pipeline.models import PipelineConfig
from src.pipeline.orchestrator import PipelineOrchestrator

__all__ = [
    "PipelineOrchestrator",
    "PipelineConfig",
    "ConfigManager",
]
