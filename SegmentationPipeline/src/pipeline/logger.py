"""Pipeline logging system with configurable verbosity and dual output.

Provides structured logging to both stdout and a log file in the configured
output directory. Each log entry includes an ISO 8601 timestamp, log level,
component name, and message text.

Validates: Requirements 9.1, 9.2
"""

import logging
import os
import sys
from datetime import datetime, timezone


# Custom log format: ISO 8601 timestamp [LEVEL] [component] message
_LOG_FORMAT = "%(asctime)s [%(levelname)s] [%(name)s] %(message)s"

# ISO 8601 date format
_DATE_FORMAT = "%Y-%m-%dT%H:%M:%S"

_VALID_LEVELS = {"DEBUG", "INFO", "WARNING", "ERROR"}

_configured = False


def setup_logging(log_level: str = "INFO", output_directory: str = "output") -> None:
    """Configure the root logger with both stdout and file handlers.

    Sets up dual output: a StreamHandler writing to stdout and a FileHandler
    writing to a log file in the specified output directory. Both handlers
    use the same ISO 8601 formatted log entries.

    Args:
        log_level: Logging verbosity level. Must be one of DEBUG, INFO,
            WARNING, ERROR. Defaults to "INFO".
        output_directory: Directory where the log file will be created.
            The directory will be created if it does not exist.
            Defaults to "output".

    Raises:
        ValueError: If log_level is not a valid level string.
        OSError: If the output directory cannot be created or is not writable.
    """
    global _configured

    level_str = log_level.upper()
    if level_str not in _VALID_LEVELS:
        raise ValueError(
            f"Invalid log level: '{log_level}'. "
            f"Must be one of: {', '.join(sorted(_VALID_LEVELS))}"
        )

    numeric_level = getattr(logging, level_str)

    # Create output directory if it doesn't exist
    os.makedirs(output_directory, exist_ok=True)

    # Build the log file path
    log_file_path = os.path.join(output_directory, "pipeline.log")

    # Get the root logger and clear any existing handlers
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)

    # Remove existing handlers to avoid duplicates on re-configuration
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
        handler.close()

    # Create formatter
    formatter = logging.Formatter(fmt=_LOG_FORMAT, datefmt=_DATE_FORMAT)

    # Stdout handler
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(numeric_level)
    stream_handler.setFormatter(formatter)
    root_logger.addHandler(stream_handler)

    # File handler
    file_handler = logging.FileHandler(log_file_path, mode="a", encoding="utf-8")
    file_handler.setLevel(numeric_level)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    _configured = True


def get_logger(component_name: str) -> logging.Logger:
    """Get a logger instance for a specific pipeline component.

    Returns a named logger that will include the component name in all
    log entries. If setup_logging() has not been called yet, a basic
    stdout-only configuration is applied as a fallback.

    Args:
        component_name: Name of the pipeline component (e.g., "FrameIngester",
            "YOLODetector", "SAM2Segmenter").

    Returns:
        A configured logging.Logger instance with the component name.
    """
    if not _configured:
        # Provide a minimal fallback so logs aren't lost if setup wasn't called
        _setup_fallback()

    return logging.getLogger(component_name)


def _setup_fallback() -> None:
    """Apply a minimal logging configuration as a fallback."""
    global _configured

    root_logger = logging.getLogger()
    if not root_logger.handlers:
        formatter = logging.Formatter(fmt=_LOG_FORMAT, datefmt=_DATE_FORMAT)
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(formatter)
        root_logger.addHandler(stream_handler)
        root_logger.setLevel(logging.INFO)

    _configured = True
