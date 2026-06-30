"""Unit tests for the pipeline logging system.

Tests cover:
- Log level configuration (DEBUG, INFO, WARNING, ERROR)
- Dual output to stdout and log file
- Log entry format (ISO 8601 timestamp, level, component, message)
- Invalid log level handling
- Output directory creation
- Component-specific logger retrieval
"""

import logging
import os
import re
import tempfile

import pytest

from src.pipeline.logger import get_logger, setup_logging, _LOG_FORMAT, _VALID_LEVELS
import src.pipeline.logger as logger_module


@pytest.fixture(autouse=True)
def reset_logging():
    """Reset logging state between tests."""
    # Reset the module-level configured flag
    logger_module._configured = False

    # Clear all handlers from the root logger
    root = logging.getLogger()
    for handler in root.handlers[:]:
        root.removeHandler(handler)
        handler.close()

    # Clear any component loggers that might have handlers
    logging.Logger.manager.loggerDict.clear()

    yield

    # Cleanup after test
    logger_module._configured = False
    root = logging.getLogger()
    for handler in root.handlers[:]:
        root.removeHandler(handler)
        handler.close()


class TestSetupLogging:
    """Tests for the setup_logging function."""

    def test_creates_output_directory(self, tmp_path):
        """setup_logging creates the output directory if it doesn't exist."""
        output_dir = str(tmp_path / "logs" / "nested")
        setup_logging(log_level="INFO", output_directory=output_dir)
        assert os.path.isdir(output_dir)

    def test_creates_log_file(self, tmp_path):
        """setup_logging creates a pipeline.log file in the output directory."""
        output_dir = str(tmp_path / "output")
        setup_logging(log_level="INFO", output_directory=output_dir)
        assert os.path.isfile(os.path.join(output_dir, "pipeline.log"))

    def test_sets_root_logger_level(self, tmp_path):
        """setup_logging sets the root logger to the specified level."""
        output_dir = str(tmp_path / "output")
        setup_logging(log_level="DEBUG", output_directory=output_dir)
        assert logging.getLogger().level == logging.DEBUG

    def test_default_level_is_info(self, tmp_path):
        """setup_logging defaults to INFO level."""
        output_dir = str(tmp_path / "output")
        setup_logging(output_directory=output_dir)
        assert logging.getLogger().level == logging.INFO

    def test_invalid_log_level_raises_error(self, tmp_path):
        """setup_logging raises ValueError for invalid log levels."""
        output_dir = str(tmp_path / "output")
        with pytest.raises(ValueError, match="Invalid log level"):
            setup_logging(log_level="TRACE", output_directory=output_dir)

    def test_case_insensitive_log_level(self, tmp_path):
        """setup_logging accepts log levels case-insensitively."""
        output_dir = str(tmp_path / "output")
        setup_logging(log_level="debug", output_directory=output_dir)
        assert logging.getLogger().level == logging.DEBUG

    def test_has_two_handlers(self, tmp_path):
        """setup_logging configures exactly two handlers (stream + file)."""
        output_dir = str(tmp_path / "output")
        setup_logging(log_level="INFO", output_directory=output_dir)
        root = logging.getLogger()
        assert len(root.handlers) == 2

    def test_has_stream_handler(self, tmp_path):
        """setup_logging adds a StreamHandler."""
        output_dir = str(tmp_path / "output")
        setup_logging(log_level="INFO", output_directory=output_dir)
        root = logging.getLogger()
        stream_handlers = [h for h in root.handlers if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)]
        assert len(stream_handlers) == 1

    def test_has_file_handler(self, tmp_path):
        """setup_logging adds a FileHandler."""
        output_dir = str(tmp_path / "output")
        setup_logging(log_level="INFO", output_directory=output_dir)
        root = logging.getLogger()
        file_handlers = [h for h in root.handlers if isinstance(h, logging.FileHandler)]
        assert len(file_handlers) == 1

    def test_reconfiguration_replaces_handlers(self, tmp_path):
        """Calling setup_logging twice doesn't duplicate handlers."""
        output_dir = str(tmp_path / "output")
        setup_logging(log_level="INFO", output_directory=output_dir)
        setup_logging(log_level="DEBUG", output_directory=output_dir)
        root = logging.getLogger()
        assert len(root.handlers) == 2


class TestLogFormat:
    """Tests for log entry format compliance."""

    def test_log_entry_contains_iso8601_timestamp(self, tmp_path):
        """Log entries contain an ISO 8601 formatted timestamp."""
        output_dir = str(tmp_path / "output")
        setup_logging(log_level="INFO", output_directory=output_dir)
        logger = get_logger("TestComponent")
        logger.info("test message")

        log_file = os.path.join(output_dir, "pipeline.log")
        with open(log_file) as f:
            content = f.read()

        # ISO 8601 pattern: YYYY-MM-DDThh:mm:ss
        iso_pattern = r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}"
        assert re.search(iso_pattern, content)

    def test_log_entry_contains_level(self, tmp_path):
        """Log entries contain the log level in brackets."""
        output_dir = str(tmp_path / "output")
        setup_logging(log_level="INFO", output_directory=output_dir)
        logger = get_logger("TestComponent")
        logger.info("test message")

        log_file = os.path.join(output_dir, "pipeline.log")
        with open(log_file) as f:
            content = f.read()

        assert "[INFO]" in content

    def test_log_entry_contains_component_name(self, tmp_path):
        """Log entries contain the component name in brackets."""
        output_dir = str(tmp_path / "output")
        setup_logging(log_level="INFO", output_directory=output_dir)
        logger = get_logger("FrameIngester")
        logger.info("Processing frame")

        log_file = os.path.join(output_dir, "pipeline.log")
        with open(log_file) as f:
            content = f.read()

        assert "[FrameIngester]" in content

    def test_log_entry_contains_message(self, tmp_path):
        """Log entries contain the actual message text."""
        output_dir = str(tmp_path / "output")
        setup_logging(log_level="INFO", output_directory=output_dir)
        logger = get_logger("TestComponent")
        logger.info("Processing frame video001_frame_0042")

        log_file = os.path.join(output_dir, "pipeline.log")
        with open(log_file) as f:
            content = f.read()

        assert "Processing frame video001_frame_0042" in content

    def test_full_log_format_matches_spec(self, tmp_path):
        """Log format matches: {timestamp} [{level}] [{component}] {message}."""
        output_dir = str(tmp_path / "output")
        setup_logging(log_level="INFO", output_directory=output_dir)
        logger = get_logger("FrameIngester")
        logger.info("Processing frame video001_frame_0042")

        log_file = os.path.join(output_dir, "pipeline.log")
        with open(log_file) as f:
            content = f.read().strip()

        # Full expected format pattern
        pattern = r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2} \[INFO\] \[FrameIngester\] Processing frame video001_frame_0042$"
        assert re.match(pattern, content), f"Log entry does not match expected format: {content}"


class TestGetLogger:
    """Tests for the get_logger function."""

    def test_returns_named_logger(self, tmp_path):
        """get_logger returns a logger with the specified component name."""
        output_dir = str(tmp_path / "output")
        setup_logging(log_level="INFO", output_directory=output_dir)
        logger = get_logger("YOLODetector")
        assert logger.name == "YOLODetector"

    def test_fallback_when_not_configured(self, caplog):
        """get_logger works even if setup_logging hasn't been called."""
        logger = get_logger("TestComponent")
        with caplog.at_level(logging.INFO):
            logger.info("fallback test")
        assert "fallback test" in caplog.text

    def test_different_components_get_different_loggers(self, tmp_path):
        """Different component names return distinct logger instances."""
        output_dir = str(tmp_path / "output")
        setup_logging(log_level="INFO", output_directory=output_dir)
        logger1 = get_logger("Component1")
        logger2 = get_logger("Component2")
        assert logger1 is not logger2
        assert logger1.name != logger2.name


class TestLogLevels:
    """Tests for log level filtering."""

    def test_debug_level_logs_all(self, tmp_path):
        """DEBUG level captures messages at all levels."""
        output_dir = str(tmp_path / "output")
        setup_logging(log_level="DEBUG", output_directory=output_dir)
        logger = get_logger("Test")
        logger.debug("debug msg")
        logger.info("info msg")
        logger.warning("warn msg")
        logger.error("error msg")

        log_file = os.path.join(output_dir, "pipeline.log")
        with open(log_file) as f:
            content = f.read()

        assert "debug msg" in content
        assert "info msg" in content
        assert "warn msg" in content
        assert "error msg" in content

    def test_info_level_filters_debug(self, tmp_path):
        """INFO level does not capture DEBUG messages."""
        output_dir = str(tmp_path / "output")
        setup_logging(log_level="INFO", output_directory=output_dir)
        logger = get_logger("Test")
        logger.debug("debug msg")
        logger.info("info msg")

        log_file = os.path.join(output_dir, "pipeline.log")
        with open(log_file) as f:
            content = f.read()

        assert "debug msg" not in content
        assert "info msg" in content

    def test_warning_level_filters_info(self, tmp_path):
        """WARNING level does not capture INFO or DEBUG messages."""
        output_dir = str(tmp_path / "output")
        setup_logging(log_level="WARNING", output_directory=output_dir)
        logger = get_logger("Test")
        logger.info("info msg")
        logger.warning("warn msg")

        log_file = os.path.join(output_dir, "pipeline.log")
        with open(log_file) as f:
            content = f.read()

        assert "info msg" not in content
        assert "warn msg" in content

    def test_error_level_filters_warning(self, tmp_path):
        """ERROR level only captures ERROR messages."""
        output_dir = str(tmp_path / "output")
        setup_logging(log_level="ERROR", output_directory=output_dir)
        logger = get_logger("Test")
        logger.warning("warn msg")
        logger.error("error msg")

        log_file = os.path.join(output_dir, "pipeline.log")
        with open(log_file) as f:
            content = f.read()

        assert "warn msg" not in content
        assert "error msg" in content


class TestDualOutput:
    """Tests for simultaneous stdout and file output."""

    def test_writes_to_stdout(self, tmp_path, capsys):
        """Log messages appear on stdout."""
        output_dir = str(tmp_path / "output")
        setup_logging(log_level="INFO", output_directory=output_dir)
        logger = get_logger("Test")
        logger.info("stdout test")

        captured = capsys.readouterr()
        assert "stdout test" in captured.out

    def test_writes_to_file(self, tmp_path):
        """Log messages appear in the log file."""
        output_dir = str(tmp_path / "output")
        setup_logging(log_level="INFO", output_directory=output_dir)
        logger = get_logger("Test")
        logger.info("file test")

        log_file = os.path.join(output_dir, "pipeline.log")
        with open(log_file) as f:
            content = f.read()

        assert "file test" in content

    def test_both_outputs_have_same_format(self, tmp_path, capsys):
        """Both stdout and file output use the same log format."""
        output_dir = str(tmp_path / "output")
        setup_logging(log_level="INFO", output_directory=output_dir)
        logger = get_logger("Sync")
        logger.info("format check")

        captured = capsys.readouterr()
        log_file = os.path.join(output_dir, "pipeline.log")
        with open(log_file) as f:
            file_content = f.read().strip()

        # Both should contain the component name and message
        assert "[Sync]" in captured.out
        assert "[Sync]" in file_content
        assert "format check" in captured.out
        assert "format check" in file_content
