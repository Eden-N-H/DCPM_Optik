"""CLI entry point for the road defect segmentation pipeline.

Allows running the pipeline with:
    python -m src.pipeline --config config/default_config.yaml /path/to/images/

Accepts one or more input paths (files or directories). Directories are
expanded to all supported files (*.jpg, *.jpeg, *.png, *.mp4, *.mov).

Exit codes:
    0 - Success
    1 - Fatal error (config invalid, model load failure, etc.)

Validates: Requirements 8.3, 9.3, 9.4
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

from src.pipeline.orchestrator import PipelineOrchestrator

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".mp4", ".mov"}


def _expand_input_paths(paths: List[str]) -> List[str]:
    """Expand directories to supported files and validate file paths.

    For each path:
    - If it's a directory, collect all files with supported extensions.
    - If it's a file, include it directly.

    Args:
        paths: List of file or directory paths from CLI arguments.

    Returns:
        A flat list of file paths to process.
    """
    expanded: List[str] = []
    for p in paths:
        path = Path(p)
        if path.is_dir():
            for ext in sorted(SUPPORTED_EXTENSIONS):
                expanded.extend(
                    str(f) for f in sorted(path.glob(f"*{ext}"))
                )
                # Also check uppercase extensions
                expanded.extend(
                    str(f) for f in sorted(path.glob(f"*{ext.upper()}"))
                )
        else:
            expanded.append(str(path))
    return expanded


def _parse_args(args: List[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        args: Optional argument list (defaults to sys.argv[1:]).

    Returns:
        Parsed namespace with config and input_paths attributes.
    """
    parser = argparse.ArgumentParser(
        prog="python -m src.pipeline",
        description="Road Defect Segmentation Pipeline - Detect, segment, and measure road surface defects.",
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the YAML configuration file",
    )
    parser.add_argument(
        "input_paths",
        nargs="+",
        help="One or more input file or directory paths to process",
    )
    return parser.parse_args(args)


def main(args: List[str] | None = None) -> int:
    """Main entry point for the pipeline CLI.

    Args:
        args: Optional argument list for testing (defaults to sys.argv[1:]).

    Returns:
        Exit code: 0 for success, 1 for fatal error.
    """
    try:
        parsed = _parse_args(args)

        # Expand directories to individual files
        input_paths = _expand_input_paths(parsed.input_paths)

        if not input_paths:
            print(
                "Error: No supported files found in the provided input paths.",
                file=sys.stderr,
            )
            return 1

        # Create orchestrator and run the pipeline
        orchestrator = PipelineOrchestrator(parsed.config)
        orchestrator.run(input_paths)

        return 0

    except SystemExit as e:
        # Propagate argparse exit (--help) or orchestrator fatal exits
        code = e.code if isinstance(e.code, int) else 1
        return code
    except Exception as e:
        print(f"Fatal error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
