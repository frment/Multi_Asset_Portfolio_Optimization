"""
General-purpose utility helpers.

This module contains small, reusable functions that don't belong to any
specific domain (data, optimization, backtesting, etc.). Keep this lean —
add business logic to dedicated modules instead.
"""

from pathlib import Path


def ensure_directory(path: str | Path) -> Path:
    """Create a directory (and parents) if it doesn't already exist.

    Args:
        path: Directory path to create.

    Returns:
        The resolved Path object.
    """
    dir_path = Path(path)
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path.resolve()


def get_project_root() -> Path:
    """Return the absolute path to the project root directory.

    The project root is defined as two levels up from this file:
    src/utils.py -> src/ -> project root.

    Returns:
        Path to the project root.
    """
    return Path(__file__).resolve().parent.parent
