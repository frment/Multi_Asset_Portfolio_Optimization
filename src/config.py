"""
Configuration loader for the project.

Reads YAML config files from the config/ directory and returns them
as plain Python dictionaries. Keeps things simple — no custom config
classes or validation frameworks for now.
"""

from pathlib import Path
from typing import Any

import yaml


# Project root is two levels up from this file: src/config.py -> src/ -> project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = PROJECT_ROOT / "config"


def load_yaml(filename: str) -> dict[str, Any]:
    """Load a YAML file from the config/ directory and return it as a dict.

    Args:
        filename: Name of the YAML file (e.g., "assets.yaml" or "settings.yaml").

    Returns:
        Dictionary with the parsed YAML contents.

    Raises:
        FileNotFoundError: If the config file does not exist.
    """
    filepath = CONFIG_DIR / filename
    if not filepath.exists():
        raise FileNotFoundError(f"Config file not found: {filepath}")

    with filepath.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)

    # yaml.safe_load returns None for empty files; keep return type stable.
    return data if isinstance(data, dict) else {}


def load_assets() -> dict[str, Any]:
    """Load the asset universe configuration.

    Returns:
        Dictionary with asset universe, tickers, and categories.
    """
    return load_yaml("assets.yaml")


def load_settings() -> dict[str, Any]:
    """Load the portfolio settings configuration.

    Returns:
        Dictionary with portfolio constraints, benchmarks, and parameters.
    """
    return load_yaml("settings.yaml")


def load_robustness() -> dict[str, Any]:
    """Load Chapter 2 robustness configuration.

    Returns:
        Dictionary with first-pass robustness setup.
    """
    return load_yaml("robustness.yaml")


def load_tail_risk() -> dict[str, Any]:
    """Load Chapter 3 tail-risk configuration.

    Returns:
        Dictionary with CVaR setup, comparators, costs, and stress windows.
    """
    return load_yaml("tail_risk.yaml")


def load_regime_analysis() -> dict[str, Any]:
    """Load Chapter 4 regime-analysis configuration.

    Returns:
        Dictionary with feature windows, paths, and NaN-handling strategy.
    """
    return load_yaml("regime_analysis.yaml")
