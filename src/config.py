"""
Configuration loader for the project.

Reads YAML config files from the config/ directory and returns them
as plain Python dictionaries. Keeps things simple — no custom config
classes or validation frameworks for now.
"""

import json
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


def load_dataset_metadata(path: str | Path | None = None) -> dict[str, Any]:
    """Load dataset metadata JSON if it exists.

    Args:
        path: Optional custom metadata path. Defaults to
            data/processed/dataset_metadata.json.

    Returns:
        Parsed metadata dictionary, or an empty dict if file is missing.
    """
    if path is None:
        settings = load_settings()
        data_processed = settings.get("paths", {}).get("data_processed", "data/processed")
        metadata_path = PROJECT_ROOT / data_processed / "dataset_metadata.json"
    else:
        metadata_path = Path(path)

    if not metadata_path.exists():
        return {}

    with metadata_path.open("r", encoding="utf-8") as file:
        raw = json.load(file)
    return raw if isinstance(raw, dict) else {}


def get_calendar_settings(settings: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return normalized calendar settings from config/settings.yaml."""
    cfg = load_settings() if settings is None else settings
    data_cfg = cfg.get("data", {})
    calendar_cfg = data_cfg.get("calendar", {})
    return {
        "policy": str(calendar_cfg.get("policy", "business_day_aligned")),
        "tradfi_assets": [str(x) for x in calendar_cfg.get("tradfi_assets", ["SPY", "QQQ", "GLD", "TLT"])],
        "crypto_assets": [str(x) for x in calendar_cfg.get("crypto_assets", ["BTC-USD", "ETH-USD"])],
        "require_tradfi_observation": bool(calendar_cfg.get("require_tradfi_observation", True)),
        "allow_weekend_rebalances": bool(calendar_cfg.get("allow_weekend_rebalances", False)),
        "annualization_factor": float(calendar_cfg.get("annualization_factor", 252.0)),
        "calendar_day_annualization_factor": float(
            calendar_cfg.get("calendar_day_annualization_factor", 365.25)
        ),
    }


def resolve_annualization_factor(
    settings: dict[str, Any] | None = None,
    dataset_metadata: dict[str, Any] | None = None,
) -> float:
    """Resolve annualization factor from metadata first, then settings fallback."""
    if dataset_metadata and dataset_metadata.get("annualization_factor") is not None:
        return float(dataset_metadata["annualization_factor"])

    cfg = load_settings() if settings is None else settings
    backtest_cfg = cfg.get("backtest", {})
    if backtest_cfg.get("annualization_factor") is not None:
        return float(backtest_cfg["annualization_factor"])

    calendar_cfg = get_calendar_settings(cfg)
    return float(calendar_cfg.get("annualization_factor", 252.0))
