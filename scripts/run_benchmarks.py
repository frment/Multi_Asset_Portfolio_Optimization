"""Script entry point for computing benchmark portfolio metrics.

Loads the processed simple returns, builds three benchmark portfolios,
computes key performance metrics for each, prints a summary table, and
saves the summary to data/processed/benchmark_summary.csv.

Usage:
    python scripts/run_benchmarks.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

# Add project root to sys.path so "src" imports work when run as a script.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.benchmarks import build_all_benchmarks  # noqa: E402
from src.config import load_settings  # noqa: E402
from src.metrics import compute_all_metrics  # noqa: E402
from src.utils import ensure_directory  # noqa: E402


def _load_returns_simple(returns_csv: Path) -> pd.DataFrame:
    """Load the processed simple returns CSV.

    Args:
        returns_csv: Path to returns_simple.csv.

    Returns:
        DataFrame with a DatetimeIndex and one column per ticker.

    Raises:
        FileNotFoundError: If the CSV does not exist.
    """
    if not returns_csv.exists():
        raise FileNotFoundError(
            f"Simple returns file not found: {returns_csv}\n"
            "Run scripts/run_download.py and scripts/run_build_dataset.py first."
        )
    return pd.read_csv(returns_csv, index_col=0, parse_dates=True)


def _format_summary_table(summary: pd.DataFrame) -> str:
    """Format the metrics summary DataFrame as a readable text table.

    Args:
        summary: DataFrame indexed by benchmark name, one metric per column.

    Returns:
        Formatted multi-line string ready to print to terminal.
    """
    # Column display configuration: (header text, decimal places, multiplier, suffix)
    col_config = {
        "ann_return":     ("Ann. Return",  2, 100.0, " %"),
        "ann_volatility": ("Ann. Vol",     2, 100.0, " %"),
        "sharpe":         ("Sharpe",       3, 1.0,   ""),
        "max_drawdown":   ("Max Drawdown", 2, 100.0, " %"),
        "calmar":         ("Calmar",       3, 1.0,   ""),
    }

    # Build header row.
    name_col_width = 22
    metric_col_width = 15
    header = f"{'Benchmark':<{name_col_width}}" + "".join(
        f"{cfg[0]:>{metric_col_width}}" for cfg in col_config.values()
    )
    separator = "-" * len(header)

    rows = [header, separator]
    for benchmark_name, row in summary.iterrows():
        cells = f"{str(benchmark_name):<{name_col_width}}"
        for col, (_, decimals, multiplier, suffix) in col_config.items():
            value = row[col]
            if pd.isna(value):
                cell_str = "n/a"
            else:
                formatted = f"{value * multiplier:.{decimals}f}{suffix}"
                cell_str = formatted
            cells += f"{cell_str:>{metric_col_width}}"
        rows.append(cells)

    return "\n".join(rows)


def main() -> None:
    """Build benchmarks, compute metrics, print table, and save CSV."""
    try:
        settings = load_settings()
        processed_dir = PROJECT_ROOT / settings["paths"]["data_processed"]
        returns_csv = processed_dir / "returns_simple.csv"
        output_csv = processed_dir / "benchmark_summary.csv"

        # --- 1. Load returns --------------------------------------------------
        returns = _load_returns_simple(returns_csv)
        print(f"Loaded returns: {returns.shape[0]} days x {returns.shape[1]} assets")
        print(f"  Date range : {returns.index[0].date()} to {returns.index[-1].date()}")

        # --- 2. Build benchmark return series ---------------------------------
        benchmarks = build_all_benchmarks(returns)
        print(f"\nBenchmarks built: {list(benchmarks.columns)}")

        # --- 3. Compute metrics for each benchmark ----------------------------
        risk_free_rate = float(settings.get("backtest", {}).get("risk_free_rate", 0.0))
        rows = {}
        for name in benchmarks.columns:
            rows[name] = compute_all_metrics(
                benchmarks[name], risk_free_rate=risk_free_rate
            )

        summary = pd.DataFrame(rows).T  # shape: (n_benchmarks, n_metrics)
        summary.index.name = "benchmark"

        # --- 4. Print readable table ------------------------------------------
        print("\n" + "=" * 60)
        print("Benchmark Performance Summary")
        print("=" * 60)
        print(_format_summary_table(summary))
        print("=" * 60)
        print("(All returns and volatilities are annualised. Risk-free rate = "
              f"{risk_free_rate:.1%})")

        # --- 5. Save CSV ------------------------------------------------------
        ensure_directory(output_csv.parent)
        # Round to 6 decimal places to keep the CSV concise.
        summary.round(6).to_csv(output_csv)
        print(f"\nSummary saved to: {output_csv.relative_to(PROJECT_ROOT)}")

    except Exception as error:
        print(f"\nBenchmark run failed. Reason: {error}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
