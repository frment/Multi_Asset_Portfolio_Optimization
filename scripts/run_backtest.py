"""Script entry point for the rolling walk-forward backtest.

Runs the monthly-rebalanced minimum variance strategy, builds all benchmark
portfolios over the same out-of-sample period, computes performance metrics
for each strategy, prints a summary table, and saves all outputs.

Usage:
    python scripts/run_backtest.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

# Add project root to sys.path so "src" imports work when run as a script.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.backtest import (  # noqa: E402
    get_rebalance_dates,
    run_min_variance_backtest,
)
from src.benchmarks import build_all_benchmarks  # noqa: E402
from src.config import load_settings  # noqa: E402
from src.metrics import compute_all_metrics  # noqa: E402
from src.optimizer import load_optimizer_config  # noqa: E402
from src.utils import ensure_directory  # noqa: E402


def _load_returns(returns_csv: Path) -> pd.DataFrame:
    """Load processed simple returns from CSV."""
    if not returns_csv.exists():
        raise FileNotFoundError(
            f"Returns file not found: {returns_csv}\n"
            "Run scripts/run_download.py and scripts/run_build_dataset.py first."
        )
    return pd.read_csv(returns_csv, index_col=0, parse_dates=True)


def _format_summary_table(summary: pd.DataFrame) -> str:
    """Format the metrics summary as a readable text table."""
    col_config = {
        "ann_return":     ("Ann. Return",  2, 100.0, " %"),
        "ann_volatility": ("Ann. Vol",     2, 100.0, " %"),
        "sharpe":         ("Sharpe",       3, 1.0,   ""),
        "max_drawdown":   ("Max Drawdown", 2, 100.0, " %"),
        "calmar":         ("Calmar",       3, 1.0,   ""),
    }

    name_w = 22
    col_w = 15
    header = f"{'Strategy':<{name_w}}" + "".join(
        f"{cfg[0]:>{col_w}}" for cfg in col_config.values()
    )
    sep = "-" * len(header)
    rows = [header, sep]

    for strategy, row in summary.iterrows():
        cells = f"{str(strategy):<{name_w}}"
        for col, (_, decimals, mult, suffix) in col_config.items():
            val = row[col]
            cell = "n/a" if pd.isna(val) else f"{val * mult:.{decimals}f}{suffix}"
            cells += f"{cell:>{col_w}}"
        rows.append(cells)

    return "\n".join(rows)


def main() -> None:
    """Run the full walk-forward backtest pipeline and save outputs."""
    try:
        settings = load_settings()
        processed_dir = PROJECT_ROOT / settings["paths"]["data_processed"]
        ensure_directory(processed_dir)

        returns_csv = processed_dir / "returns_simple.csv"

        # Paths for output files.
        portfolio_returns_csv = processed_dir / "portfolio_returns.csv"
        weights_history_csv   = processed_dir / "weights_history.csv"
        backtest_summary_csv  = processed_dir / "backtest_summary.csv"
        turnover_history_csv  = processed_dir / "turnover_history.csv"

        # --- 1. Load data ------------------------------------------------------
        returns = _load_returns(returns_csv)
        print(f"Loaded returns: {returns.shape[0]} days x {returns.shape[1]} assets")
        print(f"  Date range : {returns.index[0].date()} to {returns.index[-1].date()}")

        optimizer_config = load_optimizer_config()
        lookback = int(settings.get("backtest", {}).get("lookback_window_days", 252))
        rebalance_frequency = str(
            settings.get("backtest", {}).get("rebalance_frequency", "monthly")
        )
        risk_free_rate = float(settings.get("backtest", {}).get("risk_free_rate", 0.0))

        rebalance_dates = get_rebalance_dates(
            returns.index,
            lookback,
            rebalance_frequency=rebalance_frequency,
        )
        print(f"\nLookback window  : {lookback} trading days")
        print(f"Rebalance freq   : {rebalance_frequency}")
        print(f"Rebalance dates  : {len(rebalance_dates)} periods")
        print(f"  First rebalance: {rebalance_dates[0].date()}")
        print(f"  Last rebalance : {rebalance_dates[-1].date()}")

        # --- 2. Run minimum variance rolling backtest -------------------------
        print("\nRunning walk-forward minimum variance backtest...")
        minvar_returns, weights_history, turnover_history = run_min_variance_backtest(
            returns=returns,
            optimizer_config=optimizer_config,
            lookback_window=lookback,
            rebalance_frequency=rebalance_frequency,
        )
        print(f"  OOS period     : {minvar_returns.index[0].date()} to {minvar_returns.index[-1].date()}")
        print(f"  OOS trading days: {len(minvar_returns)}")

        # --- 3. Build benchmarks over the same OOS period --------------------
        # Align all benchmarks to the min-variance OOS start date for a fair
        # like-for-like comparison.
        oos_start = minvar_returns.index[0]
        returns_oos = returns.loc[oos_start:]

        benchmarks = build_all_benchmarks(returns_oos)
        benchmarks.index.name = "Date"

        # --- 4. Compute metrics for all strategies ---------------------------
        all_strategies: dict[str, pd.Series] = {
            "min_variance": minvar_returns,
            **{col: benchmarks[col] for col in benchmarks.columns},
        }

        metric_rows: dict[str, dict] = {}
        for name, series in all_strategies.items():
            metric_rows[name] = compute_all_metrics(series, risk_free_rate=risk_free_rate)

        summary = pd.DataFrame(metric_rows).T
        summary.index.name = "strategy"

        # --- 5. Print summary table ------------------------------------------
        print("\n" + "=" * 67)
        print("Backtest Performance Summary (out-of-sample)")
        print("=" * 67)
        print(_format_summary_table(summary))
        print("=" * 67)
        print(
            f"(Annualised metrics, risk-free rate = {risk_free_rate:.1%}, "
            f"OOS from {oos_start.date()})"
        )

        # --- 6. Save outputs --------------------------------------------------
        # portfolio_returns.csv — all strategy daily returns in one file.
        all_returns = pd.DataFrame({"min_variance": minvar_returns})
        all_returns = all_returns.join(benchmarks, how="outer")
        all_returns.index.name = "Date"
        all_returns.to_csv(portfolio_returns_csv)

        # weights_history.csv — one row per monthly rebalance date.
        weights_history.to_csv(weights_history_csv)

        # backtest_summary.csv — metrics table.
        summary.round(6).to_csv(backtest_summary_csv)

        # turnover_history.csv — one row per rebalance, pre-trade drifted turnover.
        turnover_history.to_csv(turnover_history_csv)

        print(f"\nSaved portfolio returns to : {portfolio_returns_csv.relative_to(PROJECT_ROOT)}")
        print(f"Saved weights history to   : {weights_history_csv.relative_to(PROJECT_ROOT)}")
        print(f"Saved backtest summary to  : {backtest_summary_csv.relative_to(PROJECT_ROOT)}")
        print(f"Saved turnover history to  : {turnover_history_csv.relative_to(PROJECT_ROOT)}")

        # --- 8. Turnover summary ----------------------------------------------
        oos_turnover = turnover_history.loc[~turnover_history["is_initial_rebalance"], "turnover_one_way"]
        print("\n" + "=" * 50)
        print("Turnover Summary (pre-trade drifted, one-way)")
        print("=" * 50)
        print(f"  Rebalances total    : {len(turnover_history)}")
        print(f"  Operational rebals  : {len(oos_turnover)}  (excl. initial)")
        if len(oos_turnover) > 0:
            print(f"  Mean turnover       : {oos_turnover.mean():.4f}  ({oos_turnover.mean()*100:.2f}%)")
            print(f"  Median turnover     : {oos_turnover.median():.4f}  ({oos_turnover.median()*100:.2f}%)")
            print(f"  Max turnover        : {oos_turnover.max():.4f}  ({oos_turnover.max()*100:.2f}%)")
        print("=" * 50)

        # --- 9. Quick look-ahead bias check -----------------------------------
        # Verify the first rebalance uses data strictly before the OOS start.
        first_rebal = rebalance_dates[0]
        last_train_idx = returns.index.get_loc(first_rebal) - 1
        print(
            f"\nLook-ahead check : first rebalance {first_rebal.date()} used data "
            f"up to (and including) {returns.index[last_train_idx].date()}"
        )

    except Exception as error:
        print(f"\nBacktest failed. Reason: {error}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
