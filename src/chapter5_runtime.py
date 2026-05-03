"""Lightweight runtime helpers for Chapter 5 pipeline observability.

Provides:
- log_event(event, **kwargs)  – timestamped key=value line to stdout
- timed_phase(name)           – context manager that logs START / END + duration
- parse_debug_args(parser)    – attaches standard debug/limit arguments to an ArgumentParser
- limit_sequence(seq, max_n)  – return first max_n items of a sequence (or all if None)
"""
from __future__ import annotations

import contextlib
from datetime import datetime, timezone
from time import perf_counter
from typing import Any, Generator, Sequence


# ---------------------------------------------------------------------------
# Core log primitive
# ---------------------------------------------------------------------------

def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def log_event(event: str, **kwargs: Any) -> None:
    """Print a structured log line to stdout with flush, e.g.
    [MODEL_START] target=foo model=ridge split=3 n_train=500 …
    """
    parts = " ".join(f"{k}={v}" for k, v in kwargs.items())
    print(f"[{event}] {parts} ts={_utc_now()}", flush=True)


# ---------------------------------------------------------------------------
# Phase timer
# ---------------------------------------------------------------------------

@contextlib.contextmanager
def timed_phase(name: str) -> Generator[None, None, None]:
    """Context manager that logs [START] name and [END] name duration_seconds=X."""
    log_event("START", phase=name)
    t0 = perf_counter()
    try:
        yield
    finally:
        elapsed = perf_counter() - t0
        log_event("END", phase=name, duration_seconds=f"{elapsed:.3f}")


# ---------------------------------------------------------------------------
# Argument helpers
# ---------------------------------------------------------------------------

def parse_debug_args(parser: Any) -> None:
    """Attach standard debug/limit arguments to an ArgumentParser in-place."""
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Run a bounded, fast debug pass (limits targets, models, splits)",
    )
    parser.add_argument(
        "--max-targets",
        type=int,
        default=None,
        metavar="N",
        help="Cap the number of target columns to train",
    )
    parser.add_argument(
        "--max-models",
        type=int,
        default=None,
        metavar="N",
        help="Cap the number of models per target",
    )
    parser.add_argument(
        "--max-splits",
        type=int,
        default=None,
        metavar="N",
        help="Cap the number of walk-forward splits per target/model",
    )
    parser.add_argument(
        "--skip-overlay",
        action="store_true",
        help="Skip the overlay backtest phase",
    )
    parser.add_argument(
        "--skip-models",
        action="store_true",
        help="Skip the supervised models phase",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing outputs (default: debug outputs go to a separate dir)",
    )


# ---------------------------------------------------------------------------
# Sequence limiter
# ---------------------------------------------------------------------------

def limit_sequence(seq: Sequence[Any], max_n: int | None) -> list[Any]:
    """Return the first max_n items of seq, or all items if max_n is None."""
    items = list(seq)
    if max_n is None or max_n <= 0:
        return items
    return items[: int(max_n)]
