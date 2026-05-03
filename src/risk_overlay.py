from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def compute_volatility_scale(
    forecast_vol: float,
    target_vol: float,
    min_scale: float,
    max_scale: float,
    smoothing: float | None = None,
) -> float:
    if forecast_vol is None or np.isnan(forecast_vol) or forecast_vol <= 0.0:
        return 1.0

    raw = float(target_vol) / float(forecast_vol)
    clipped = float(np.clip(raw, float(min_scale), float(max_scale)))

    if smoothing is None:
        return clipped

    s = float(np.clip(smoothing, 0.0, 1.0))
    # Blend toward 1.0 to avoid unstable regime-switching leverage changes.
    return float((1.0 - s) * clipped + s * 1.0)


def compute_dynamic_crypto_cap(
    forecast_vol: float | None,
    crash_probability: float | None,
    regime_high_stress: bool | None,
    config: dict[str, Any],
) -> float:
    cfg = config.get("overlay", {}).get("dynamic_crypto_cap", config.get("dynamic_crypto_cap", {}))
    normal_cap = float(cfg.get("normal_cap", 0.20))
    medium_cap = float(cfg.get("medium_risk_cap", 0.05))
    high_cap = float(cfg.get("high_risk_cap", 0.00))

    p_cfg = cfg.get("crash_probability_thresholds", {})
    p_medium = float(p_cfg.get("medium", 0.25))
    p_high = float(p_cfg.get("high", 0.40))

    level = "normal"
    if regime_high_stress:
        level = "high"

    if crash_probability is not None and not np.isnan(crash_probability):
        if float(crash_probability) >= p_high:
            level = "high"
        elif float(crash_probability) >= p_medium and level != "high":
            level = "medium"

    # Volatility is optional here because absolute levels are not universally comparable.
    if forecast_vol is not None and not np.isnan(forecast_vol):
        if float(forecast_vol) >= 0.40:
            level = "high"
        elif float(forecast_vol) >= 0.25 and level == "normal":
            level = "medium"

    if level == "high":
        return high_cap
    if level == "medium":
        return medium_cap
    return normal_cap


def apply_crypto_cap(
    weights: pd.Series,
    crypto_assets: list[str],
    crypto_cap: float,
    redistribute_to: str = "pro_rata_non_crypto",
) -> pd.Series:
    w = weights.astype(float).copy()
    crypto = [a for a in crypto_assets if a in w.index]
    if not crypto:
        return w / float(w.sum())

    current_crypto = float(w[crypto].sum())
    cap = float(max(0.0, crypto_cap))

    if current_crypto <= cap + 1e-12:
        return w / float(w.sum())

    excess = current_crypto - cap
    if current_crypto > 0.0:
        w.loc[crypto] = w.loc[crypto] * (cap / current_crypto)
    else:
        w.loc[crypto] = 0.0

    if redistribute_to == "pro_rata_non_crypto":
        non_crypto = [a for a in w.index if a not in crypto]
        non_sum = float(w[non_crypto].sum()) if non_crypto else 0.0
        if non_sum > 0.0:
            w.loc[non_crypto] = w.loc[non_crypto] + excess * (w.loc[non_crypto] / non_sum)

    w = w.clip(lower=0.0)
    return w / float(w.sum())


def apply_risk_scale(
    weights: pd.Series,
    scale: float,
    cash_asset: str | None = None,
) -> pd.Series:
    w = weights.astype(float).copy()
    s = float(scale)

    if cash_asset is None or cash_asset not in w.index:
        # No explicit cash in universe: keep fully invested and avoid synthetic cash assumptions.
        return w / float(w.sum())

    risky_assets = [a for a in w.index if a != cash_asset]
    w.loc[risky_assets] = w.loc[risky_assets] * s
    w.loc[cash_asset] = 1.0 - float(w.loc[risky_assets].sum())
    w = w.clip(lower=0.0)
    return w / float(w.sum())


def make_overlay_decision(
    date: pd.Timestamp,
    base_weights: pd.Series,
    forecasts: dict[str, float | None],
    regime_state: dict[str, Any] | None,
    config: dict[str, Any],
) -> dict[str, Any]:
    overlay_cfg = config.get("overlay", {})
    vol_cfg = overlay_cfg.get("volatility_targeting", {})
    cap_cfg = overlay_cfg.get("dynamic_crypto_cap", {})
    de_cfg = overlay_cfg.get("de_risking", {})

    adjusted = base_weights.astype(float).copy()
    reasons: list[str] = []

    forecast_vol = forecasts.get("forecast_vol")
    crash_probability = forecasts.get("crash_probability")

    regime_high_stress = None
    if regime_state is not None:
        regime_high_stress = bool(regime_state.get("regime_high_stress_dummy", False))

    risk_scale = 1.0
    if bool(vol_cfg.get("enabled", False)):
        risk_scale = compute_volatility_scale(
            forecast_vol=float(forecast_vol) if forecast_vol is not None else np.nan,
            target_vol=float(vol_cfg.get("target_vol", 0.10)),
            min_scale=float(vol_cfg.get("min_scale", 0.50)),
            max_scale=float(vol_cfg.get("max_scale", 1.25)),
            smoothing=vol_cfg.get("smoothing"),
        )
        if not np.isclose(risk_scale, 1.0):
            reasons.append("vol_targeting")

    crypto_cap = float(cap_cfg.get("normal_cap", 0.20))
    if bool(cap_cfg.get("enabled", False)):
        crypto_cap = compute_dynamic_crypto_cap(
            forecast_vol=float(forecast_vol) if forecast_vol is not None else None,
            crash_probability=float(crash_probability) if crash_probability is not None else None,
            regime_high_stress=regime_high_stress,
            config=config,
        )
        before_crypto = float(adjusted[[c for c in adjusted.index if c in {"BTC-USD", "ETH-USD"}]].sum())
        adjusted = apply_crypto_cap(
            weights=adjusted,
            crypto_assets=["BTC-USD", "ETH-USD"],
            crypto_cap=crypto_cap,
        )
        after_crypto = float(adjusted[[c for c in adjusted.index if c in {"BTC-USD", "ETH-USD"}]].sum())
        if after_crypto + 1e-12 < before_crypto:
            reasons.append("dynamic_crypto_cap")

    de_risk_flag = False
    if bool(de_cfg.get("enabled", False)):
        defensive_assets = [a for a in de_cfg.get("defensive_assets", ["TLT", "GLD"]) if a in adjusted.index]
        if defensive_assets and crash_probability is not None and not np.isnan(crash_probability):
            p = float(crash_probability)
            high_thr = float(cap_cfg.get("crash_probability_thresholds", {}).get("high", 0.40))
            medium_thr = float(cap_cfg.get("crash_probability_thresholds", {}).get("medium", 0.25))

            if p >= high_thr:
                shift = 0.10
                de_risk_flag = True
            elif p >= medium_thr:
                shift = 0.05
                de_risk_flag = True
            else:
                shift = 0.0

            if shift > 0.0:
                non_def = [a for a in adjusted.index if a not in defensive_assets]
                removable = adjusted.loc[non_def].clip(lower=0.0)
                removable_sum = float(removable.sum())
                if removable_sum > 0.0:
                    reduction = np.minimum(removable.values, shift * (removable.values / removable_sum))
                    adjusted.loc[non_def] = adjusted.loc[non_def] - reduction
                    add = shift / len(defensive_assets)
                    adjusted.loc[defensive_assets] = adjusted.loc[defensive_assets] + add
                    adjusted = adjusted.clip(lower=0.0)
                    adjusted = adjusted / float(adjusted.sum())
                    reasons.append("de_risking")

    if not np.isclose(risk_scale, 1.0):
        adjusted = apply_risk_scale(adjusted, risk_scale, cash_asset=None)

    base_crypto = float(base_weights[[c for c in base_weights.index if c in {"BTC-USD", "ETH-USD"}]].sum())
    adjusted_crypto = float(adjusted[[c for c in adjusted.index if c in {"BTC-USD", "ETH-USD"}]].sum())

    return {
        "date": pd.Timestamp(date),
        "adjusted_weights": adjusted,
        "risk_scale": float(risk_scale),
        "crypto_cap": float(crypto_cap),
        "de_risk_flag": bool(de_risk_flag),
        "reason": "|".join(reasons) if reasons else "none",
        "model_inputs": {
            "forecast_vol": None if forecast_vol is None else float(forecast_vol),
            "crash_probability": None if crash_probability is None else float(crash_probability),
            "regime_high_stress_dummy": bool(regime_high_stress) if regime_high_stress is not None else None,
        },
        "base_crypto_weight": base_crypto,
        "adjusted_crypto_weight": adjusted_crypto,
    }
