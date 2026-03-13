from __future__ import annotations


def _clip(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def proba_to_position(
    proba: float,
    rv_24: float,
    trend: float,
    base_threshold: float = 0.56,
    target_vol: float = 0.0015,
    max_leverage: float = 1.2,
    min_edge: float = 0.06,
    vol_k: float = 7.0,
) -> float:
    """Convert ML probability to signed position with volatility/trend-aware sizing."""
    dynamic = min(0.63, base_threshold + rv_24 * vol_k)

    edge = abs(proba - 0.5)
    adaptive_min_edge = min_edge + min(0.05, rv_24 * 10.0)

    if edge < adaptive_min_edge:
        raw = 0.0
    elif proba > dynamic:
        raw = 1.0
    elif proba < 1.0 - dynamic:
        raw = -1.0
    else:
        raw = 0.0

    trend_scale = min(1.0, abs(trend) / 0.0015 + 0.2)
    vol_scale = min(max_leverage, target_vol / max(rv_24, 1e-6))
    pos = raw * trend_scale * vol_scale
    return _clip(pos, -max_leverage, max_leverage)


def trend_follow_position(trend: float, rv_24: float, max_leverage: float = 1.0) -> float:
    """Pure trend-following sleeve driven by trend magnitude and volatility targeting."""
    if abs(trend) < 0.0004:
        return 0.0
    direction = 1.0 if trend > 0 else -1.0
    strength = min(1.0, abs(trend) / 0.0025)
    vol_scale = min(max_leverage, 0.0016 / max(rv_24, 1e-6))
    return _clip(direction * strength * vol_scale, -max_leverage, max_leverage)


def mean_reversion_position(range_z: float, trend: float, rv_24: float, max_leverage: float = 0.8) -> float:
    """Mean-reversion sleeve: fade extreme intrabar range when trend is weak."""
    if abs(trend) > 0.0015:
        return 0.0
    if abs(range_z) < 1.0:
        return 0.0
    direction = -1.0 if range_z > 0 else 1.0
    strength = min(1.0, (abs(range_z) - 1.0) / 2.0)
    vol_scale = min(max_leverage, 0.0012 / max(rv_24, 1e-6))
    return _clip(direction * strength * vol_scale, -max_leverage, max_leverage)


def multi_strategy_position(
    ml_pos: float,
    trend: float,
    range_z: float,
    rv_24: float,
    max_leverage: float = 1.2,
) -> float:
    """Blend ML + trend-following + mean-reversion sleeves with regime-aware weights."""
    tr_pos = trend_follow_position(trend=trend, rv_24=rv_24, max_leverage=1.0)
    mr_pos = mean_reversion_position(range_z=range_z, trend=trend, rv_24=rv_24, max_leverage=0.8)

    high_vol = rv_24 > 0.0018
    strong_trend = abs(trend) > 0.0015

    if strong_trend and not high_vol:
        w_ml, w_tr, w_mr = 0.45, 0.45, 0.10
    elif high_vol:
        w_ml, w_tr, w_mr = 0.50, 0.20, 0.30
    else:
        w_ml, w_tr, w_mr = 0.55, 0.25, 0.20

    combined = w_ml * ml_pos + w_tr * tr_pos + w_mr * mr_pos

    # Disagreement damping: reduce leverage if sleeves strongly disagree.
    disagreement = abs(ml_pos - tr_pos) + abs(ml_pos - mr_pos)
    damp = max(0.6, 1.0 - 0.15 * disagreement)
    return _clip(combined * damp, -max_leverage, max_leverage)


def market_timing_exposure(
    trend: float,
    rv_24: float,
    prev_exposure: float,
    min_exposure: float = 0.25,
    max_exposure: float = 1.0,
) -> float:
    """Independent timing layer to control aggregate portfolio exposure."""
    trend_signal = min(1.0, abs(trend) / 0.0020)
    vol_penalty = min(1.0, rv_24 / 0.0025)

    target = min_exposure + (max_exposure - min_exposure) * (0.75 * trend_signal + 0.25 * (1.0 - vol_penalty))
    target = _clip(target, min_exposure, max_exposure)

    # Smooth timing exposure to avoid frequent full-notional flips.
    alpha = 0.35
    exposure = (1.0 - alpha) * prev_exposure + alpha * target
    return _clip(exposure, min_exposure, max_exposure)
