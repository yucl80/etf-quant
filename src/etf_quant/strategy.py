from __future__ import annotations


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
    """Convert probability to signed position with volatility/trend-aware sizing."""
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
    return max(-max_leverage, min(max_leverage, pos))
