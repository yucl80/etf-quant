from __future__ import annotations

import math

from .data import Bar


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def run_backtest(
    bars: list[Bar],
    timestamps: list,
    positions: list[float],
    probs: list[float],
    labels: list[int],
    rv_24_seq: list[float] | None = None,
    fee_bps: float = 1.2,
    slippage_bps: float = 0.8,
    impact_k: float = 250.0,
    bars_per_day: int = 48,
) -> dict:
    bar_map = {b.timestamp: b for b in bars}
    closes = [bar_map[t].close for t in timestamps]

    if rv_24_seq is None:
        rv_24_seq = [0.001 for _ in positions]
    if len(rv_24_seq) != len(positions):
        raise ValueError("rv_24_seq length must equal positions length")

    pnl: list[float] = []
    equity = 1.0
    curve: list[float] = []
    cost_series: list[float] = []

    prev_pos = 0.0
    prev_close = closes[0]
    for i, pos in enumerate(positions):
        ret = 0.0 if i == 0 else (closes[i] / prev_close - 1.0)
        traded = abs(pos - prev_pos)

        # Dynamic cost model: base fee + volatility-dependent slippage + impact by trade size.
        dyn_slippage_bps = slippage_bps * (1.0 + min(2.0, rv_24_seq[i] * impact_k))
        cost = traded * (fee_bps + dyn_slippage_bps) / 10000

        p = prev_pos * ret - cost
        pnl.append(p)
        cost_series.append(cost)
        equity *= 1.0 + p
        curve.append(equity)
        prev_pos = pos
        prev_close = closes[i]

    mean_pnl = _mean(pnl)
    var = sum((x - mean_pnl) ** 2 for x in pnl) / max(1, len(pnl) - 1)
    std = math.sqrt(var)

    downside = [min(0.0, x) for x in pnl]
    downside_var = sum(x * x for x in downside) / max(1, len(downside) - 1)
    downside_std = math.sqrt(downside_var)

    ann_factor = 252 * max(1, bars_per_day)
    sharpe = (math.sqrt(ann_factor) * mean_pnl / std) if std > 1e-12 else 0.0
    sortino = (math.sqrt(ann_factor) * mean_pnl / downside_std) if downside_std > 1e-12 else 0.0

    peak = curve[0]
    mdd = 0.0
    for v in curve:
        peak = max(peak, v)
        mdd = min(mdd, v / peak - 1.0)

    turnover = sum(abs(positions[i] - (positions[i - 1] if i > 0 else 0.0)) for i in range(len(positions))) / len(positions)
    turnover *= ann_factor

    pred = [1 if p >= 0.5 else 0 for p in probs]
    acc = sum(int(a == b) for a, b in zip(pred, labels)) / len(labels)
    brier = sum((p - y) ** 2 for p, y in zip(probs, labels)) / len(labels)
    win_rate = sum(int(x > 0) for x in pnl) / len(pnl)
    calmar = ((equity - 1.0) / max(abs(mdd), 1e-8))

    return {
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": mdd,
        "annual_turnover": turnover,
        "final_equity": curve[-1],
        "n_predictions": len(positions),
        "accuracy": acc,
        "brier": brier,
        "win_rate": win_rate,
        "calmar": calmar,
        "avg_cost": _mean(cost_series),
    }
