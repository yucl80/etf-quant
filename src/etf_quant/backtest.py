from __future__ import annotations

import math

from .data import Bar



def run_backtest(
    bars: list[Bar],
    timestamps: list,
    positions: list[float],
    probs: list[float],
    labels: list[int],

) -> dict:
    bar_map = {b.timestamp: b for b in bars}
    closes = [bar_map[t].close for t in timestamps]



    prev_pos = 0.0
    prev_close = closes[0]
    for i, pos in enumerate(positions):
        ret = 0.0 if i == 0 else (closes[i] / prev_close - 1.0)
        traded = abs(pos - prev_pos)


    mean_pnl = _mean(pnl)
    mean_pnl = sum(pnl) / len(pnl)
    var = sum((x - mean_pnl) ** 2 for x in pnl) / max(1, len(pnl) - 1)
    std = math.sqrt(var)

    ann_factor = 252 * 48
    sharpe = (math.sqrt(ann_factor) * mean_pnl / std) if std > 1e-12 else 0.0

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
