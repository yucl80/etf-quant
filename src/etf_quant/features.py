from __future__ import annotations

from statistics import mean, pstdev

from .data import Bar


FEATURES = [
    "ret_1",
    "ret_3",
    "ret_6",
    "ret_12",
    "rv_12",
    "rv_24",
    "hl_spread",
    "oc_spread",
    "vol_chg",
    "vol_z",
    "trend",
    "atr_14",
    "range_z",
    "ret_skew_12",
    "vpin_proxy",
]


def _pct(a: float, b: float) -> float:
    return 0.0 if b == 0 else a / b - 1.0


def _rolling_std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    return pstdev(values)


def _skew(values: list[float]) -> float:
    if len(values) < 3:
        return 0.0
    m = mean(values)
    s = pstdev(values)
    if s <= 1e-12:
        return 0.0
    third = mean((x - m) ** 3 for x in values)
    return third / (s**3)


def build_dataset(bars: list[Bar], horizon: int = 6) -> tuple[list[dict[str, float]], list[int], list]:
    feats: list[dict[str, float]] = []
    y: list[int] = []
    ts: list = []

    close = [b.close for b in bars]
    vol = [b.volume for b in bars]

    for i in range(30, len(bars) - horizon):
        ret_1 = _pct(close[i], close[i - 1])
        ret_3 = _pct(close[i], close[i - 3])
        ret_6 = _pct(close[i], close[i - 6])
        ret_12 = _pct(close[i], close[i - 12])

        rets12 = [_pct(close[j], close[j - 1]) for j in range(i - 11, i + 1)]
        rets24 = [_pct(close[j], close[j - 1]) for j in range(i - 23, i + 1)]
        rv_12 = _rolling_std(rets12)
        rv_24 = _rolling_std(rets24)

        b = bars[i]
        hl_spread = _pct(b.high, b.low)
        oc_spread = _pct(b.close, b.open)

        vol_chg = _pct(vol[i], vol[i - 1])
        vol_mean = mean(vol[i - 23 : i + 1])
        vol_std = pstdev(vol[i - 23 : i + 1]) or 1.0
        vol_z = (vol[i] - vol_mean) / vol_std

        ema_fast = mean(close[i - 7 : i + 1])
        ema_slow = mean(close[i - 20 : i + 1])
        trend = 0.0 if ema_slow == 0 else (ema_fast - ema_slow) / ema_slow

        tr14 = []
        for j in range(i - 13, i + 1):
            prev_c = close[j - 1]
            tr = max(bars[j].high - bars[j].low, abs(bars[j].high - prev_c), abs(bars[j].low - prev_c))
            tr14.append(tr / (prev_c or 1.0))
        atr_14 = mean(tr14)

        range_hist = [(_pct(bars[j].high, bars[j].low)) for j in range(i - 29, i + 1)]
        range_mean = mean(range_hist)
        range_std = pstdev(range_hist) or 1.0
        range_z = (hl_spread - range_mean) / range_std

        ret_skew_12 = _skew(rets12)

        signed_flow = []
        for j in range(i - 11, i + 1):
            direction = 1.0 if bars[j].close >= bars[j].open else -1.0
            signed_flow.append(direction * vol[j])
        vpin_proxy = abs(sum(signed_flow)) / (sum(vol[i - 11 : i + 1]) + 1e-12)

        feats.append(
            {
                "ret_1": ret_1,
                "ret_3": ret_3,
                "ret_6": ret_6,
                "ret_12": ret_12,
                "rv_12": rv_12,
                "rv_24": rv_24,
                "hl_spread": hl_spread,
                "oc_spread": oc_spread,
                "vol_chg": vol_chg,
                "vol_z": vol_z,
                "trend": trend,
                "atr_14": atr_14,
                "range_z": range_z,
                "ret_skew_12": ret_skew_12,
                "vpin_proxy": vpin_proxy,
            }
        )

        future_ret = _pct(close[i + horizon], close[i])
        y.append(1 if future_ret > 0 else 0)
        ts.append(bars[i].timestamp)

    return feats, y, ts
