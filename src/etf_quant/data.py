from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import csv
import math
import random


@dataclass
class Bar:
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


def load_ohlcv(csv_path: str) -> list[Bar]:
    out: list[Bar] = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"timestamp", "open", "high", "low", "close", "volume"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError("CSV must include timestamp,open,high,low,close,volume")
        for row in reader:
            ts = datetime.fromisoformat(row["timestamp"].replace("Z", "+00:00"))
            out.append(
                Bar(
                    timestamp=ts,
                    open=float(row["open"]),
                    high=float(row["high"]),
                    low=float(row["low"]),
                    close=float(row["close"]),
                    volume=float(row["volume"]),
                )
            )
    out.sort(key=lambda x: x.timestamp)
    return out


def generate_synthetic_ohlcv(n_bars: int = 2500, freq_minutes: int = 5, seed: int = 42) -> list[Bar]:
    rng = random.Random(seed)
    ts = datetime(2024, 1, 1, tzinfo=timezone.utc)
    price = 100.0
    bars: list[Bar] = []

    for _ in range(n_bars):
        regime = rng.choices([0, 1, 2], weights=[0.55, 0.30, 0.15])[0]
        drift = [0.00003, 0.0, -0.00002][regime]
        vol = [0.0008, 0.0015, 0.0022][regime]

        ret = drift + rng.gauss(0.0, vol)
        new_close = price * math.exp(ret)
        open_ = price
        high = max(open_, new_close) * (1 + abs(rng.gauss(0.0005, 0.0004)))
        low = min(open_, new_close) * (1 - abs(rng.gauss(0.0005, 0.0004)))
        volume = rng.lognormvariate(11.0, 0.3) * (1.25 if regime == 2 else 1.0)

        bars.append(Bar(ts, open_, high, low, new_close, volume))
        ts += timedelta(minutes=freq_minutes)
        price = new_close

    return bars
