from __future__ import annotations

from .backtest import run_backtest
from .data import generate_synthetic_ohlcv, load_ohlcv
from .features import build_dataset
from .models import RegimeAwareEnsemble
from .strategy import proba_to_position


def _walk_forward_indices(n: int, train_size: int, test_size: int, embargo: int = 0):
    start = 0
    while start + train_size + embargo + test_size <= n:
        train = list(range(start, start + train_size))
        test_start = start + train_size + embargo
        test = list(range(test_start, test_start + test_size))
        yield train, test
        start += test_size


def _score_train_policy(probs: list[float], xs: list[dict[str, float]], ys: list[int], min_edge: float, base_thr: float) -> float:
    # proxy score: directional accuracy on high-confidence samples
    hit = 0
    total = 0
    for p, x, y in zip(probs, xs, ys):
        pos = proba_to_position(
            p,
            x["rv_24"],
            x["trend"],
            min_edge=min_edge,
            base_threshold=base_thr,
        )
        if abs(pos) < 1e-12:
            continue
        pred = 1 if pos > 0 else 0
        hit += int(pred == y)
        total += 1
    if total == 0:
        return -1.0
    coverage = total / len(probs)
    return (hit / total) * 0.8 + coverage * 0.2


def _tune_policy_on_train(model: RegimeAwareEnsemble, x_train: list[dict[str, float]], y_train: list[int]) -> tuple[float, float]:
    probs = [model.predict_proba(x) for x in x_train]
    best = (-1.0, 0.06, 0.56)
    for min_edge in [0.04, 0.06, 0.08, 0.10]:
        for base_thr in [0.54, 0.56, 0.58, 0.60]:
            s = _score_train_policy(probs, x_train, y_train, min_edge=min_edge, base_thr=base_thr)
            if s > best[0]:
                best = (s, min_edge, base_thr)
    return best[1], best[2]


def run_pipeline(
    csv_path: str | None = None,
    bars: int = 3000,
    freq: str = "5min",
    horizon: int = 6,
    embargo: int = 2,
) -> dict:
    freq_minutes = int(freq.replace("min", ""))
    series = load_ohlcv(csv_path) if csv_path else generate_synthetic_ohlcv(n_bars=bars, freq_minutes=freq_minutes)

    x, y, ts = build_dataset(series, horizon=horizon)
    n = len(x)
    train_size = int(n * 0.55)
    test_size = int(n * 0.15)

    if train_size < 120 or test_size < 30:
        raise ValueError("Not enough data for walk-forward. Increase bars.")

    pred_ts: list = []
    positions: list[float] = []
    probs: list[float] = []
    labels: list[int] = []
    prev_pos = 0.0

    for train_idx, test_idx in _walk_forward_indices(n, train_size, test_size, embargo=embargo):
        model = RegimeAwareEnsemble()
        x_train = [x[i] for i in train_idx]
        y_train = [y[i] for i in train_idx]
        model.fit(x_train, y_train)

        tuned_min_edge, tuned_base_thr = _tune_policy_on_train(model, x_train, y_train)

        for i in test_idx:
            p = model.predict_proba(x[i])
            target_pos = proba_to_position(
                p,
                x[i]["rv_24"],
                x[i]["trend"],
                min_edge=tuned_min_edge,
                base_threshold=tuned_base_thr,
            )
            step_cap = 0.35
            pos = max(prev_pos - step_cap, min(prev_pos + step_cap, target_pos))
            prev_pos = pos
            pred_ts.append(ts[i])
            positions.append(pos)
            probs.append(p)
            labels.append(y[i])

    if not positions:
        raise ValueError("Walk-forward produced no predictions.")

    return run_backtest(series, pred_ts, positions, probs, labels)
