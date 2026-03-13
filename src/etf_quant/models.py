from __future__ import annotations

import math

from .features import FEATURES


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


class OnlineLogistic:
    """带标准化的在线逻辑回归。"""

    def __init__(self, lr: float = 0.03, l2: float = 1e-4):
        self.lr = lr
        self.l2 = l2
        self.w = {f: 0.0 for f in FEATURES}
        self.b = 0.0
        self.mu = {f: 0.0 for f in FEATURES}
        self.var = {f: 1.0 for f in FEATURES}
        self.n = 0

    def _sigmoid(self, z: float) -> float:
        z = _clamp(z, -30.0, 30.0)
        return 1.0 / (1.0 + math.exp(-z))

    def _update_stats(self, x: dict[str, float]) -> None:
        self.n += 1
        for f in FEATURES:
            delta = x[f] - self.mu[f]
            self.mu[f] += delta / self.n
            delta2 = x[f] - self.mu[f]
            self.var[f] += delta * delta2

    def _norm(self, f: str, v: float) -> float:
        if self.n < 2:
            return v
        std = math.sqrt(max(self.var[f] / (self.n - 1), 1e-8))
        return (v - self.mu[f]) / std

    def predict(self, x: dict[str, float]) -> float:
        z = self.b
        for f in FEATURES:
            z += self.w[f] * self._norm(f, x[f])
        return self._sigmoid(z)

    def fit(self, x_list: list[dict[str, float]], y_list: list[int], epochs: int = 3) -> None:
        for _ in range(epochs):
            for x, y in zip(x_list, y_list):
                self._update_stats(x)
                p = self.predict(x)
                g = p - y
                for f in FEATURES:
                    xn = self._norm(f, x[f])
                    self.w[f] -= self.lr * (g * xn + self.l2 * self.w[f])
                self.b -= self.lr * g


class RegimeAwareEnsemble:
    """多专家状态感知集成：趋势/均值回复/稳健模型。"""

    def __init__(self) -> None:
        self.trend_expert = OnlineLogistic(lr=0.05)
        self.mr_expert = OnlineLogistic(lr=0.02)
        self.robust_expert = OnlineLogistic(lr=0.03)
        self.rv_threshold = 0.0
        self.trend_threshold = 0.0

    def fit(self, x_train: list[dict[str, float]], y_train: list[int]) -> None:
        self.trend_expert.fit(x_train, y_train, epochs=2)

        flip_x = []
        flip_y = []
        for x, y in zip(x_train, y_train):
            xx = dict(x)
            xx["ret_1"] *= -1.0
            xx["ret_3"] *= -1.0
            xx["ret_6"] *= -1.0
            flip_x.append(xx)
            flip_y.append(y)
        self.mr_expert.fit(flip_x, flip_y, epochs=4)

        self.robust_expert.fit(x_train, y_train, epochs=5)

        rv = sorted(v["rv_24"] for v in x_train)
        tr = sorted(abs(v["trend"]) for v in x_train)
        self.rv_threshold = rv[int(0.66 * len(rv))] if rv else 0.0
        self.trend_threshold = tr[int(0.60 * len(tr))] if tr else 0.0

    def predict_proba(self, x: dict[str, float]) -> float:
        p_tr = self.trend_expert.predict(x)

        xx = dict(x)
        xx["ret_1"] *= -1.0
        xx["ret_3"] *= -1.0
        xx["ret_6"] *= -1.0
        p_mr = self.mr_expert.predict(xx)

        p_rb = self.robust_expert.predict(x)

        high_vol = x["rv_24"] >= self.rv_threshold
        strong_trend = abs(x["trend"]) >= self.trend_threshold

        if high_vol and strong_trend:
            w = (0.55, 0.10, 0.35)
        elif high_vol:
            w = (0.20, 0.45, 0.35)
        elif strong_trend:
            w = (0.50, 0.15, 0.35)
        else:
            w = (0.30, 0.35, 0.35)

        p = w[0] * p_tr + w[1] * p_mr + w[2] * p_rb
        return _clamp(p, 0.0, 1.0)
