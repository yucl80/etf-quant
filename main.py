from __future__ import annotations

import argparse

from src.etf_quant.pipeline import run_pipeline


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ETF 中高频机器学习量化交易程序")
    p.add_argument("--csv", type=str, default=None, help="本地 OHLCV CSV 路径")
    p.add_argument("--bars", type=int, default=3000, help="合成数据 K 线数量")
    p.add_argument("--freq", type=str, default="5min", help="数据频率")
    p.add_argument("--horizon", type=int, default=6, help="预测周期（bar）")
    p.add_argument("--embargo", type=int, default=2, help="walk-forward 训练测试隔离 bar 数")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    metrics = run_pipeline(
        csv_path=args.csv,
        bars=args.bars,
        freq=args.freq,
        horizon=args.horizon,
        embargo=args.embargo,
    )

    print("=== Backtest Metrics ===")
    ordered = [
        "sharpe",
        "max_drawdown",
        "annual_turnover",
        "final_equity",
        "accuracy",
        "brier",
        "win_rate",
        "calmar",
        "n_predictions",
        "avg_cost",
    ]
    for key in ordered:
        value = metrics[key]
        print(f"{key}: {value:.6f}" if isinstance(value, float) else f"{key}: {value}")


if __name__ == "__main__":
    main()
