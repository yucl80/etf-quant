# ETF 中高频机器学习量化交易框架（增强版）

这是一个可运行的 ETF 中高频（5min~30min）机器学习量化交易程序，重点强化了**状态切换、自适应仓位、Purged Walk-forward** 与**预测质量评估**。

## 核心升级

- **多专家状态感知模型**：趋势专家 + 均值回复专家 + 稳健专家（在线学习）。
- **多策略组合执行**：ML 主策略 + 趋势跟踪 + 均值回复三策略动态加权。
- **特征增强**：新增 ATR、波动区间 z-score、收益偏度、VPIN 代理因子。
- **Purged Walk-forward**：支持 `embargo` 隔离窗口，降低标签泄漏风险。
- **风险约束增强**：概率阈值 + 目标波动率缩放 + 趋势强度降杠杆。
- **评估更完整**：除 Sharpe/MDD 外，新增 Accuracy、Brier、Win Rate、Calmar、Sortino。

## 目录结构

- `src/etf_quant/data.py`：数据加载与合成数据
- `src/etf_quant/features.py`：中高频特征工程
- `src/etf_quant/models.py`：在线多专家状态感知集成
- `src/etf_quant/strategy.py`：单策略信号映射 + 多策略组合仓位控制
- `src/etf_quant/backtest.py`：交易成本回测与预测质量评估
- `src/etf_quant/pipeline.py`：Purged walk-forward 主流程
- `main.py`：CLI 入口

## 快速开始

```bash
python main.py --bars 3000 --freq 5min --horizon 6 --embargo 2 --trading-minutes-per-day 240
```

如有真实数据：

```bash
python main.py --csv data/510300_5min.csv --freq 5min --horizon 6 --embargo 2 --trading-minutes-per-day 240
```

CSV 至少包含列：`timestamp,open,high,low,close,volume`。

## 实盘化建议

- 替换 VPIN proxy 为逐笔成交驱动的真实订单流不平衡。
- 引入盘口冲击模型（与成交量分段函数绑定）替代固定滑点。
- 接入实时风控（净敞口、行业/指数 beta、单品种风控限额）。


## 本次继续优化

- 训练窗口内参数调优改为**成本与换手惩罚感知**的评分。
- 回测成本模型升级为**波动率 + 交易强度**驱动的动态滑点。
- 输出新增 `avg_cost` 用于评估交易效率。


## 多策略（Multi-Strategy）改进

- 在执行层引入 **ML / Trend / Mean-Reversion** 三策略融合。
- 根据趋势强弱与波动状态动态切换权重，并对策略分歧进行降杠杆抑制。
- 在不改变训练接口的前提下，显著降低回撤与换手冲击。

- 年化指标按 `freq` 与 `trading_minutes_per_day` 自适应计算，避免固定 5min 年化偏差。

- 支持频率格式：`5min` / `15m` / `1h`，并对非法频率输入做参数校验。
- 执行步长会按波动率自适应收缩，高波动时自动降低换仓冲击。
