"""仿真交易所 —— 单次遍历，双账本（毛收益 / 净收益）。

关键设计：**同一次遍历同时维护两个账本**，两者的交易决策完全相同，
差别只在于是否扣除成本。这样"成本吃掉了多少收益"是严格同口径的对比，
而不是跑两遍拿可能不一致的结果来比。

执行约定（防未来函数）：
  - 策略在第 t 根 bar 收盘后做决策，只能看到 0..t
  - 成交发生在第 t+latency 根 bar 的**开盘价**
  - 冲击模型用 t 时刻可见的 σ 与成交额，不用未来值
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..env.market_view import MarketView, Precomputed
from .costs import CostModel


@dataclass
class SimConfig:
    initial_cash: float = 10_000.0
    warmup: int = 300
    latency_bars: int = 1
    allow_short: bool = False
    max_gross: float = 1.0
    sigma_window: int = 168          # 一周小时线，用于冲击模型
    vol_window: int = 168            # 冲击模型里的成交量窗口
    min_trade_notional: float = 1.0  # 小于此金额不下单（避免无穷小换手）


@dataclass
class SimResult:
    index: pd.DatetimeIndex
    net_equity: np.ndarray
    gross_equity: np.ndarray
    cost_paid: np.ndarray
    turnover_notional: np.ndarray
    n_trades: int
    initial_cash: float = 10_000.0
    insolvent_at: int | None = None
    symbols: list[str] = field(default_factory=list)

    @property
    def net_ret(self) -> np.ndarray:
        return np.diff(self.net_equity) / self.net_equity[:-1]

    @property
    def gross_ret(self) -> np.ndarray:
        return np.diff(self.gross_equity) / self.gross_equity[:-1]

    @property
    def insolvent(self) -> bool:
        return self.insolvent_at is not None

    def final_net(self) -> float:
        return float(self.net_equity[-1])

    def final_gross(self) -> float:
        return float(self.gross_equity[-1])

    def cost_drag_bp(self) -> float:
        """全周期成本占**初始本金**的比例（基点）—— 不能用 e[0] 做基数。"""
        return float(self.cost_paid.sum() / self.initial_cash * 1e4)


class SimExchange:
    def __init__(self, frames: dict[str, pd.DataFrame], gross_costs: CostModel,
                 net_costs: CostModel, cfg: SimConfig):
        self.frames = frames
        self.symbols = list(frames)
        self.gross_costs = gross_costs     # enabled=False
        self.net_costs = net_costs         # enabled=True
        self.cfg = cfg
        lens = {len(f) for f in frames.values()}
        if len(lens) != 1:
            raise ValueError(f"各标的 bar 数不一致: {lens}，请先对齐")
        self.T = lens.pop()
        idx = frames[self.symbols[0]].index
        for s in self.symbols[1:]:
            if not frames[s].index.equals(idx):
                raise ValueError(f"{s} 的时间轴与其他标的不一致，请先对齐")
        # 滚动统计预计算：等价但快一到两个数量级（详见 Precomputed 的说明）
        self.pre = Precomputed(frames, {cfg.sigma_window}, {cfg.vol_window})

    def _sanitize(self, target: dict, view: MarketView) -> dict[str, float]:
        out = {}
        for s in self.symbols:
            w = float(target.get(s, 0.0))
            if not np.isfinite(w):
                w = 0.0
            if not self.cfg.allow_short:
                w = max(w, 0.0)
            out[s] = w
        gross = sum(abs(v) for v in out.values())
        if gross > self.cfg.max_gross and gross > 0:
            k = self.cfg.max_gross / gross
            out = {s: v * k for s, v in out.items()}
        return out

    def run(self, agent) -> SimResult:
        cfg = self.cfg
        start = cfg.warmup
        if start >= self.T - cfg.latency_bars - 1:
            raise ValueError("warmup 过长，剩余 bar 不足")

        # 两个账本各自的现金与持仓
        cash_n = cash_g = cfg.initial_cash
        units_n = {s: 0.0 for s in self.symbols}
        units_g = {s: 0.0 for s in self.symbols}

        idx_out, net_eq, gross_eq = [], [], []
        cost_paid, turnover = [], []
        n_trades = 0
        insolvent_at: int | None = None

        for t in range(start, self.T - cfg.latency_bars):
            view = MarketView(self.frames, t, self.pre)
            target = agent.decide(view)
            ex = t + cfg.latency_bars

            # 契约：decide 返回完整目标权重 dict；返回 None 表示"维持现状"
            # 没有这个语义，买入持有会被翻译成每根 bar 精确再平衡，换手虚高。
            if target is None:
                px_close = {s: float(self.frames[s]["close"].iloc[ex])
                            for s in self.symbols}
                idx_out.append(self.frames[self.symbols[0]].index[ex])
                net_eq.append(cash_n + sum(units_n[s] * px_close[s] for s in self.symbols))
                gross_eq.append(cash_g + sum(units_g[s] * px_close[s] for s in self.symbols))
                cost_paid.append(0.0)
                turnover.append(0.0)
                if insolvent_at is None and net_eq[-1] < 0.01 * cfg.initial_cash:
                    insolvent_at = len(net_eq) - 1
                continue

            target = self._sanitize(target, view)
            px = {s: float(self.frames[s]["open"].iloc[ex]) for s in self.symbols}

            eq_n = cash_n + sum(units_n[s] * px[s] for s in self.symbols)
            eq_g = cash_g + sum(units_g[s] * px[s] for s in self.symbols)

            bar_cost = 0.0
            bar_turnover = 0.0

            for s in self.symbols:
                sigma = view.sigma(s, cfg.sigma_window)
                period_notional = view.period_notional(s, cfg.vol_window)

                # --- 净账本：真实成交，计成本 ---
                want_n = target[s] * eq_n / px[s]
                d_n = want_n - units_n[s]
                notional_n = abs(d_n) * px[s]
                if notional_n >= cfg.min_trade_notional:
                    c = self.net_costs.trade_cost(s, notional_n, sigma, period_notional)
                    cash_n -= c
                    cash_n -= d_n * px[s]
                    units_n[s] = want_n
                    bar_cost += c
                    n_trades += 1

                # --- 毛账本：同一决策、同一目标权重，零成本 ---
                # 换手按毛账本统计：净账本一旦接近归零就停止交易，
                # 会让换手/成本统计被截断，从而低估策略真实的交易强度。
                want_g = target[s] * eq_g / px[s]
                d_g = want_g - units_g[s]
                notional_g = abs(d_g) * px[s]
                if notional_g >= cfg.min_trade_notional:
                    cash_g -= d_g * px[s]
                    units_g[s] = want_g
                    bar_turnover += notional_g

            px_close = {s: float(self.frames[s]["close"].iloc[ex]) for s in self.symbols}
            idx_out.append(self.frames[self.symbols[0]].index[ex])
            net_eq.append(cash_n + sum(units_n[s] * px_close[s] for s in self.symbols))
            gross_eq.append(cash_g + sum(units_g[s] * px_close[s] for s in self.symbols))
            cost_paid.append(bar_cost)
            turnover.append(bar_turnover)

        return SimResult(
            index=pd.DatetimeIndex(idx_out),
            net_equity=np.asarray(net_eq, dtype=float),
            gross_equity=np.asarray(gross_eq, dtype=float),
            cost_paid=np.asarray(cost_paid, dtype=float),
            turnover_notional=np.asarray(turnover, dtype=float),
            n_trades=n_trades,
            initial_cash=cfg.initial_cash,
            insolvent_at=insolvent_at,
            symbols=self.symbols,
        )
