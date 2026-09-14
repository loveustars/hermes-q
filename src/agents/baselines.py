"""基线策略 —— 先把标尺立起来，再谈智能体。

L0 层的意义：任何自学习策略必须先证明自己能打败"什么都不做"和"随机乱做"。
"""
from __future__ import annotations

import numpy as np

from ..env.market_view import MarketView


class Agent:
    """策略接口。

    契约：decide(view) 返回**完整**的目标权重 {symbol: weight}；
    返回 None 表示"维持现状、不做任何交易"。
    """

    name = "agent"

    def decide(self, view: MarketView) -> dict[str, float] | None:  # pragma: no cover
        raise NotImplementedError


class Cash(Agent):
    """全现金 —— 零收益基线。"""

    name = "cash"

    def __init__(self):
        self._done = False

    def decide(self, view: MarketView) -> dict[str, float] | None:
        if self._done:
            return None
        self._done = True
        return {s: 0.0 for s in view.symbols()}


class BuyHold(Agent):
    """等权买入并持有：第一次决策下单，之后一直维持。"""

    name = "buy_hold"

    def __init__(self, weights: dict[str, float] | None = None):
        self.weights = weights
        self._done = False

    def decide(self, view: MarketView) -> dict[str, float] | None:
        if self._done:
            return None
        self._done = True
        syms = view.symbols()
        if self.weights:
            tot = sum(abs(v) for v in self.weights.values()) or 1.0
            return {s: self.weights.get(s, 0.0) / tot for s in syms}
        return {s: 1.0 / len(syms) for s in syms}


class SingleAssetBuyHold(Agent):
    """单标的买入持有 —— 最干净的对照组。"""

    def __init__(self, symbol: str):
        self.symbol = symbol
        self.name = f"bh_{symbol}"
        self._done = False

    def decide(self, view: MarketView) -> dict[str, float] | None:
        if self._done:
            return None
        self._done = True
        return {s: (1.0 if s == self.symbol else 0.0) for s in view.symbols()}


class RandomWeights(Agent):
    """随机权重并按固定频率换仓 —— 换手成本的直接标尺。"""

    def __init__(self, seed: int = 0, rebalance_every: int = 24, long_only: bool = True):
        self.rng = np.random.default_rng(seed)
        self.rebalance_every = max(1, rebalance_every)
        self.long_only = long_only
        self.name = f"random_rb{rebalance_every}_s{seed}"
        self._cur: dict[str, float] = {}
        self._last = -10**9

    def decide(self, view: MarketView) -> dict[str, float] | None:
        if self._cur and view.step - self._last < self.rebalance_every:
            return None                       # 维持现状，不交易
        self._last = view.step
        syms = view.symbols()
        if self.long_only:
            w = self.rng.random(len(syms))
        else:
            w = self.rng.standard_normal(len(syms))
        tot = np.abs(w).sum() or 1.0
        self._cur = {s: float(v / tot) for s, v in zip(syms, w)}
        return self._cur


class Momentum(Agent):
    """最简单的时序动量：过去 n 根收益为正则持有，否则空仓。"""

    def __init__(self, symbol: str, lookback: int = 168, rebalance_every: int = 24):
        self.symbol = symbol
        self.lookback = lookback
        self.rebalance_every = rebalance_every
        self.name = f"mom_{symbol}_{lookback}_rb{rebalance_every}"
        self._cur: dict[str, float] = {}
        self._last = -10**9

    def decide(self, view: MarketView) -> dict[str, float] | None:
        if self._cur and view.step - self._last < self.rebalance_every:
            return None
        self._last = view.step
        if view.available_history() <= self.lookback:
            return None
        r = view.returns(self.symbol, self.lookback)
        on = (len(r) > 0 and r.sum() > 0)
        self._cur = {s: (1.0 if (s == self.symbol and on) else 0.0)
                     for s in view.symbols()}
        return self._cur
