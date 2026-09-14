"""因果环境：MarketView 只暴露 t 及之前的数据。

铁律 1 的架构落实。策略永远拿不到未来的 bar——越权访问直接抛异常，
而不是靠"写代码的时候注意点"。
"""
from __future__ import annotations

import numpy as np
import pandas as pd


class LookaheadError(RuntimeError):
    """策略试图读取未来数据。"""


class MarketView:
    """t 时刻的市场视图。索引 i 必须 <= t，否则抛 LookaheadError。"""

    __slots__ = ("_frames", "_t", "_ret_cache")

    def __init__(self, frames: dict[str, pd.DataFrame], t: int):
        self._frames = frames
        self._t = t
        self._ret_cache: dict[tuple[str, int], np.ndarray] = {}

    # ---------------- 元信息 ----------------
    @property
    def step(self) -> int:
        return self._t

    def symbols(self) -> list[str]:
        return list(self._frames)

    def timestamp(self, symbol: str):
        return self._frames[symbol].index[self._t]

    # ---------------- 守卫 ----------------
    def _check(self, i: int) -> int:
        if i < 0 or i > self._t:
            raise LookaheadError(
                f"策略试图访问第 {i} 根 bar，但当前只允许访问 0..{self._t}。"
                "这是未来函数，架构直接拒绝。"
            )
        return i

    # ---------------- 价格 ----------------
    def price(self, symbol: str, field: str = "close", i: int | None = None) -> float:
        idx = self._t if i is None else self._check(i)
        return float(self._frames[symbol][field].iloc[idx])

    def close(self, symbol: str, i: int | None = None) -> float:
        return self.price(symbol, "close", i)

    # ---------------- 历史窗口 ----------------
    def window(self, symbol: str, field: str, n: int) -> np.ndarray:
        lo = max(0, self._t - n + 1)
        return self._frames[symbol][field].iloc[lo:self._t + 1].to_numpy(dtype=float)

    def returns(self, symbol: str, n: int) -> np.ndarray:
        key = (symbol, n)
        if key not in self._ret_cache:
            px = self.window(symbol, "close", n + 1)
            self._ret_cache[key] = np.diff(np.log(px)) if len(px) > 1 else np.array([])
        return self._ret_cache[key]

    def sigma(self, symbol: str, n: int, floor: float = 1e-4) -> float:
        r = self.returns(symbol, n)
        return max(float(r.std()), floor) if len(r) > 1 else floor

    def period_notional(self, symbol: str, n: int) -> float:
        """同周期市场成交名义额中位数 —— 冲击模型的分母。"""
        qv = self.window(symbol, "quote_volume", n)
        return float(np.median(qv)) if len(qv) else 0.0

    def available_history(self) -> int:
        return self._t + 1

    def __repr__(self) -> str:
        return f"<MarketView t={self._t} symbols={self.symbols()}>"
