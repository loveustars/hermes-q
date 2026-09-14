"""因果环境：MarketView 只暴露 t 及之前的数据。

铁律 1 的架构落实。策略永远拿不到未来的 bar——越权访问直接抛异常，
而不是靠"写代码的时候注意点"。
"""
from __future__ import annotations

import numpy as np
import pandas as pd


class LookaheadError(RuntimeError):
    """策略试图读取未来数据。"""


class Precomputed:
    """滚动统计预计算 —— sigma 与成交额中位数。

    为什么需要：MarketView 每根 bar 都新建，缓存不跨 bar 生效，
    于是每根 bar 都要做一次 pandas 切片 + 统计。7.7 万根 × 3 标的 × 上百次仿真
    会退化成几十分钟的纯 pandas 开销。

    等价性：只要 warmup >= n-1，MarketView 的 lo = max(0, t-n+1) 恒等于 t-n+1，
    所以 rolling(n) 与逐根现算完全一致。注意标准差用 ddof=0，与 numpy .std() 默认一致。
    """

    def __init__(self, frames: dict[str, pd.DataFrame],
                 sigma_windows: set[int], vol_windows: set[int],
                 with_gaps: bool = True):
        self.sigma: dict[tuple[str, int], np.ndarray] = {}
        self.vmed: dict[tuple[str, int], np.ndarray] = {}
        for s, f in frames.items():
            logret = np.log(f["close"]).diff()
            qv = f["quote_volume"]
            for n in sigma_windows:
                self.sigma[(s, n)] = logret.rolling(n).std(ddof=0).to_numpy()
            for n in vol_windows:
                self.vmed[(s, n)] = qv.rolling(n).median().to_numpy()
        # 缺口标记：任一标的在该 bar 之前有缺口即为 True（成交是全组合同时发生的）
        self.gaps: np.ndarray | None = None
        if with_gaps:
            from ..data.quality import union_gap_flags
            self.gaps = union_gap_flags(frames)


class MarketView:
    """t 时刻的市场视图。索引 i 必须 <= t，否则抛 LookaheadError。"""

    __slots__ = ("_frames", "_t", "_ret_cache", "_pre")

    def __init__(self, frames: dict[str, pd.DataFrame], t: int,
                 pre: "Precomputed | None" = None):
        self._frames = frames
        self._t = t
        self._pre = pre
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
        # 预计算只在窗口完整时等价：t < n-1 时现算会截断窗口，
        # 而 rolling(n) 返回 NaN。此时回退到现算，保证无条件一致。
        if self._pre is not None and self._t >= n - 1:
            arr = self._pre.sigma.get((symbol, n))
            if arr is not None:
                v = float(arr[self._t])
                if np.isfinite(v):
                    return max(v, floor)
        r = self.returns(symbol, n)
        return max(float(r.std()), floor) if len(r) > 1 else floor

    def period_notional(self, symbol: str, n: int) -> float:
        """同周期市场成交名义额中位数 —— 冲击模型的分母。"""
        if self._pre is not None and self._t >= n - 1:
            arr = self._pre.vmed.get((symbol, n))
            if arr is not None:
                v = float(arr[self._t])
                if np.isfinite(v):
                    return v
        qv = self.window(symbol, "quote_volume", n)
        return float(np.median(qv)) if len(qv) else 0.0

    def available_history(self) -> int:
        return self._t + 1

    def is_gap(self, i: int | None = None) -> bool:
        """该 bar 之前是否存在数据缺口（交易所停机）。

        用途：跨缺口的成交，实际间隔不是 1 根 bar（实测最长 34 小时），
        所以"延迟 1 根 bar 成交"的假设在那些 bar 上不成立。
        策略与统计口径都可以据此排除或区别处理。
        """
        idx = self._t if i is None else self._check(i)
        if self._pre is None or self._pre.gaps is None or idx >= len(self._pre.gaps):
            return False
        return bool(self._pre.gaps[idx])

    def __repr__(self) -> str:
        return f"<MarketView t={self._t} symbols={self.symbols()}>"
