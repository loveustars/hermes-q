"""数据质量层：缺口识别与标记。

背景（实测）：
  BTC / ETH 各 28 处缺口、累计缺 128 根（0.161%），最长一处 34 小时（2018-02-09）；
  BNB 27 处、缺 122 根。**三个标的的缺口位置完全一致**（BTC 与 ETH 连时间点都相同），
  说明是交易所级别的停机/维护，而不是单标的问题。

为什么必须处理：
  1. 跨缺口的 next-bar 成交，实际间隔不是 1 小时（最长 34 小时），
     "延迟 1 根 bar" 这个假设在那些 bar 上不成立。
  2. 缺口 bar 的收益率是跨 34 小时的变化，会污染以"根"为单位的年化与波动率统计。
  3. 冲击模型的 σ 与成交额中位数按"根"取窗口，跨缺口时窗口实际跨度变长。

策略：标记而非删除。删除会改变时间轴的连续性，反而引入更多问题。
标记后由上层决定怎么用（默认：成交照旧，因为交易所当时确实闭市、下一个可成交价就是复牌开盘价；
但统计口径与敏感性分析要能排除它们）。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

HOUR = pd.Timedelta(hours=1)


def gap_flags(index: pd.DatetimeIndex, step: pd.Timedelta = HOUR) -> np.ndarray:
    """返回布尔数组：第 i 个位置为 True 表示该 bar 之前存在缺口。

    第 0 个位置恒为 False（没有前一根可比）。
    """
    if len(index) < 2:
        return np.zeros(len(index), dtype=bool)
    d = index.to_series().diff()
    return (d > step).to_numpy()


def gap_report(index: pd.DatetimeIndex, step: pd.Timedelta = HOUR) -> dict:
    d = index.to_series().diff().dropna()
    gaps = d[d > step]
    missing = int(((gaps - step) / step).sum()) if len(gaps) else 0
    return {
        "bars": int(len(index)),
        "n_gaps": int(len(gaps)),
        "missing_bars": missing,
        "missing_pct": round(missing / len(index) * 100, 4) if len(index) else 0.0,
        "longest": str(gaps.max()) if len(gaps) else "0",
        "first_gap": str(gaps.index[0]) if len(gaps) else None,
    }


def union_gap_flags(frames: dict[str, pd.DataFrame],
                    step: pd.Timedelta = HOUR) -> np.ndarray:
    """多标的的缺口并集。

    只要任一标的在该 bar 之前有缺口，就标记 —— 因为成交是对全组合同时发生的。
    各标的必须已对齐到同一时间轴。
    """
    idx = None
    for f in frames.values():
        idx = f.index if idx is None else idx.intersection(f.index)
    flags = np.zeros(len(idx), dtype=bool)
    for f in frames.values():
        sub = f.loc[idx]
        flags |= gap_flags(sub.index, step)
    return flags


def trading_time_weights(index: pd.DatetimeIndex,
                         step: pd.Timedelta = HOUR) -> np.ndarray:
    """每个 bar 实际代表的时间长度（以 step 为单位）。

    正常 bar = 1.0；跨缺口的 bar = 实际间隔 / step（可能远大于 1）。
    用于把"按根统计"的量折算成"按时间统计"，避免缺口把年化算歪。
    """
    if len(index) == 0:
        return np.array([])
    d = index.to_series().diff()
    w = (d / step).to_numpy()
    w[0] = 1.0
    w = np.where(np.isfinite(w) & (w > 0), w, 1.0)
    return w
