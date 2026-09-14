"""诊断：G1 100% 假阳性的来源是不是"再平衡溢价"。

假设：随机多头权重策略按固定频率再平衡，而基准是等权买入持有（不再平衡）。
在加密这种高波动、资产间不完全相关的市场里，恒定混合再平衡本身会产生额外收益
（分散化/波动率收割）。如果这是主因，那么：
  - 把固定的等权组合按不同频率再平衡，也应该对基准产生显著正 alpha
  - 再平衡频率越接近"从不"（长间隔），alpha 越接近 0
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src import config as cfgmod  # noqa: E402
from src.agents.baselines import Agent  # noqa: E402
from src.data import store  # noqa: E402
from src.eval import protocol  # noqa: E402
from src.sim.costs import CostModel  # noqa: E402
from src.sim.exchange import SimConfig, SimExchange  # noqa: E402

BARS_PER_YEAR = 24 * 365


class FixedWeightRebalance(Agent):
    """固定权重、按固定频率回到该权重（恒定混合）。"""

    def __init__(self, weights: dict[str, float], rebalance_every: int):
        self.weights = weights
        self.rebalance_every = rebalance_every
        self.name = f"fixedmix_rb{rebalance_every}"
        self._last = -10**9
        self._cur: dict[str, float] = {}

    def decide(self, view):
        if self._cur and view.step - self._last < self.rebalance_every:
            return None
        self._last = view.step
        self._cur = {s: self.weights.get(s, 0.0) for s in view.symbols()}
        return self._cur


def main() -> None:
    cfg = cfgmod.load("base")
    syms = cfg["universe"]["core"]
    fr = {s: store.load_bars(s, "1h")[
        ["open", "high", "low", "close", "volume", "quote_volume"]] for s in syms}
    idx = None
    for f in fr.values():
        idx = f.index if idx is None else idx.intersection(f.index)
    fr = {s: f.loc[idx] for s, f in fr.items()}
    n = len(fr[syms[0]])
    print(f"窗口 {n:,} 根  {fr[syms[0]].index[0]:%Y-%m-%d} ~ {fr[syms[0]].index[-1]:%Y-%m-%d}")

    close = pd.DataFrame({s: fr[s]["close"] for s in syms})
    _br = np.log(close / close.shift()).dropna(); bench = _br.mean(axis=1).to_numpy(); bench_index = _br.index

    eq_w = {s: 1.0 / len(syms) for s in syms}
    print("\n固定等权组合，按不同频率再平衡（零成本），对基准做 alpha 回归：")
    print(f"{'再平衡间隔':>12}{'净终值':>14}{'alpha年化':>12}{'beta':>8}{'t值':>9}{'p值':>9}")
    for k in [1, 6, 24, 72, 168, 720, 2000, 100000]:
        ag = FixedWeightRebalance(eq_w, k)
        res = SimExchange(fr, CostModel(enabled=False), CostModel(enabled=False),
                          SimConfig(initial_cash=10_000.0, warmup=300)).run(ag)
        r = np.diff(res.gross_equity) / res.gross_equity[:-1]
        r = r[np.isfinite(r)]
        at = protocol.alpha_vs_benchmark(r, bench, BARS_PER_YEAR, r_index=res.index[1: len(r)+1], b_index=bench_index)
        label = "从不" if k >= n else str(k)
        print(f"{label:>12}{res.final_gross():>14,.0f}{at.alpha_ann*100:>11.2f}%"
              f"{at.beta:>8.3f}{at.t_stat:>9.2f}{at.p_value:>9.4f}")

    print("\n结论：若'从不'再平衡的 alpha 接近 0 而高频再平衡显著为正，")
    print("      则 G1 的 100% 假阳性来自再平衡溢价，而不是评估器的统计错误。")


if __name__ == "__main__":
    main()
