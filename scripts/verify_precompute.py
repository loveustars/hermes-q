"""预计算等价性验证 —— 优化不许改变任何数字。

逐根现算 vs 滚动预计算，在多个换仓频率下比对两条净值曲线必须完全一致，
并给出加速比。

用法：python3 scripts/verify_precompute.py
"""
from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from src import config as cfgmod  # noqa: E402
from src.agents import baselines as B  # noqa: E402
from src.agents.online import HedgeEnsemble  # noqa: E402
from src.data import store  # noqa: E402
from src.env.market_view import MarketView  # noqa: E402
from src.sim.costs import CostModel  # noqa: E402
from src.sim.exchange import SimConfig, SimExchange  # noqa: E402

BARS = 20_000


class SlowSim(SimExchange):
    """禁用预计算的版本，用于对拍。"""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.pre = None


def main() -> None:
    cfg = cfgmod.load("base")
    syms = cfg["universe"]["core"]
    fr = {s: store.load_bars(s, "1h")[
        ["open", "high", "low", "close", "volume", "quote_volume"]] for s in syms}
    idx = None
    for f in fr.values():
        idx = f.index if idx is None else idx.intersection(f.index)
    fr = {s: f.loc[idx].iloc[-BARS:] for s, f in fr.items()}
    print(f"对拍窗口 {len(fr[syms[0]]):,} 根")

    # 先单独验证 MarketView 的数值一致
    print("\n=== 1. MarketView 数值一致性 ===")
    from src.env.market_view import Precomputed
    pre = Precomputed(fr, {168, 720}, {168, 720})
    worst_s, worst_v = 0.0, 0.0
    for t in (300, 5_000, 12_345, len(fr[syms[0]]) - 1):
        v_fast = MarketView(fr, t, pre)
        v_slow = MarketView(fr, t, None)
        for s in syms:
            for n in (168, 720):
                d1 = abs(v_fast.sigma(s, n) - v_slow.sigma(s, n))
                d2 = abs(v_fast.period_notional(s, n) - v_slow.period_notional(s, n))
                worst_s = max(worst_s, d1)
                worst_v = max(worst_v, d2)
    print(f"  sigma 最大绝对差 {worst_s:.3e}   "
          f"成交额中位数最大绝对差 {worst_v:.3e}")
    assert worst_s < 1e-12 and worst_v < 1e-6, "预计算与现算不等价，优化被拒绝"

    # 再对拍整条净值曲线
    print("\n=== 2. 净值曲线对拍（含在线学习体）===")
    print(f"{'策略':<26}{'慢路径终值':>16}{'快路径终值':>16}{'最大偏差':>12}{'加速比':>9}")
    cases = [
        ("random_rb1", B.RandomWeights(seed=3, rebalance_every=1)),
        ("random_rb24", B.RandomWeights(seed=3, rebalance_every=24)),
        ("random_rb168", B.RandomWeights(seed=3, rebalance_every=168)),
        ("bh_BTCUSDT", B.SingleAssetBuyHold("BTCUSDT")),
    ]
    rates = np.array([0.0012, 0.0013, 0.0014])
    cases.append(("hedge_eta0.05_b0.05",
                  HedgeEnsemble(syms, cost_rate=rates, eta=0.05, band=0.05)))

    ok = True
    for name, ag in cases:
        cm_g = CostModel(enabled=False)
        cm_n = CostModel.from_config(cfg, enabled=True)
        sc = SimConfig(initial_cash=10_000.0, warmup=300)
        t0 = time.time()
        r_slow = SlowSim(fr, cm_g, cm_n, sc).run(ag)
        t_slow = time.time() - t0
        # 重建 agent 以保证状态干净
        ag2 = _rebuild(name, ag, syms, rates)
        t0 = time.time()
        r_fast = SimExchange(fr, cm_g, cm_n, sc).run(ag2)
        t_fast = time.time() - t0
        dev = float(np.max(np.abs(r_slow.net_equity - r_fast.net_equity)))
        rel = dev / 10_000.0
        speed = t_slow / max(t_fast, 1e-9)
        print(f"{name:<26}{r_slow.final_net():>16,.2f}{r_fast.final_net():>16,.2f}"
              f"{dev:>12.2e}{speed:>8.1f}x")
        if rel > 1e-9:
            ok = False
            print(f"    !! 偏差过大，超出浮点误差：{rel:.2e}")

    print(f"\n结论：{'通过，预计算不改变任何数字' if ok else '不通过'}")
    raise SystemExit(0 if ok else 1)


def _rebuild(name, ag, syms, rates):
    if name == "random_rb1":
        return B.RandomWeights(seed=3, rebalance_every=1)
    if name == "random_rb24":
        return B.RandomWeights(seed=3, rebalance_every=24)
    if name == "random_rb168":
        return B.RandomWeights(seed=3, rebalance_every=168)
    if name == "bh_BTCUSDT":
        return B.SingleAssetBuyHold("BTCUSDT")
    return HedgeEnsemble(syms, cost_rate=rates, eta=0.05, band=0.05)


if __name__ == "__main__":
    main()
