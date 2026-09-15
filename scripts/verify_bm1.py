"""核验 B 那个「学习体在 be 池赢 2.63 倍」所依赖的分母。

B 的 robustness_per_pool.be: bm1_final = 41,247.85（等权再平衡基准，be 池，收资金费）
学习体 final = 108,420（我已在 audit_verify ④ 独立复现过）

如果分母我也能复现 ⇒ 2.63x 这个比值可信。
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from src import config as cfgmod  # noqa: E402
from src.data import store  # noqa: E402
from src.sim.costs import CostModel  # noqa: E402
from src.sim.exchange import SimConfig, SimExchange  # noqa: E402
from src.sim.funding import FundingTable  # noqa: E402
from src.sim.margin import MarginConfig  # noqa: E402

cfg = cfgmod.load("base")
FUNDING = FundingTable.load(os.path.join(store.project_root(), "data", "funding.csv"))
ALL3 = cfg["universe"]["core"]

# 复刻 pool_ceiling / vol_attribution 的取法：三标的交集，再切出 be
fr3 = {s: store.load_bars(s, "1h")[
    ["open", "high", "low", "close", "volume", "quote_volume"]] for s in ALL3}
ix = None
for f in fr3.values():
    ix = f.index if ix is None else ix.intersection(f.index)
fr3 = {s: f.loc[ix] for s, f in fr3.items()}
be = ["BTCUSDT", "ETHUSDT"]
fr_be = {s: fr3[s] for s in be}
print(f"三标的交集窗口 {len(fr3[ALL3[0]]):,} 根；be 池取同一窗口 {len(fr_be['BTCUSDT']):,} 根")


class Rebal:
    """每 bar 重发等权目标（= 逐 bar 再平衡）。"""

    def __init__(self, syms):
        self.w = {s: 1.0 / len(syms) for s in syms}

    def decide(self, view):
        return dict(self.w)


def go(funding_on, margin_k, tag):
    mc = (MarginConfig(initial_margin_ratio=margin_k, maintenance_margin_ratio=0.1,
                       topup_trigger_ratio=0.5) if margin_k else None)
    r = SimExchange(fr_be, CostModel(enabled=False),
                    CostModel.from_config(cfg, enabled=True),
                    SimConfig(initial_cash=1e4, warmup=300, max_gross=1.0,
                              max_exposure_per_symbol=1.0, margin=mc,
                              allow_short=True, instrument="perp"),
                    funding=(FUNDING if funding_on else None)).run(Rebal(be))
    fc = float(np.sum(r.funding_paid)) if len(r.funding_paid) else 0.0
    print(f"  {tag:<40} final {r.net_equity[-1]:>12,.0f}  资金费 {fc:>12,.1f}"
          f"  交易 {r.n_trades:,}")
    return float(r.net_equity[-1])


print()
print("=== be 池 等权逐 bar 再平衡 ===")
b1 = go(True, 1.0, "收资金费 + margin k=1")
b2 = go(False, 1.0, "不收资金费 + margin k=1")
print()
print(f"B 报的 bm1_final = 41,247.85  ⇒ 我复现 {'一致' if abs(b1-41247.85)/41247.85<0.02 else '不一致'}")
print(f"学习体 108,420（已独立复现）⇒ 比值 {108420/b1:.3f}x")
