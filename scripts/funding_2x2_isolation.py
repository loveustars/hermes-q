"""最后的干净隔离：2×2。

设计：**引擎的资金费现金流在四格里保持同一套（都传同一张表）**，
      只改 agent 是否知道资金费。这样 (ON,ON) 与 (ON,OFF) 的差异
      **只能来自决策路径**，与经济效应无关。

      对照格 (OFF,OFF) 是项目现状（两边都不知道/不收费）。

读出：
  (ON,OFF) vs (ON,ON)  → 纯决策路径效应（agent 的信息扰动 ⇒ 混沌放大器）
  (OFF,OFF) vs (ON,ON) → 项目现状 vs 应有状态的总偏差
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from src import config as cfgmod  # noqa: E402
from src.agents.online import HedgeEnsemble  # noqa: E402
from src.data import store  # noqa: E402
from src.sim.costs import CostModel  # noqa: E402
from src.sim.exchange import SimConfig, SimExchange  # noqa: E402
from src.sim.funding import FundingTable  # noqa: E402
from src.sim.margin import MarginConfig  # noqa: E402

cfg = cfgmod.load("base")
FUNDING = FundingTable.load(os.path.join(store.project_root(), "data", "funding.csv"))
COST_BY_SYM = {"BTCUSDT": 0.0012, "ETHUSDT": 0.0013, "BNBUSDT": 0.0014}
SYMS = ["BTCUSDT", "ETHUSDT"]
fr = {s: store.load_bars(s, "1h")[
    ["open", "high", "low", "close", "volume", "quote_volume"]] for s in SYMS}
ix = None
for f in fr.values():
    ix = f.index if ix is None else ix.intersection(f.index)
fr = {s: f.loc[ix] for s, f in fr.items()}
MG = MarginConfig(initial_margin_ratio=1.0, maintenance_margin_ratio=0.1,
                  topup_trigger_ratio=0.5)


def go(engine_funding, agent_knows, tag):
    ag = HedgeEnsemble(SYMS, cost_rate=np.array([COST_BY_SYM[s] for s in SYMS]),
                       eta=0.20, band=0.20, max_exposure=1.0,
                       funding=(FUNDING if agent_knows else None))
    r = SimExchange(fr, CostModel(enabled=False),
                    CostModel.from_config(cfg, enabled=True),
                    SimConfig(initial_cash=1e4, warmup=300, max_gross=1.0,
                              max_exposure_per_symbol=1.0, margin=MG,
                              allow_short=True, instrument="perp"),
                    funding=(FUNDING if engine_funding else None)).run(ag)
    fc = float(np.sum(r.funding_paid)) if len(r.funding_paid) else 0.0
    print(f"  {tag:<42} 终值 {r.net_equity[-1]:>12,.0f}  成本 {r.cost_paid.sum():>9,.0f}"
          f"  资金费现金流 {fc:>11,.1f}")
    return float(r.net_equity[-1])


print("2 标的 (BTC, ETH)，窗口", len(fr[SYMS[0]]), "根   η=0.20 band=0.20")
print()
a = go(False, False, "引擎OFF agent不知  ← 项目现状")
b = go(True, False, "引擎ON  agent不知")
c = go(True, True, "引擎ON  agent知道  ← 应当如此")
d = go(False, True, "引擎OFF agent知道")

print()
print("=== 读法 ===")
print(f"  纯决策路径效应： (ON,不知) vs (ON,知道) = {c / b:.4f}x"
      f"   （引擎现金流相同，差异只能来自 agent 的信息扰动）")
print(f"  现状 vs 应有：   (OFF,不知) vs (ON,知道) = {c / a:.4f}x")
print(f"  总偏差倍数：     {a / c:.2f}x" if c > 0 else "")
