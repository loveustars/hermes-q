"""M5 冒烟测试 —— 在短窗口上确认在线学习体能跑通、不归零、确实在换手。"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from src import config as cfgmod  # noqa: E402
from src.agents.online import HedgeEnsemble  # noqa: E402
from src.data import store  # noqa: E402
from src.sim.costs import CostModel  # noqa: E402
from src.sim.exchange import SimConfig, SimExchange  # noqa: E402

BARS = 6_000

cfg = cfgmod.load("base")
syms = cfg["universe"]["core"]
fr = {s: store.load_bars(s, "1h")[
    ["open", "high", "low", "close", "volume", "quote_volume"]] for s in syms}
idx = None
for f in fr.values():
    idx = f.index if idx is None else idx.intersection(f.index)
fr = {s: f.loc[idx].iloc[-BARS:] for s, f in fr.items()}
print(f"窗口 {len(fr[syms[0]]):,} 根  "
      f"{fr[syms[0]].index[0]:%Y-%m-%d} ~ {fr[syms[0]].index[-1]:%Y-%m-%d}")

rates = np.array([0.0012, 0.0013, 0.0014])
ag = HedgeEnsemble(syms, cost_rate=rates, eta=0.05, band=0.05)
t0 = time.time()
res = SimExchange(fr, CostModel(enabled=False),
                  CostModel.from_config(cfg, enabled=True),
                  SimConfig(initial_cash=10_000.0, warmup=300)).run(ag)
dt = time.time() - t0

print(f"耗时 {dt:.1f}s   成交笔数 {res.n_trades:,}   归零={res.insolvent}")
print(f"毛终值 {res.final_gross():>12,.0f}   净终值 {res.final_net():>12,.0f}"
      f"   累计成本 {res.cost_paid.sum():>10,.0f}")
print(f"换手名义额 {res.turnover_notional.sum():>14,.0f}"
      f"  = 本金 {res.turnover_notional.sum()/10000:,.1f} 倍")
print(f"专家数 {ag.K}")
print("最终专家权重:")
for k, v in sorted(zip(ag.expert_names(), ag.p), key=lambda kv: -kv[1]):
    print(f"   {k:<14}{v:>8.4f}  " + "█" * max(1, int(v * 50)))
ent = float(np.mean(ag.log["entropy"]))
print(f"平均专家熵 {ent:.4f}   上限 ln(K) = {np.log(ag.K):.4f}"
      f"   （越接近上限＝越没学到东西）")
emitted = sum(1 for t in ag.log["turnover"] if t > 0)
print(f"发出调仓 {emitted:,} 次 / 共 {len(ag.log['step']):,} 根 bar"
      f"  = 调仓率 {emitted/len(ag.log['step'])*100:.1f}%")
assert res.final_net() > 0, "学习体在短窗口上就归零了，参数需要调整"
assert not res.insolvent, "学习体归零"
print("\n冒烟测试通过")
