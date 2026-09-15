"""独立复核引擎的资金费算术：手算 Σ(units × price × rate) 与引擎的合计对比。

若两者一致 ⇒ 引擎算术正确，是我的粗估偏差。
若引擎显著更大 ⇒ 计费有 bug（量纲、频率、重复计费等）。
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
from src.sim.funding import FundingTable, floor_hour  # noqa: E402

cfg = cfgmod.load("base")
SYMS = cfg["universe"]["core"]
FUNDING = FundingTable.load(os.path.join(store.project_root(), "data", "funding.csv"))

print("=== 资金费数据本身的量纲 ===")
for s in SYMS:
    d = FUNDING.rates_by_hour.get(s, {})
    v = np.array(list(d.values()))
    print(f"  {s}: {len(v):,} 条  均值 {v.mean():.6f}  中位 {np.median(v):.6f}  "
          f"min {v.min():.6f}  max {v.max():.6f}")
    print(f"      （若为每 8h 费率：均值 × 1095 = {v.mean()*1095*100:.2f}%/年）")

fr = {s: store.load_bars(s, "1h")[
    ["open", "high", "low", "close", "volume", "quote_volume"]] for s in SYMS}
ix = None
for f in fr.values():
    ix = f.index if ix is None else ix.intersection(f.index)
fr = {s: f.loc[ix] for s, f in fr.items()}


class Once:
    def __init__(self, w):
        self.w, self.done = dict(w), False

    def decide(self, view):
        if self.done:
            return None
        self.done = True
        return dict(self.w)


res = SimExchange(fr, CostModel(enabled=False),
                  CostModel.from_config(cfg, enabled=True),
                  SimConfig(initial_cash=1e4, warmup=300, max_gross=1.0,
                            max_exposure_per_symbol=1.0, margin=None,
                            allow_short=True, instrument="perp"),
                  funding=FUNDING).run(
    Once({"BTCUSDT": 1.0, "ETHUSDT": 0.0, "BNBUSDT": 0.0}))

eng_total = float(np.sum(res.funding_paid))
print()
print(f"=== 引擎报告的合计 ===")
print(f"  资金费合计 {eng_total:,.2f}   结算笔数 {res.n_funding_events:,}   "
      f"记录长度 {len(res.net_equity):,}")

# 手算：静态持仓 units=初始名义/建仓价，一直到死亡那根
entry_bar = 301
entry_px = float(fr["BTCUSDT"]["open"].iloc[entry_bar])
units = 10_000.0 / entry_px
print()
print(f"=== 手算（静态持仓）===")
print(f"  建仓 bar {entry_bar}  open {entry_px:,.2f}   units {units:.6f} BTC")

manual = 0.0
n = 0
for j in range(entry_bar, len(res.net_equity)):
    ts_ms = int(res.index[j].timestamp() * 1000)
    r = FUNDING.rates_by_hour.get("BTCUSDT", {}).get(floor_hour(ts_ms))
    if r is None:
        continue
    px = float(fr["BTCUSDT"]["close"].iloc[j])
    manual += units * px * r
    n += 1
print(f"  手算合计 {manual:,.2f}   笔数 {n:,}")
print(f"  引擎 / 手算 = {eng_total / manual:.4f}" if manual else "  手算为 0")
print()
print("  逐年手算（看是否有异常年份）：")
for y in range(2018, 2023):
    sub = 0.0
    for j in range(entry_bar, len(res.net_equity)):
        if res.index[j].year != y:
            continue
        ts_ms = int(res.index[j].timestamp() * 1000)
        r = FUNDING.rates_by_hour.get("BTCUSDT", {}).get(floor_hour(ts_ms))
        if r is None:
            continue
        sub += units * float(fr["BTCUSDT"]["close"].iloc[j]) * r
    print(f"    {y}: {sub:>10,.1f}")
