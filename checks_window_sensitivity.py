"""口径核验：冲击占比对「成交额窗口」有多敏感。

M2 暴露的问题：同一个 BNB，用不同窗口算成交额中位数，
冲击占比从 20.6% 跳到 53.0%。定池门槛必须固定口径，否则门槛形同虚设。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.data import store  # noqa: E402
from src.sim.costs import CostModel  # noqa: E402

ORDER_USD = 10_000.0
FEE_ONE_SIDE = 0.001


def metrics(df: pd.DataFrame, label: str) -> dict:
    sigma = float(np.log(df["close"] / df["close"].shift()).dropna().std())
    v = float(df["quote_volume"].median())
    imp = sigma * np.sqrt(ORDER_USD / v)
    total = 2 * FEE_ONE_SIDE + 2 * imp
    return {"window": label, "bars": len(df), "sigma_pct": round(sigma * 100, 4),
            "vol_median_M": round(v / 1e6, 3),
            "participation_pct": round(ORDER_USD / v * 100, 4),
            "impact_bp": round(imp * 1e4, 2),
            "rt_total_bp": round(total * 1e4, 2),
            "impact_share_pct": round(2 * imp / total * 100, 1)}


print(f"{'标的':<9}{'窗口':<16}{'根数':>7}{'σ(小时)':>10}{'成交额中位':>12}"
      f"{'参与率':>9}{'冲击':>9}{'往返合计':>10}{'冲击占比':>10}")
for sym in ["BTCUSDT", "ETHUSDT", "BNBUSDT"]:
    full = store.load_bars(sym, "1h")
    cut = full.index[-1] - pd.Timedelta(days=365)
    for label, df in [("全历史", full), ("近 12 个月", full[full.index >= cut])]:
        m = metrics(df, label)
        print(f"{sym:<9}{label:<16}{m['bars']:>7,}{m['sigma_pct']:>9.4f}%"
              f"{m['vol_median_M']:>10,.1f}M{m['participation_pct']:>8.4f}%"
              f"{m['impact_bp']:>8.2f}bp{m['rt_total_bp']:>9.2f}bp"
              f"{m['impact_share_pct']:>9.1f}%")

print()
print("结论：冲击占比对窗口极其敏感，差额可达 2.5 倍。")
print("定池门槛统一采用「近 12 个月小时成交额中位数」，与实盘流动性最接近。")
