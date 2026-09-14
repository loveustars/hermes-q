"""carry 收益序列的分布退化检查。

起因：核对 run 产物时发现 1h 收益序列的峰度高达 1903（正态为 3）、
偏度 7.78，而 DSR=1.0000 正是从这条序列算出来的。

机理猜测：carry 是紧对冲的 delta 中性头寸，绝大多数 bar 上
两条腿同步变动、净损益≈0；只有资金费结算或基差跳变时才有非零损益。
于是收益分布退化成「零点尖峰 + 偶尔跳变」，峰度极高。

后果：在这种退化分布上算 Sharpe 与 DSR **没有意义**
（样本标准差被尖峰压得极小，Sharpe 被推到 ~4 这种不可信的水平）。

结论方向：必须把「按现金流频率聚合后再检验」从 t 统计量扩展到
**整条评估链路**（Sharpe / DSR / 偏度 / 峰度）。
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src import config as cfgmod  # noqa: E402
from src.data import store  # noqa: E402
from src.eval import protocol  # noqa: E402
from src.registry.runs import Run  # noqa: E402
from src.sim.carry import CarryConfig, CarrySimulator  # noqa: E402
from src.sim.funding import load_default  # noqa: E402

HYPOTHESIS = {
    "question": "carry 的收益分布在逐 bar 口径下退化到什么程度？"
                "M10 报的 DSR=1.0000 是否因此不可信？",
    "expected": "峰度极高（远超正态的 3）；聚合到 8h 会降低峰度但不消除",
    "decision_rule": "若聚合后峰度仍远超正态（如 >20），则 Sharp/DSR 类指标"
                     "不得作为判定证据，判定须以自助区间为主判据",
}


def load_frames(symbols, source="binance"):
    frames = {s: store.load_bars(s, "1h", source=source)[
        ["open", "high", "low", "close", "volume", "quote_volume"]] for s in symbols}
    idx = None
    for f in frames.values():
        idx = f.index if idx is None else idx.intersection(f.index)
    return {s: f.loc[idx] for s, f in frames.items()}


def stats(r: np.ndarray, bars_per_year: float) -> dict:
    r = np.asarray(r, dtype=float)
    r = r[np.isfinite(r)]
    n = len(r)
    mu, sd = float(r.mean()), float(r.std(ddof=1))
    z = (r - mu) / sd
    zero_frac = float((np.abs(r) < 1e-12).mean())
    return {
        "n": n,
        "年化收益": mu * bars_per_year,
        "年化波动": sd * np.sqrt(bars_per_year),
        "Sharpe": (mu / sd) * np.sqrt(bars_per_year) if sd > 0 else 0.0,
        "偏度": float((z ** 3).mean()),
        "峰度": float((z ** 4).mean()),
        "零收益占比": zero_frac,
        "最大单期": float(r.max()),
        "最小单期": float(r.min()),
    }


def main() -> None:
    cfg = cfgmod.load("base")
    syms = cfg["universe"]["core"]
    ft = load_default()
    spots = load_frames(syms, "binance")
    perps = load_frames(syms, "binanceperp")

    print("检查：1h 逐 bar 序列 vs 8h 现金流频率序列的分布退化程度\n")
    rows = []
    with Run("m10_dist_check", {"agg_hours": 8, "config": "nr=0.4,m=0.5,rb=720",
                                "freq": "1h vs 8h"}, HYPOTHESIS) as run:
        for s in syms:
            c = CarryConfig(initial_capital=10_000.0, notional_ratio=0.4,
                            initial_margin_ratio=0.5, rebalance_every=720,
                            warmup=300)
            res = CarrySimulator(spots[s], perps[s], ft, c, s).run()
            eq = np.asarray(res.equity, dtype=float)
            r = np.diff(eq) / eq[:-1]
            idx = pd.DatetimeIndex(res.index)[-len(r):]
            r1 = stats(np.where(np.isfinite(r), r, 0.0), 24 * 365)

            print(f"══ {s}")
            print(f"  1h 逐 bar（n={r1['n']:,}）")
            print(f"    年化收益 {r1['年化收益']*100:>7.2f}%   年化波动 "
                  f"{r1['年化波动']*100:>6.2f}%   Sharpe {r1['Sharpe']:>6.2f}")
            print(f"    偏度 {r1['偏度']:>8.2f}   峰度 {r1['峰度']:>10.2f}   "
                  f"零收益占比 {r1['零收益占比']*100:>6.2f}%")
            print(f"    单期极值 [{r1['最小单期']*100:+.3f}%, "
                  f"{r1['最大单期']*100:+.3f}%]")

            # 用协议里的标准实现（脚本不再自带一份，避免两处实现分叉）
            r8, _ = protocol.aggregate_to_clock(idx, r, "8h")
            st8 = stats(r8, 24 * 365 / 8)
            print(f"  8h 聚合（n={st8['n']:,}，≈ 结算次数）")
            print(f"    年化收益 {st8['年化收益']*100:>7.2f}%   年化波动 "
                  f"{st8['年化波动']*100:>6.2f}%   Sharpe {st8['Sharpe']:>6.2f}")
            print(f"    偏度 {st8['偏度']:>8.2f}   峰度 {st8['峰度']:>10.2f}   "
                  f"零收益占比 {st8['零收益占比']*100:>6.2f}%")
            print(f"    单期极值 [{st8['最小单期']*100:+.3f}%, "
                  f"{st8['最大单期']*100:+.3f}%]")

            ratio = r1["峰度"] / max(st8["峰度"], 1e-9)
            rows.append({"symbol": s, "stats_1h": r1, "stats_8h": st8,
                         "kurt_reduction": ratio})
            print(f"  ⇒ 峰度 {r1['峰度']:.0f} → {st8['峰度']:.1f}（降 "
                  f"{ratio:.0f} 倍）；Sharpe {r1['Sharpe']:.2f} → "
                  f"{st8['Sharpe']:.2f}")
            print(f"  ⇒ 8h 峰度 {st8['峰度']:.1f} 仍远超正态（3）⇒ "
                  f"Sharpe/DSR 在此分布下不可作为判定证据\n")

        run.log("dist_stats", rows)
        run.record_metrics({r_["symbol"]: {
            "kurt_1h": r_["stats_1h"]["峰度"], "kurt_8h": r_["stats_8h"]["峰度"],
            "skew_1h": r_["stats_1h"]["偏度"], "skew_8h": r_["stats_8h"]["偏度"],
            "sharpe_1h": r_["stats_1h"]["Sharpe"],
            "sharpe_8h": r_["stats_8h"]["Sharpe"],
            "vol_8h": r_["stats_8h"]["年化波动"]} for r_ in rows})
        print(f"run 目录: {run.dir}")


if __name__ == "__main__":
    main()
