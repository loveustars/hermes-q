"""carry alpha 的 t 统计量对 HAC 滞后阶数的敏感性 + 按现金流频率重采样。

问题背景：资金费每 8 小时才结算一次，两次结算之间 carry 的净值几乎是一条直线，
所以收益序列在 8 根 bar 的尺度上高度自相关。
默认 Newey-West（4·(n/100)^(2/9)，n=5.7 万时约 16 阶）不足以吸收，
于是 t 统计量被**大幅高估**——实测滞后阶数从 16 加到 672，t 从 19 掉到 6.3 仍未收敛。

两种修正：
  (A) 加大 HAC 滞后阶数，看 t 是否收敛
  (B) **按现金流频率（8 小时）把净值聚合成 8h 收益再检验** —— 这是更干净的做法，
      因为聚合后的序列自相关结构大大减弱，且观测数回到"结算次数"这个真实的独立样本量

心态：先假设 t 是虚高的，再找出它虚高多少，而不是用它来庆祝。
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
from src.eval.benchmark import equal_weight_buyhold  # noqa: E402
from src.registry.runs import Run  # noqa: E402
from src.sim.carry import CarryConfig, CarrySimulator  # noqa: E402
from src.sim.funding import load_default  # noqa: E402

BARS_PER_YEAR = 24 * 365
LAGS = [16, 48, 168, 672]
AGG = 8          # 资金费结算间隔（小时）

HYPOTHESIS = {
    "question": "M10 报的 t=20.39 是否被逐 bar 口径虚高了？虚高多少？",
    "expected": "t 不是虚高的——但它未经检验，且序列峰度极高（1900+），先假设它虚高",
    "decision_rule": "以按绝对时间桶聚合到 8h 现金流频率的口径为准；"
                     "若聚合后显著性不成立，则 M10 的判定必须撤回",
}


def load_frames(symbols, source="binance"):
    frames = {s: store.load_bars(s, "1h", source=source)[
        ["open", "high", "low", "close", "volume", "quote_volume"]] for s in symbols}
    idx = None
    for f in frames.values():
        idx = f.index if idx is None else idx.intersection(f.index)
    return {s: f.loc[idx] for s, f in frames.items()}


def resample_equity(eq: np.ndarray, k: int):
    """把逐 bar 净值聚合成每 k 根的净值（取每 k 根的最后一个点）。"""
    n = (len(eq) // k) * k
    return eq[:n].reshape(-1, k)[:, -1]


def rets_and_index(eq, index) -> tuple[np.ndarray, pd.DatetimeIndex]:
    """从净值序列取收益与对应时间戳。

    用**尾部切片**对齐长度：净值数组与索引长度不一定严格相差 1，
    早期版本按 1:len+1 硬切会错位一根。
    """
    vals = np.asarray(eq, dtype=float)
    r = np.diff(vals) / vals[:-1]
    r = np.where(np.isfinite(r), r, 0.0)
    idx = pd.DatetimeIndex(index)[-len(r):]
    return r, idx


def main() -> None:
    cfg = cfgmod.load("base")
    syms = cfg["universe"]["core"]
    ft = load_default()
    spots = load_frames(syms, "binance")
    perps = load_frames(syms, "binanceperp")

    starts = [min(d) for d in (ft.rates_by_hour.get(s) or {} for s in syms) if d]
    start = pd.Timestamp(max(starts), unit="ms", tz="UTC")

    # 基准必须建在与 carry 相同的公共时间轴上：
    # spot ∩ perp 的交集（因为 carry 需要两条腿都有数据）。
    common = None
    for s in syms:
        i = spots[s].index.intersection(perps[s].index)
        i = i[i >= start]
        common = i if common is None else common.intersection(i)
    bm_frames = {s: spots[s].loc[common] for s in syms}
    bm = equal_weight_buyhold(bm_frames, cfg)
    bm_r_1h, bm_idx_1h = rets_and_index(bm.equity, bm.index)
    bm_r_8h, bm_idx_8h = protocol.aggregate_to_clock(bm_idx_1h, bm_r_1h, "8h")
    print(f"公共时间轴 {common[0]:%Y-%m-%d} ~ {common[-1]:%Y-%m-%d}  {len(common):,} 根")
    print("心态：先假设 t=20 是虚高的，再找出它虚高多少。\n")

    rows = []
    with Run("m10_hac_sensitivity",
             {"agg_hours": AGG, "lags": LAGS, "grid_n_trials": 72},
             HYPOTHESIS) as run:
        for s in syms:
            c = CarryConfig(initial_capital=10_000.0, notional_ratio=0.4,
                            initial_margin_ratio=0.5, rebalance_every=720,
                            warmup=300)
            res = CarrySimulator(spots[s], perps[s], ft, c, s).run()
            r, idx = rets_and_index(res.equity, res.index)

            rec = {"symbol": s, "annualized": res.annualized(),
                   "n_bars": int(len(r)), "lag_sweep": {}}
            print(f"══ {s}  年化 {res.annualized()*100:.2f}%  bar 数 {len(r):,}")
            print("  (A) 逐 bar 检验、逐步加大 HAC 滞后：")
            for lag in LAGS:
                at = protocol.alpha_vs_benchmark(r, bm_r_1h, BARS_PER_YEAR,
                                                 hac_lags=lag, r_index=idx,
                                                 b_index=bm_idx_1h)
                rec["lag_sweep"][lag] = {"t": at.t_stat, "p": at.p_value,
                                         "n": int(at.n)}
                print(f"      滞后 {lag:>4} → t = {at.t_stat:>7.2f}"
                      f"   p = {at.p_value:.6f}   n = {at.n:,}")

            print(f"  (B) 按绝对时间桶聚合到 {AGG}h 再检验：")
            # 用协议里的标准实现，脚本不再自己抄一份（避免两处实现分叉）
            r8, idx8 = protocol.aggregate_to_clock(idx, r, f"{AGG}h")
            at8 = protocol.alpha_vs_benchmark(r8, bm_r_8h, 24 * 365 / AGG,
                                              r_index=idx8, b_index=bm_idx_8h)
            ev8 = protocol.evaluate_strategy(r8, bm_r_8h, n_trials=72,
                                             bars_per_year=24 * 365 / AGG,
                                             r_index=idx8, b_index=bm_idx_8h)
            at_default = protocol.alpha_vs_benchmark(r, bm_r_1h, BARS_PER_YEAR,
                                                     r_index=idx,
                                                     b_index=bm_idx_1h)
            inflation = (at_default.t_stat / at8.t_stat) if at8.t_stat > 0 else None
            rec["aggregated_8h"] = {
                "n": int(len(r8)), "alpha_ann": at8.alpha_ann,
                "beta": at8.beta, "t": at8.t_stat, "p": at8.p_value,
                "ci_low": ev8["alpha_bootstrap"]["annualized_ci_low"],
                "ci_high": ev8["alpha_bootstrap"]["annualized_ci_high"],
                "dsr": ev8["dsr"]["dsr"], "verdict": ev8["verdict"]}
            rec["default_1h_t"] = at_default.t_stat
            rec["t_inflation"] = inflation
            rows.append(rec)

            print(f"      观测数 {len(r8):,}（≈ 结算次数）  对齐后 {at8.n:,}")
            print(f"      alpha 年化 {at8.alpha_ann*100:>6.2f}%   "
                  f"beta {at8.beta:>6.3f}   t = {at8.t_stat:>6.2f}   "
                  f"p = {at8.p_value:.6f}")
            print(f"      DSR = {ev8['dsr']['dsr']:.4f}   "
                  f"bootstrap CI "
                  f"[{ev8['alpha_bootstrap']['annualized_ci_low']*100:.2f}%, "
                  f"{ev8['alpha_bootstrap']['annualized_ci_high']*100:.2f}%] 年化")
            print(f"      判定：{ev8['verdict']}")
            print(f"  对比：默认滞后 t={at_default.t_stat:.2f} → "
                  f"聚合到 {AGG}h t={at8.t_stat:.2f}")
            if inflation:
                print(f"  ⇒ 默认口径把 t 高估了约 {inflation:.1f} 倍；以聚合口径为准，"
                      f"显著性{'仍然成立' if at8.p_value < 0.05 else '不成立'}\n")
            else:
                print("  ⇒ 聚合后回归退化，需检查对齐\n")

        run.log("hac_sensitivity", rows)
        run.record_metrics({r_["symbol"]: {
            "t_1h_default": r_["default_1h_t"],
            "t_8h_aggregated": r_["aggregated_8h"]["t"],
            "t_inflation": r_["t_inflation"],
            "alpha_ann_8h": r_["aggregated_8h"]["alpha_ann"],
            "ci_low_8h": r_["aggregated_8h"]["ci_low"],
            "ci_high_8h": r_["aggregated_8h"]["ci_high"],
            "verdict_8h": r_["aggregated_8h"]["verdict"]} for r_ in rows})
        print(f"run 目录: {run.dir}")


if __name__ == "__main__":
    main()
