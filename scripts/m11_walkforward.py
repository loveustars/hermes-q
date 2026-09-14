"""G3 样本外验证 —— carry 的「有边际」是真能泛化，还是网格挑出来的。

为什么要做：M10 的 72 次网格本身就是**选择**。选出最优配置再在全样本上
报 alpha，等于用同一批数据既选又评。G3 要回答两个问题：

  A. **滚动的样本外**：每折只用该折样本内的数据选配置，再在紧接着的
     样本外窗口上评估。选出来的配置在样本外还赚不赚？
  B. **封存样本（一次性）**：用前 75% 选配置，然后**开封**后 25% 评一次。
     封存段整个项目只允许动一次（铁律 4），所以必须显式 allow=True，
     且结果无论好坏都照写。

协议要求：所有检验都在 8 小时现金流频率上做（PLAN §14.7），
且以**自助区间**为主判据（PLAN §14.4b：分布峰度 44~105，DSR 与正态 p 值不算证据）。
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
from src.eval.holdout import HoldoutGuard  # noqa: E402
from src.registry.runs import Run  # noqa: E402
from src.sim.carry import CarryConfig, CarrySimulator  # noqa: E402
from src.sim.funding import load_default  # noqa: E402

AGG = "8h"
BARS_PER_8H = 24 * 365 / 8

# 与 M10 同一网格（nr × (1+m) ≤ 1 的可行组合 × 再平衡间隔）
GRID = [(nr, m, rb)
        for nr in (0.2, 0.3, 0.4)
        for m in (0.5, 1.0, 2.0)
        if nr * (1.0 + m) <= 1.0
        for rb in (24, 168, 720)]

TRAIN_W = 24 * 365          # 1 年
TEST_W = 24 * 182           # 半年
STEP = 24 * 182


def load_frames(symbols):
    """加载现货与永续，并**统一到同一条公共时间轴**。

    这条轴 = 所有标的的 (现货 ∩ 永续) 的交集。carry 需要两条腿同时有数据，
    且资金费只在永续数据存在时才有 —— 实测该轴始于 2020-02-10。

    早期版本分别对现货与永续取"跨标的交集"，得到两条不同的索引
    （现货轴始于 2017-11、永续轴始于 2020-02），却按同一个位置区间去切两者，
    于是窗口整体错位、所有配置在无永续数据的窗口里全部强平 → 折数恒为 0。
    """
    def _one(src):
        fs = {s: store.load_bars(s, "1h", source=src)[
            ["open", "high", "low", "close", "volume", "quote_volume"]]
            for s in symbols}
        idx = None
        for f in fs.values():
            idx = f.index if idx is None else idx.intersection(f.index)
        return {s: f.loc[idx] for s, f in fs.items()}

    sp_all, pp_all = _one("binance"), _one("binanceperp")
    idx = None
    for s in symbols:
        i = sp_all[s].index.intersection(pp_all[s].index)
        idx = i if idx is None else idx.intersection(i)
    return {s: sp_all[s].loc[idx] for s in symbols}, \
           {s: pp_all[s].loc[idx] for s in symbols}


def make_cfg(nr, m, rb) -> CarryConfig:
    return CarryConfig(initial_capital=10_000.0, notional_ratio=nr,
                       initial_margin_ratio=m, rebalance_every=rb, warmup=300)


def select_best(spots, perps, ft, symbol, s0, s1):
    """在 [s0, s1) 样本内选最优配置（按年化，剔除强平）。"""
    best = None
    for (nr, m, rb) in GRID:
        try:
            r = CarrySimulator(spots.iloc[s0:s1], perps.iloc[s0:s1], ft,
                               make_cfg(nr, m, rb), symbol).run()
        except Exception:
            continue
        if r.liquidated:
            continue
        ann = r.annualized()
        if best is None or ann > best[1]:
            best = ((nr, m, rb), ann, r)
    return best


def rets8(eq, index):
    """净值 → 收益 → 按 8h 现金流频率（绝对时间桶）聚合。"""
    eq = np.asarray(eq, dtype=float)
    r = np.diff(eq) / eq[:-1]
    r = np.where(np.isfinite(r), r, 0.0)
    idx = pd.DatetimeIndex(index)[-len(r):]
    return protocol.aggregate_to_clock(idx, r, AGG)


def main() -> None:
    cfg = cfgmod.load("base")
    syms = cfg["universe"]["core"]
    ft = load_default()
    spots, perps = load_frames(syms)

    idx = spots[syms[0]].index
    n = len(idx)
    g = HoldoutGuard(n=n, fraction=0.25, allow=False,
                     log_path=os.path.join("runs", "_holdout_access.log"))
    print(f"公共时间轴 {idx[0]:%Y-%m-%d} ~ {idx[-1]:%Y-%m-%d}  共 {n:,} 根")
    print(f"训练段 [0, {g.cut:,})   封存段 [{g.cut:,}, {n:,})  "
          f"约 {(n - g.cut) / (24 * 365):.2f} 年"
          f"（{idx[g.cut]:%Y-%m-%d} ~ {idx[-1]:%Y-%m-%d}）")
    print(f"网格 {len(GRID)} 个配置；全部检验在 {AGG} 现金流频率上做\n")

    bm = equal_weight_buyhold({s: spots[s] for s in syms}, cfg)
    bm_eq = np.asarray(bm.equity, dtype=float)
    bm_r = np.diff(bm_eq) / bm_eq[:-1]
    bm_r = np.where(np.isfinite(bm_r), bm_r, 0.0)
    bm_idx = pd.DatetimeIndex(bm.index)[-len(bm_r):]
    bm8, bm8_idx = protocol.aggregate_to_clock(bm_idx, bm_r, AGG)

    # 铁律 5：假设与判定阈值必须在跑之前登记
    hypothesis = {
        "question": "M10 的 carry「有边际」是网格挑出来的，还是能泛化到样本外？",
        "expected": "若 alpha 来自真实结构，样本外年化应与样本内同量级（3%~6%）；"
                    "若网格挑选是主因，样本外应明显退化甚至转负",
        "decision_rule": "以自助区间为主判据（分布峰度 44~105，DSR 与正态 p 值不算证据）："
                         "封存段自助 95% 下界 > 0 且样本外年化为正 ⇒ 确认有边际；"
                         "否则记为未通过样本外验证，并报告退化幅度",
        "holdout_fraction": 0.25,
        "agg": AGG,
        "grid_size": len(GRID),
    }

    with Run("m11_walkforward", {"agg": AGG, "grid": GRID,
                                 "train_w": TRAIN_W, "test_w": TEST_W,
                                 "holdout_fraction": 0.25}, hypothesis) as run:
        wf_all, hold_rows = {}, []

        # ============ A. 滚动样本外（严格只用训练段）============
        print("=" * 76)
        print("A. 滚动样本外：每折样本内选配置 → 紧接的样本外窗口评估")
        print("=" * 76)
        for s in syms:
            sp, pp = spots[s], perps[s]
            rows, start = [], TRAIN_W
            while start + TEST_W <= g.cut:
                g.assert_clean([start + TEST_W - 1])      # 越界即抛
                best = select_best(sp, pp, ft, s, start - TRAIN_W, start)
                if best is None:
                    start += STEP
                    continue
                (nr, m, rb), ins_ann, _ = best
                r_oos = CarrySimulator(sp.iloc[start:start + TEST_W],
                                       pp.iloc[start:start + TEST_W], ft,
                                       make_cfg(nr, m, rb), s).run()
                o8, i8 = rets8(r_oos.equity, r_oos.index)
                at = protocol.alpha_vs_benchmark(o8, bm8, BARS_PER_8H,
                                                 r_index=i8, b_index=bm8_idx)
                rows.append({"from": f"{idx[start]:%Y-%m-%d}",
                             "config": f"nr={nr},m={m},rb={rb}",
                             "in": ins_ann, "oos": r_oos.annualized(),
                             "liquidated": bool(r_oos.liquidated),
                             "alpha": at.alpha_ann, "t": at.t_stat, "n": at.n})
                start += STEP
            wf_all[s] = rows
            print(f"\n{s}   折数 {len(rows)}")
            print(f"  {'窗口起点':<12}{'选中配置':<20}{'样本内':>10}{'样本外':>10}"
                  f"{'样本外alpha':>13}{'t':>8}")
            for r in rows:
                print(f"  {r['from']:<12}{r['config']:<20}"
                      f"{r['in']*100:>9.2f}%{r['oos']*100:>9.2f}%"
                      f"{r['alpha']*100:>12.2f}%{r['t']:>8.2f}"
                      f"{'  ← 强平' if r['liquidated'] else ''}")
            if rows:
                oos = np.array([r["oos"] for r in rows])
                al = np.array([r["alpha"] for r in rows])
                uni = sorted({r["config"] for r in rows})
                print(f"  样本外年化 {oos.mean()*100:+.2f}%（均值）  "
                      f"为正 {int((oos > 0).sum())}/{len(oos)}  "
                      f"最差 {oos.min()*100:+.2f}%")
                print(f"  样本外 alpha 均值 {al.mean()*100:+.2f}%  "
                      f"为正 {int((al > 0).sum())}/{len(al)}")
                print(f"  选中配置 {len(uni)} 种 / {len(rows)} 折"
                      f"{'  ← 选择稳定' if len(uni) == 1 else ''}: {uni}")

        # ============ B. 封存样本，一次性开封 ============
        print("\n" + "=" * 76)
        print("B. 封存样本（一次性开封）：前 75% 选配置 → 后 25% 评一次")
        print("=" * 76)
        # 开封必须是**显式**的：另起一个 allow=True 的守卫，且访问会写进日志。
        # 上面的 g（allow=False）在整段 A 里都禁止触碰封存段。
        g_open = HoldoutGuard(n=n, fraction=0.25, allow=True,
                              log_path=os.path.join("runs",
                                                    "_holdout_access.log"))
        for s in syms:
            best = select_best(spots[s], perps[s], ft, s, 0, g.cut)
            if best is None:
                print(f"\n  {s}: 训练段无存活配置")
                continue
            (nr, m, rb), ins_ann, _ = best
            hs = g_open.holdout_slice()               # ← 显式开封
            r_h = CarrySimulator(spots[s].iloc[hs], perps[s].iloc[hs], ft,
                                 make_cfg(nr, m, rb), s).run()
            h8, h8i = rets8(r_h.equity, r_h.index)
            ev = protocol.evaluate_strategy(h8, bm8, n_trials=len(GRID),
                                            bars_per_year=BARS_PER_8H,
                                            r_index=h8i, b_index=bm8_idx)
            at = ev["alpha_test"]
            ci_lo = ev["alpha_bootstrap"]["annualized_ci_low"]
            ci_hi = ev["alpha_bootstrap"]["annualized_ci_high"]
            hold_rows.append({"symbol": s, "config": f"nr={nr},m={m},rb={rb}",
                              "in": ins_ann, "oos": r_h.annualized(),
                              "liquidated": bool(r_h.liquidated),
                              "alpha": at["alpha_annualized"],
                              "beta": at["beta"],
                              "t": at["t_stat"],
                              "p": at["p_value"],
                              "dsr": ev["dsr"]["dsr"],
                              "ci_low": ci_lo, "ci_high": ci_hi,
                              "verdict": ev["verdict"],
                              "n": at["n_obs"]})
            print(f"\n  {s}   选中配置 nr={nr}, m={m}, rb={rb}"
                  f"（样本内 {ins_ann*100:+.2f}%）")
            print(f"    封存段样本外年化 {r_h.annualized()*100:+.2f}%"
                  f"{'   ← 强平' if r_h.liquidated else ''}"
                  f"   8h 观测 {at['n_obs']:,}")
            print(f"    alpha 年化 {at['alpha_annualized']*100:+.2f}%   "
                  f"beta {at['beta']:+.4f}   t {at['t_stat']:.2f}   "
                  f"p {at['p_value']:.6f}   DSR {ev['dsr']['dsr']:.4f}")
            print(f"    自助 95% 区间 [{ci_lo*100:+.2f}%, {ci_hi*100:+.2f}%] 年化")
            print(f"    判定（以自助区间为主判据）：{ev['verdict']}"
                  f"（区间下界{'为正' if ci_lo > 0 else '未过零线'}）")

        print(f"\n开封记录：{g_open.accesses}")
        run.log("walkforward", wf_all)
        run.log("holdout", hold_rows)
        run.log("holdout_guard", {**g.summary(), "unseal": g_open.accesses})
        run.record_metrics({"walkforward": {
            s: {"mean_oos": float(np.mean([r["oos"] for r in v])) if v else None,
                "pos": int(sum(1 for r in v if r["oos"] > 0)), "n": len(v)}
            for s, v in wf_all.items()},
            "holdout": {r["symbol"]: {"oos": r["oos"], "alpha": r["alpha"],
                                      "ci_low": r["ci_low"],
                                      "ci_high": r["ci_high"],
                                      "verdict": r["verdict"]}
                        for r in hold_rows}})
        print(f"\nrun 目录: {run.dir}")


if __name__ == "__main__":
    main()
