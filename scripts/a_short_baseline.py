"""A 阶段训练目标基线 —— 在真实数据上快速验证"放开做空"有没有用。

目的：
  - 复用 m5_online.py 的评估口径（训练段、等权买入持有、扣成本）
  - 单跑 1 组中等超参（η=0.05, band=0.05，避开 9 组搜索），目标是把时间从
    分钟级压到秒级，**专门盯"做空是否活跃"这件事**
  - 不更新 run registry（这是 diagnostic，不是新策略）

判断 A 是否成功：
  - 在下跌段 / 横盘段，Hedge 终态 `p_short_all` 显著高于 1/K（说明学到了做空）
  - 净终值不能比 buy_hold 差到不可接受（A 阶段用 max_gross=1.0，理论下限 = buy_hold）
  - 做空活跃度（mixed_weight 中 |w_min| 的均值）显著 > 0

不做：
  - DSR / 多重试验惩罚（A 阶段目标是验证训练目标"有能力做空"，不是评估 alpha）
  - 超参搜索（避免搜索成本 + 与 M5 混淆）
  - 封存段评估（holdout 的目的是测 alpha，不是测做空能力）
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src import config as cfgmod  # noqa: E402
from src.agents.online import HedgeEnsemble  # noqa: E402
from src.data import store  # noqa: E402
from src.eval.benchmark import equal_weight_buyhold, strategy_returns  # noqa: E402
from src.sim.costs import CostModel  # noqa: E402
from src.sim.exchange import SimConfig, SimExchange  # noqa: E402

BARS_PER_YEAR = 24 * 365


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
    print(f"窗口 {n:,} 根  "
          f"{fr[syms[0]].index[0]:%Y-%m-%d} ~ {fr[syms[0]].index[-1]:%Y-%m-%d}")

    bm = equal_weight_buyhold(fr, cfg)
    bench_ret = bm.returns
    print(f"基准（等权买入持有，零成本）累计收益 {bm.cumulative_return()*100:,.1f}%\n")

    rates = np.array([0.0012, 0.0013, 0.0014])

    # 单跑 1 套超参
    eta, band = 0.05, 0.05
    ag = HedgeEnsemble(syms, cost_rate=rates, eta=eta, band=band)

    res = SimExchange(fr, CostModel(enabled=False),
                      CostModel.from_config(cfg, enabled=True),
                      SimConfig(initial_cash=10_000.0, warmup=300)).run(ag)

    print("=== A 阶段基线（HedgeEnsemble × 14 专家，含 3 做空）===")
    print(f"η={eta}  band={band}  专家数 {ag.K}")
    print(f"成交笔数 {res.n_trades:,}   归零={res.insolvent}")
    print(f"毛终值 {res.final_gross():>12,.0f}   净终值 {res.final_net():>12,.0f}"
          f"   累计成本 {res.cost_paid.sum():>10,.0f}")
    print(f"换手名义额 {res.turnover_notional.sum():>14,.0f}"
          f"  = 本金 {res.turnover_notional.sum()/10000:,.1f} 倍\n")

    print(f"基准（buy_hold）净终值  {10_000*(1+bm.cumulative_return()):>12,.0f}")
    print(f"A agent  净终值         {res.final_net():>12,.0f}"
          f"   alpha {(res.final_net()/(10_000*(1+bm.cumulative_return())) - 1)*100:>+8.2f}%\n")

    # ---------- A 关键指标：做空活跃度 ----------
    mixed_log = ag.log["mixed_weight"]
    mixed = np.stack(mixed_log, axis=0) if len(mixed_log) > 1 else mixed_log[0][None, :]
    n_bars = mixed.shape[0]

    short_active_bars = int((mixed.min(axis=1) < -1e-6).sum())
    print(f"=== 做空活跃度（{n_bars:,} 根 bar）===")
    print(f"  mixed_weight 中 min(w) < -1e-6 的 bar：{short_active_bars:>5}"
          f"  ({short_active_bars/n_bars*100:.1f}%)")
    print(f"  min(w) 历史最低值：{mixed.min():>+.4f}")
    print(f"  min(w) 历史均值（仅做空 bar）："
          f"{mixed[mixed < -1e-6].min() if short_active_bars else 'N/A':}\n")

    # ---------- 专家终态 ----------
    print("=== 专家权重终态（按 p 降序）===")
    for k, v in sorted(zip(ag.expert_names(), ag.p), key=lambda kv: -kv[1]):
        marker = "★" if v > 2.0 / ag.K else " "
        bar = "█" * max(1, int(v * 50))
        print(f"  {marker} {k:<16}{v:>7.4f}  {bar}")

    # ---------- 分段：下跌段做空应该更活跃 ----------
    # 把 mixed_weight 历史切成 5 段，看 short_active_bars 在各段的占比
    print("\n=== 分段做空活跃度（按时间分 5 段）===")
    rt = np.diff(np.log(fr[syms[0]]["close"].to_numpy()))
    seg_len = len(mixed) // 5
    for i in range(5):
        seg_mixed = mixed[i * seg_len:(i + 1) * seg_len]
        if len(seg_mixed) == 0:
            continue
        # align 价格段：mixed 的 bar i 对应 close[i+1]，所以价格段从 (i*seg_len)+1 起
        ps = i * seg_len + 1
        pe = (i + 1) * seg_len + 1
        ret = float(np.exp(rt[ps:pe].sum()) - 1)
        active = int((seg_mixed.min(axis=1) < -1e-6).sum())
        print(f"  段 {i + 1}/5  bar {i*seg_len:>5}~{(i+1)*seg_len:<5}  "
              f"区间收益 {ret*100:+7.2f}%  做空 bar {active:>4}/{len(seg_mixed)} "
              f"({active/len(seg_mixed)*100:>5.1f}%)")

    # ---------- 落盘 ----------
    out_dir = os.path.join(store.project_root(), "runs", "a_short_baseline", "curves")
    os.makedirs(out_dir, exist_ok=True)
    pd.DataFrame({
        "dt": res.index, "net": res.net_equity, "gross": res.gross_equity,
        "cost": res.cost_paid, "turnover": res.turnover_notional,
    }).to_csv(os.path.join(out_dir, "hedge.csv"), index=False)
    pd.DataFrame(ag.expert_weight_history(),
                 columns=ag.expert_names()).to_csv(
        os.path.join(out_dir, "expert_weights.csv"), index=False)
    pd.DataFrame({"step": ag.log["step"],
                  "min_weight": [m.min() for m in mixed_log],
                  "max_weight": [m.max() for m in mixed_log],
                  "turnover": ag.log["turnover"],
                  "entropy": ag.log["entropy"]}).to_csv(
        os.path.join(out_dir, "learning_curve.csv"), index=False)
    print(f"\n落盘到 {os.path.relpath(out_dir, store.project_root())}")


if __name__ == "__main__":
    main()
