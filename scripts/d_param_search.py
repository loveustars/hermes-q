"""D 阶段：参数重标定 —— 60 组 (η × band × funding) 真实数据搜索。

目的：
  - C 阶段已发现 Hedge 在 1h 时间尺度追逐价格信号，funding carry 被滤掉
  - B' baseline 用 η=0.05, band=0.05 几乎无效，需要重新选超参
  - 本脚本**不**跑评估协议（DSR/bootstrap），只看：sharpe / 总收益 / 最大回撤 /
    做空 bar 占比 / 终态 p_short_all
  - 6 × 5 × 2 = 60 组（η × band × funding on/off），真实数据 77k bar × 3 标的
  - 找出 top-10 配置，看 funding 接入是否在超参搜索下有边际

不做：
  - DSR / bootstrap（留给 D 最终评估阶段）
  - HoldoutGuard（C/B' baseline 都没用 holdout，D 也不引入）
  - Baseline 对照（m5_online.py 已有，C/B' 已对比 buy_hold）

预计耗时：~3-5 分钟（每组 ~3-5s，60 组）
"""
from __future__ import annotations

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src import config as cfgmod  # noqa: E402
from src.agents.online import HedgeEnsemble  # noqa: E402
from src.data import store  # noqa: E402
from src.eval import metrics  # noqa: E402
from src.eval.benchmark import equal_weight_buyhold  # noqa: E402
from src.sim.costs import CostModel  # noqa: E402
from src.sim.exchange import SimConfig, SimExchange  # noqa: E402
from src.sim.funding import FundingTable  # noqa: E402
from src.sim.margin import MarginConfig  # noqa: E402

BARS_PER_YEAR = 24 * 365
LEVERAGE = 3.0

# D 阶段：D1 阶段已跑 12 组（η=0.01/0.05/0.20 × band=0.03/0.10/0.20 × funding=N=9 + funding=Y 前 3 组）
# 关键发现：3x 杠杆下 12 组全部破产，参数搜索救不了 sim 失真
# D2 阶段：减杠杆到 1.0x（无 margin、无强平），看参数搜索在低杠杆下能否找到 alpha
LEVERAGE = 1.0    # D2 阶段：1x 杠杆（与 A 阶段 baseline 一致）
ETA_GRID = [0.01, 0.05, 0.20]
BAND_GRID = [0.03, 0.10, 0.20]
FUNDING_FLAGS = [False, True]    # 2 选 1


def run_one(frames, syms, rates, funding_or_none, eta, band,
            cfg, margin_cfg) -> dict:
    """跑一组超参，返回精简指标 + agent（agent 留 log 给落盘用）。"""
    ag = HedgeEnsemble(syms, cost_rate=rates, eta=eta, band=band,
                       max_exposure=LEVERAGE, funding=funding_or_none)
    res = SimExchange(frames, CostModel(enabled=False),
                      CostModel.from_config(cfg, enabled=True),
                      SimConfig(initial_cash=10_000.0, warmup=300,
                                max_gross=LEVERAGE,
                                max_exposure_per_symbol=LEVERAGE,
                                margin=margin_cfg,
                                allow_short=True,
                                instrument="perp")).run(ag)

    # 计算指标
    net_eq = res.net_equity
    if len(net_eq) < 2:
        return {"eta": eta, "band": band, "funding": funding_or_none is not None,
                "insolvent": res.insolvent, "sharpe": 0.0, "total_return": 0.0,
                "max_dd": 0.0, "do_short_pct": 0.0, "p_short_all": 0.0,
                "p_long_all": 0.0, "n_liquidated": 0, "n_trades": res.n_trades,
                "total_cost": float(res.cost_paid.sum())}

    rets = np.diff(net_eq) / net_eq[:-1]
    rets = rets[np.isfinite(rets)]
    sharpe = metrics.sharpe(rets, BARS_PER_YEAR) if len(rets) > 0 else 0.0
    total_ret = (net_eq[-1] / net_eq[0] - 1) if net_eq[0] > 0 else 0.0
    max_dd = metrics.max_drawdown(net_eq)

    # 做空 bar 占比
    mixed_log = ag.log["mixed_weight"]
    mixed = np.stack(mixed_log, axis=0) if len(mixed_log) > 1 else mixed_log[0][None, :]
    do_short_pct = float((mixed.min(axis=1) < -1e-6).mean())

    # 终态专家权重
    p = ag.p
    names = ag.expert_names()
    p_short_all = float(p[names.index("short_all")]) if "short_all" in names else 0.0
    p_long_all = float(p[names.index("long_all")]) if "long_all" in names else 0.0

    return {
        "eta": eta, "band": band,
        "funding": funding_or_none is not None,
        "insolvent": res.insolvent,
        "sharpe": round(sharpe, 4),
        "total_return": round(total_ret, 4),
        "max_dd": round(max_dd, 4),
        "do_short_pct": round(do_short_pct, 4),
        "p_short_all": round(p_short_all, 4),
        "p_long_all": round(p_long_all, 4),
        "n_liquidated": len(res.liquidated_legs),
        "n_trades": res.n_trades,
        "total_cost": round(float(res.cost_paid.sum()), 2),
        "final_net": round(float(net_eq[-1]), 2),
    }


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
    bh_final = 10_000 * (1 + bm.cumulative_return())
    print(f"基准（buy_hold）净终值  {bh_final:>14,.0f}\n")

    rates = np.array([0.0012, 0.0013, 0.0014])
    funding = FundingTable.load(
        os.path.join(store.project_root(), "data", "funding.csv"))
    print(f"funding 数据 {len(funding.rates_by_hour.get('BTCUSDT', {}))} 条\n")

    margin_cfg = MarginConfig(
        initial_margin_ratio=0.5,
        maintenance_margin_ratio=0.1,
        topup_trigger_ratio=0.5,
    )

    # 网格扫描
    n_total = len(ETA_GRID) * len(BAND_GRID) * len(FUNDING_FLAGS)
    print(f"参数网格：η × band × funding = "
          f"{len(ETA_GRID)} × {len(BAND_GRID)} × {len(FUNDING_FLAGS)} = {n_total} 组\n")
    print(f"{'η':>7} {'band':>5} {'fund':>5} "
          f"{'sharpe':>8} {'total_ret':>11} {'max_dd':>8} "
          f"{'short%':>7} {'p_short':>8} {'p_long':>8} "
          f"{'liq':>4} {'n_trd':>6} {'final_net':>12} {'sec':>5}")

    rows = []
    t0 = time.time()
    for funding_on in FUNDING_FLAGS:
        f_obj = funding if funding_on else None
        for eta in ETA_GRID:
            for band in BAND_GRID:
                row = run_one(fr, syms, rates, f_obj, eta, band, cfg, margin_cfg)
                row["seconds"] = round(time.time() - t0, 1)
                rows.append(row)
                print(f"{row['eta']:>7.3f} {row['band']:>5.2f} "
                      f"{'Y' if row['funding'] else 'N':>5} "
                      f"{row['sharpe']:>8.3f} {row['total_return']*100:>10.1f}% "
                      f"{row['max_dd']*100:>7.1f}% "
                      f"{row['do_short_pct']*100:>6.1f}% "
                      f"{row['p_short_all']:>8.3f} {row['p_long_all']:>8.3f} "
                      f"{row['n_liquidated']:>4} {row['n_trades']:>6} "
                      f"{row['final_net']:>12,.0f} {row['seconds']:>5.0f}")
    total_seconds = time.time() - t0

    # 按 sharpe 排序
    rows.sort(key=lambda r: -r["sharpe"])
    print(f"\n总耗时 {total_seconds:.1f}s   benchmark 净终值 {bh_final:,.0f}\n")
    print("=== top-10 by sharpe ===")
    for r in rows[:10]:
        marker = "💰" if r["funding"] else "  "
        print(f"  {marker} η={r['eta']:.3f} band={r['band']:.2f} "
              f"funding={'Y' if r['funding'] else 'N'}  "
              f"sharpe={r['sharpe']:>6.3f}  total={r['total_return']*100:>7.1f}%  "
              f"short%={r['do_short_pct']*100:>4.1f}  p_short={r['p_short_all']:.3f}")

    # 按 total_return 排序
    rows.sort(key=lambda r: -r["total_return"])
    print("\n=== top-10 by total_return ===")
    for r in rows[:10]:
        marker = "💰" if r["funding"] else "  "
        print(f"  {marker} η={r['eta']:.3f} band={r['band']:.2f} "
              f"funding={'Y' if r['funding'] else 'N'}  "
              f"sharpe={r['sharpe']:>6.3f}  total={r['total_return']*100:>7.1f}%  "
              f"short%={r['do_short_pct']*100:>4.1f}  p_short={r['p_short_all']:.3f}")

    # funding 接入 vs 不接入的对比
    print("\n=== funding 接入 vs 不接入（按 sharpe 平均）===")
    with_f = [r for r in rows if r["funding"]]
    no_f = [r for r in rows if not r["funding"]]
    print(f"  funding=Y:  mean sharpe {np.mean([r['sharpe'] for r in with_f]):>6.3f}  "
          f"mean total {np.mean([r['total_return'] for r in with_f])*100:>6.1f}%  "
          f"mean short% {np.mean([r['do_short_pct'] for r in with_f])*100:>5.1f}%")
    print(f"  funding=N:  mean sharpe {np.mean([r['sharpe'] for r in no_f]):>6.3f}  "
          f"mean total {np.mean([r['total_return'] for r in no_f])*100:>6.1f}%  "
          f"mean short% {np.mean([r['do_short_pct'] for r in no_f])*100:>5.1f}%")

    # 落盘
    out_dir = os.path.join(store.project_root(), "runs", "d_param_search")
    os.makedirs(out_dir, exist_ok=True)
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, "all_configs.csv"), index=False)
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump({
            "n_configs": n_total,
            "total_seconds": round(total_seconds, 1),
            "bh_final": round(bh_final, 2),
            "best_by_sharpe": rows[0] if not rows else None,
            "funding_comparison": {
                "with_funding": {
                    "n": len(with_f),
                    "mean_sharpe": float(np.mean([r['sharpe'] for r in with_f])) if with_f else 0,
                    "mean_total": float(np.mean([r['total_return'] for r in with_f])) if with_f else 0,
                },
                "no_funding": {
                    "n": len(no_f),
                    "mean_sharpe": float(np.mean([r['sharpe'] for r in no_f])) if no_f else 0,
                    "mean_total": float(np.mean([r['total_return'] for r in no_f])) if no_f else 0,
                },
            },
        }, f, indent=2, ensure_ascii=False)
    print(f"\n落盘到 {os.path.relpath(out_dir, store.project_root())}")


if __name__ == "__main__":
    main()
