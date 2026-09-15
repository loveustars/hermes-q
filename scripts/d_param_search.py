"""D 阶段：参数重标定 —— 18 组 (η × band × funding) 真实数据搜索。

目的：
  - C 阶段已发现 Hedge 在 1h 时间尺度追逐价格信号，funding carry 被滤掉
  - B' baseline 用 η=0.05, band=0.05 几乎无效，需要重新选超参
  - 本脚本**不**跑评估协议（DSR/bootstrap），只看：sharpe / 总收益 / 最大回撤 /
    做空 bar 占比 / 终态 p_short_all
  - 3 × 3 × 2 = 18 组（η × band × funding on/off），真实数据 77k bar × 3 标的
  - 找出 top-10 配置，看 funding 接入是否在超参搜索下有边际

不做：
  - DSR / bootstrap（留给 D 最终评估阶段）
  - HoldoutGuard（C/B' baseline 都没用 holdout，D 也不引入）
  - Baseline 对照（m5_online.py 已有，C/B' 已对比 buy_hold）

**杠杆**：用环境变量 `D_LEVERAGE`（默认 1.0）、输出目录用 `D_OUT`
（默认 `d_param_search`），因为 1x 与 3x 都要跑而 3x 的结论在
2026-09-15 的记账修复后必须重新验证。见 WORK_LOG §14 / §15。

耗时（2026-09-15 实测，i9-13900H）：**每组约 72 秒，18 组约 21.5 分钟**
（此前本文件写"~3-5 分钟 / 每组 3-5s"是**错的**；实测每组 72s）。
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

# 运行期开关（2026-09-15 加）：让"重跑 D"能跑不同杠杆而不必复制脚本。
#   D_LEVERAGE=3.0 D_OUT=d_param_search_3x python3 scripts/d_param_search.py
# 默认值与初版一致（1.0 / d_param_search），所以旧调用方式行为不变。
LEVERAGE = float(os.environ.get("D_LEVERAGE", "1.0"))
OUT_NAME = os.environ.get("D_OUT", "d_param_search")
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
                "insolvent": res.insolvent, "bankrupt": res.bankrupt,
                "bankrupt_equity_raw": None,
                "sharpe": 0.0, "total_return": 0.0,
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
        # 已爆仓：曲线截断在归零那一根，末期收益为 −100%。
        # **这类配置的 sharpe 没有意义**（见 main() 里的排名处理）。
        "bankrupt": res.bankrupt,
        "bankrupt_equity_raw": (round(res.bankrupt_equity_raw, 2)
                                if res.bankrupt_equity_raw is not None else None),
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

    # ---- 排名：已爆仓的配置**不参与** ----
    # 爆仓组的曲线被截断在归零那一根、末期收益恰为 −100%，其 sharpe 在数学上
    # 失去意义：简单收益率的**均值**可以为正，而复利终值已经归零，于是会得出
    # "已爆仓组 sharpe 最高"这种荒谬排名（实测出现过 sharpe=2.625 的爆仓组）。
    # 它们单独列出，只报"死时原始权益 / 换手 / 强平次数"。
    dead = [r for r in rows if r["bankrupt"]]
    alive = [r for r in rows if not r["bankrupt"]]

    alive.sort(key=lambda r: -r["sharpe"])
    best_by_sharpe = alive[0] if alive else None
    print(f"\n总耗时 {total_seconds:.1f}s   benchmark 净终值 {bh_final:,.0f}")
    print(f"存活 {len(alive)}/{len(rows)} 组，爆仓 {len(dead)} 组\n")
    print("=== top-10 by sharpe（仅存活组）===")
    for r in alive[:10]:
        marker = "💰" if r["funding"] else "  "
        print(f"  {marker} η={r['eta']:.3f} band={r['band']:.2f} "
              f"funding={'Y' if r['funding'] else 'N'}  "
              f"sharpe={r['sharpe']:>6.3f}  total={r['total_return']*100:>7.1f}%  "
              f"short%={r['do_short_pct']*100:>4.1f}  p_short={r['p_short_all']:.3f}")

    if dead:
        print(f"\n=== 已爆仓（{len(dead)} 组，不参与排名）===")
        for r in sorted(dead, key=lambda r: r["eta"]):
            print(f"     η={r['eta']:.3f} band={r['band']:.2f} "
                  f"funding={'Y' if r['funding'] else 'N'}  "
                  f"死时原始权益={r['bankrupt_equity_raw']:>12,.0f}  "
                  f"n_trd={r['n_trades']:>7}  liq={r['n_liquidated']:>4}")

    # 按 total_return 排序（同样只用存活组）
    alive.sort(key=lambda r: -r["total_return"])
    best_by_total = alive[0] if alive else None
    print("\n=== top-10 by total_return（仅存活组）===")
    for r in alive[:10]:
        marker = "💰" if r["funding"] else "  "
        print(f"  {marker} η={r['eta']:.3f} band={r['band']:.2f} "
              f"funding={'Y' if r['funding'] else 'N'}  "
              f"sharpe={r['sharpe']:>6.3f}  total={r['total_return']*100:>7.1f}%  "
              f"short%={r['do_short_pct']*100:>4.1f}  p_short={r['p_short_all']:.3f}")

    # funding 接入 vs 不接入的对比
    print("\n=== funding 接入 vs 不接入（按 sharpe 平均，仅存活组）===")
    with_f = [r for r in alive if r["funding"]]
    no_f = [r for r in alive if not r["funding"]]
    if not alive:
        print("  存活 0 组 —— 全部爆仓，本对比无意义（已跳过）")
    else:
        print(f"  仅存活组参与：爆仓组的 sharpe 在数学上无意义"
              f"（简单收益率均值可正、复利终值已归零），纳入会把均值污染成假信号。")
        print(f"  funding=Y:  mean sharpe {np.mean([r['sharpe'] for r in with_f]):>6.3f}  "
              f"mean total {np.mean([r['total_return'] for r in with_f])*100:>6.1f}%  "
              f"mean short% {np.mean([r['do_short_pct'] for r in with_f])*100:>5.1f}%"
              if with_f else "  funding=Y:  存活 0 组")
        print(f"  funding=N:  mean sharpe {np.mean([r['sharpe'] for r in no_f]):>6.3f}  "
              f"mean total {np.mean([r['total_return'] for r in no_f])*100:>6.1f}%  "
              f"mean short% {np.mean([r['do_short_pct'] for r in no_f])*100:>5.1f}%"
              if no_f else "  funding=N:  存活 0 组")

    # 落盘
    out_dir = os.path.join(store.project_root(), "runs", OUT_NAME)
    os.makedirs(out_dir, exist_ok=True)
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, "all_configs.csv"), index=False)
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump({
            "n_configs": n_total,
            "n_alive": len(alive),
            "n_bankrupt": len(dead),
            "total_seconds": round(total_seconds, 1),
            "bh_final": round(bh_final, 2),
            # 注意：这两个字段来自**仅存活组**的排名（爆仓组的 sharpe 无意义，
            # 见 main() 里的说明）。best_by_sharpe 必须在按 sharpe 排序之后、
            # 按 total_return 重排之前取 —— 初版写成 `rows[0] if not rows else None`
            # （条件写反）导致该字段恒为 null。
            "best_by_sharpe": best_by_sharpe,
            "best_by_total_return": best_by_total,
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
