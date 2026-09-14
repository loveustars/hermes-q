"""M10-2 carry 策略网格 —— 用真实数据跑完整参数网格并做正式评估。

网格维度：
  notional_ratio       每条腿的名义额 / 资本
  initial_margin_ratio 初始保证金 = m × 名义额
  rebalance_every      再平衡间隔（同时是成本参数与风控参数）
约束：notional_ratio × (1 + m) ≤ 1（否则没有备用金）

评估：carry 是 delta 中性策略，所以正确的检验是
  「剥离市场暴露后，alpha 是否显著为正」——
  beta 应该接近 0（验证方向中性），alpha 才是 carry 的真实收益。
基准用等权买入持有（与 m4/m5 同一套构造）。
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
NOTIONAL_GRID = [0.2, 0.3, 0.4]
MARGIN_GRID = [0.5, 1.0, 2.0]
REBALANCE_GRID = [24, 168, 720]


def load_frames(symbols, source="binance"):
    frames = {s: store.load_bars(s, "1h", source=source)[
        ["open", "high", "low", "close", "volume", "quote_volume"]] for s in symbols}
    idx = None
    for f in frames.values():
        idx = f.index if idx is None else idx.intersection(f.index)
    return {s: f.loc[idx] for s, f in frames.items()}


def main() -> None:
    cfg = cfgmod.load("base")
    cfgmod.guard_frozen(cfg)
    syms = cfg["universe"]["core"]
    ft = load_default()
    if not ft.rates_by_hour:
        raise SystemExit("缺资金费数据，先跑 scripts/m7_ingest_funding.py")

    spots = load_frames(syms, "binance")
    perps = load_frames(syms, "binanceperp")

    # 用现货数据构造基准（与 m4/m5 一致）；对齐到永续公共窗口
    fund_start = pd.Timestamp(max(min(d) for d in ft.rates_by_hour.values() if d),
                              unit="ms", tz="UTC")
    bench_frames = {s: spots[s].loc[spots[s].index >= fund_start] for s in syms}
    bm = equal_weight_buyhold(bench_frames, cfg)
    print(f"基准（等权买入持有）{bm.index[0]:%Y-%m-%d} ~ {bm.index[-1]:%Y-%m-%d}  "
          f"{len(bm.index):,} 根  累计 {bm.cumulative_return()*100:,.1f}%")

    combos = [(nr, m) for nr in NOTIONAL_GRID for m in MARGIN_GRID
              if nr * (1 + m) <= 1.0 + 1e-9]
    print(f"有效参数组合 {len(combos)} 组 × 再平衡 {len(REBALANCE_GRID)} 档 × "
          f"{len(syms)} 标的 = {len(combos)*len(REBALANCE_GRID)*len(syms)} 次仿真")

    hypothesis = {
        "question": "把 carry 放进显式建模保证金与强平的仿真器后，"
                    "净收益是多少？能否通过 G1 校准过的评估协议？",
        "expected": "存活配置的年化应在 M9 理论区间 3.1%~5.8% 内；"
                    "beta 接近 0（方向中性）；alpha 大概率不显著（扣掉真实成本后）",
        "decision_rule": "若存活配置的 alpha 显著为正且 beta≈0，则记为有边际；"
                         "否则如实记为无边际，并指出是哪一项成本吃掉了收益",
    }

    with Run("m10_carry_grid", cfg, hypothesis) as run:
        rows = []
        print(f"\n{'标的':<9}{'nr':>5}{'m':>5}{'再平衡':>7}{'终值':>12}"
              f"{'年化':>9}{'强平':>6}{'补保':>6}{'资金费':>10}"
              f"{'费用':>9}{'价格损益':>10}")
        for s in syms:
            for nr, m in combos:
                for rb in REBALANCE_GRID:
                    c = CarryConfig(initial_capital=10_000.0, notional_ratio=nr,
                                    initial_margin_ratio=m, rebalance_every=rb,
                                    maintenance_margin_ratio=0.005,
                                    topup_trigger_ratio=0.5, warmup=300)
                    res = CarrySimulator(spots[s], perps[s], ft, c, s).run()
                    rows.append({
                        "symbol": s, "notional_ratio": nr, "margin_ratio": m,
                        "rebalance_every": rb,
                        "final_equity": round(res.final(), 2),
                        "annualized": round(res.annualized(), 6),
                        "total_return": round(res.total_return(), 6),
                        "liquidated": res.liquidated,
                        "liquidated_at": res.liquidated_at,
                        "topups": res.n_topups,
                        "topup_amount": round(res.topup_amount, 2),
                        "funding_income": round(res.total_funding(), 2),
                        "fees": round(res.total_fees(), 2),
                        "price_pnl": round(res.total_price_pnl(), 2),
                        "bars": int(len(res.equity)),
                    })
                    flag = "★" if res.liquidated else " "
                    print(f"{s:<9}{nr:>5.1f}{m:>5.1f}{rb:>7}"
                          f"{res.final():>12,.0f}{res.annualized()*100:>8.2f}%"
                          f"{flag:>6}{res.n_topups:>6}{res.total_funding():>10,.0f}"
                          f"{res.total_fees():>9,.0f}{res.total_price_pnl():>10,.0f}")
        run.log("grid", rows)

        # ---------- 汇总 ----------
        alive = [r for r in rows if not r["liquidated"]]
        dead = [r for r in rows if r["liquidated"]]
        print(f"\n=== 网格汇总 ===")
        print(f"  存活 {len(alive)} / {len(rows)}，强平 {len(dead)}")
        if dead:
            by_sym = {}
            for r in dead:
                by_sym[r["symbol"]] = by_sym.get(r["symbol"], 0) + 1
            print(f"  强平分布：{by_sym}")
        if alive:
            best = max(alive, key=lambda r: r["annualized"])
            print(f"  最优存活配置：{best['symbol']} nr={best['notional_ratio']} "
                  f"m={best['margin_ratio']} 再平衡={best['rebalance_every']} "
                  f"→ 年化 {best['annualized']*100:.2f}%  终值 {best['final_equity']:,.0f}")
            med = float(np.median([r["annualized"] for r in alive]))
            print(f"  存活配置年化中位数 {med*100:.2f}%   "
                  f"区间 [{min(r['annualized'] for r in alive)*100:.2f}%, "
                  f"{max(r['annualized'] for r in alive)*100:.2f}%]")

        # ---------- 对最优配置做正式评估 ----------
        print(f"\n=== 最优存活配置的正式评估 ===")
        print("  carry 是 delta 中性策略，所以检验的是「剥离市场暴露后 alpha 是否显著为正」，")
        print("  同时看 beta 是否接近 0 —— 那是方向中性的验证。")
        evals = []
        for s in syms:
            cand = [r for r in alive if r["symbol"] == s]
            if not cand:
                print(f"  {s}: 无可存活配置")
                continue
            b = max(cand, key=lambda r: r["annualized"])
            c = CarryConfig(initial_capital=10_000.0,
                            notional_ratio=b["notional_ratio"],
                            initial_margin_ratio=b["margin_ratio"],
                            rebalance_every=b["rebalance_every"], warmup=300)
            res = CarrySimulator(spots[s], perps[s], ft, c, s).run()
            eq = res.equity
            r = np.diff(eq) / eq[:-1]
            r = np.where(np.isfinite(r), r, 0.0)
            idx = res.index[1:len(r) + 1]
            ev = protocol.evaluate_strategy(r, bm.returns, n_trials=len(rows),
                                            bars_per_year=BARS_PER_YEAR,
                                            r_index=idx, b_index=bm.index)
            at = ev["alpha_test"]
            print(f"\n  {s}  nr={b['notional_ratio']} m={b['margin_ratio']} "
                  f"再平衡={b['rebalance_every']}  年化 {b['annualized']*100:.2f}%")
            print(f"    alpha 年化 {at['alpha_annualized']*100:>7.2f}%   "
                  f"beta {at['beta']:>6.3f}   t = {at['t_stat']:>6.2f}   "
                  f"p = {at['p_value']:.4f}")
            print(f"    DSR = {ev['dsr']['dsr']:.4f}   "
                  f"alpha bootstrap 区间 [{ev['alpha_bootstrap']['ci_low']*BARS_PER_YEAR*100:.1f}%, "
                  f"{ev['alpha_bootstrap']['ci_high']*BARS_PER_YEAR*100:.1f}%] 年化")
            print(f"    R² = {at['r_squared']:.4f}（接近 0 即确认方向中性）")
            print(f"    判定：{ev['verdict']}")
            evals.append({"symbol": s, "config": b, "alpha_test": at,
                          "dsr": ev["dsr"], "verdict": ev["verdict"],
                          "dsr_value": ev["dsr"]["dsr"]})

        run.log("evaluation", evals)
        run.record_metrics({
            "n_configs": len(rows), "n_alive": len(alive), "n_liquidated": len(dead),
            "alive_annualized_median": round(
                float(np.median([r["annualized"] for r in alive])), 6) if alive else None,
            "verdicts": {e["symbol"]: e["verdict"] for e in evals},
            "betas": {e["symbol"]: e["alpha_test"]["beta"] for e in evals},
        })
        print(f"\nrun 目录: {os.path.relpath(run.dir, store.project_root())}")


if __name__ == "__main__":
    main()
