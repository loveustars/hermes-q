"""M5 在线自学习体评测 —— 每个 bar、每笔交易都进入学习。

评估纪律：
  - 最后 25% 的数据封存（HoldoutGuard 代码级拒绝访问），本脚本只用前 75%
  - 在线学习体的整段运行在信息意义上是**序贯样本外**的（只用过去）
  - 真正的多重试验风险来自超参数搜索（η × 带宽），所以 DSR 的试验次数
    按"本次搜索过的配置数"计，并同时报告按全项目累计试验数惩罚的版本
  - 分段报告：前 1/2 与后 1/2 的表现对比 —— 这是"持续学习到底有没有用"的检验

对照基线：等权买入持有、单持 BTC、逆波动率满仓、固定动量、随机换仓。
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src import config as cfgmod  # noqa: E402
from src.agents import baselines as B  # noqa: E402
from src.agents.online import HedgeEnsemble, InverseVolExpert  # noqa: E402
from src.data import store  # noqa: E402
from src.eval import metrics, protocol  # noqa: E402
from src.eval.benchmark import equal_weight_buyhold, strategy_returns  # noqa: E402
from src.eval.holdout import HoldoutGuard  # noqa: E402
from src.registry.runs import Run, trial_count  # noqa: E402
from src.sim.costs import CostModel  # noqa: E402
from src.sim.exchange import SimConfig, SimExchange  # noqa: E402

BARS_PER_YEAR = 24 * 365
ETA_GRID = [0.02, 0.05, 0.10]
BAND_GRID = [0.02, 0.05, 0.10]


def load_frames(symbols: list[str]) -> dict[str, pd.DataFrame]:
    frames = {s: store.load_bars(s, "1h")[
        ["open", "high", "low", "close", "volume", "quote_volume"]] for s in symbols}
    idx = None
    for f in frames.values():
        idx = f.index if idx is None else idx.intersection(f.index)
    return {s: f.loc[idx] for s, f in frames.items()}


def benchmark_returns(frames, cfg):
    """等权买入持有基准（同一撮合器、零成本）—— 与 m4 校准用的是同一套构造。"""
    bm = equal_weight_buyhold(frames, cfg)
    return bm.returns, bm.index, bm


def per_asset_cost_rate(frames, equity=10_000.0, sigma_window=168) -> np.ndarray:
    out = []
    for s, f in frames.items():
        sigma = float(np.log(f["close"] / f["close"].shift()).dropna().std())
        v = float(f["quote_volume"].median())
        part = equity * 0.1 / v if v > 0 else 0.0
        out.append(0.001 + 0.0001 + sigma * np.sqrt(max(part, 0.0)))
    return np.array(out)


def run_agent(frames, agent, cfg, costs_on=True) -> dict:
    cm_net = CostModel.from_config(cfg, enabled=costs_on)
    simcfg = SimConfig(initial_cash=10_000.0, warmup=300,
                       latency_bars=cfg["costs"]["latency_bars"])
    res = SimExchange(frames, CostModel.from_config(cfg, enabled=False),
                      cm_net, simcfg).run(agent)
    return {"res": res,
            "gross": metrics.summary(res.gross_equity, res.turnover_notional, None,
                                     BARS_PER_YEAR, initial=res.initial_cash),
            "net": metrics.summary(res.net_equity, res.turnover_notional,
                                   res.cost_paid, BARS_PER_YEAR,
                                   initial=res.initial_cash)}


def split_report(res, frac: float = 0.5) -> dict:
    """把净值曲线切成两段，看学习是在进步还是在退化。"""
    e = res.net_equity
    k = int(len(e) * frac)
    out = {}
    for name, seg in [("first_half", e[:k]), ("second_half", e[k - 1:])]:
        if len(seg) > 2:
            out[name] = {
                "bars": int(len(seg)),
                "total_return_pct": round((seg[-1] / seg[0] - 1) * 100, 2),
                "sharpe": metrics.sharpe(np.diff(seg) / seg[:-1], BARS_PER_YEAR),
                "max_drawdown_pct": round(metrics.max_drawdown(seg) * 100, 2),
            }
    return out


def main() -> None:
    cfg = cfgmod.load("base")
    cfgmod.guard_frozen(cfg)
    syms = cfg["universe"]["core"]
    frames_all = load_frames(syms)
    n_all = len(frames_all[syms[0]])

    # ---- 铁律 4：封存最后 25% ----
    guard = HoldoutGuard(n=n_all, fraction=cfg["eval"]["holdout_fraction"],
                         allow=False,
                         log_path=os.path.join(store.project_root(), "runs",
                                               "holdout_access.log"))
    frames = {s: f.iloc[guard.train_slice()] for s, f in frames_all.items()}
    n = len(frames[syms[0]])
    print(f"全量 {n_all:,} 根  |  训练段 {n:,} 根（封存 {guard.n - guard.cut:,} 根）")
    print(f"训练段 {frames[syms[0]].index[0]:%Y-%m-%d} ~ {frames[syms[0]].index[-1]:%Y-%m-%d}")
    print(f"封存段从 {frames_all[syms[0]].index[guard.cut]:%Y-%m-%d} 开始，本脚本禁止访问")

    bench, bench_index, bm = benchmark_returns(frames, cfg)
    bench_ret = bench
    rates = per_asset_cost_rate(frames)
    print(f"基准（等权买入持有）累计收益 {bm.cumulative_return()*100:,.1f}%")
    print(f"单位换手成本率：{dict(zip(syms, np.round(rates*1e4, 2)))} (bp)")

    n_configs = len(ETA_GRID) * len(BAND_GRID)
    n_baselines = 5
    n_trials_this = n_configs + n_baselines
    total_trials = trial_count() + n_trials_this

    hypothesis = {
        "question": "在线学习体（每 bar 更新、含成本信号）能否在扣除成本后跑出"
                    "相对市场暴露的显著 alpha？持续学习会不会随时间退化？",
        "expected": "大概率无显著 alpha；成本会压制换手；分段表现应大致持平而非发散",
        "decision_rule": "若净口径 alpha 不显著，则结论为'无边际'，如实记录并转信号源研究；"
                         "若分段退化为负，说明在线更新在追逐噪声，需下调 η",
    }

    with Run("m5_online_learning", cfg, hypothesis) as run:
        # ---------- 1. 基线 ----------
        print("\n=== 1. 对照基线（训练段，净口径）===")
        baseline_agents = [
            B.SingleAssetBuyHold("BTCUSDT"),
            B.BuyHold(),
            B.Momentum("BTCUSDT", lookback=720, rebalance_every=24),
            B.RandomWeights(seed=7, rebalance_every=24),
            B.Cash(),
        ]
        rows = []
        print(f"{'策略':<24}{'净终值':>13}{'净收益':>11}{'Sharpe':>9}"
              f"{'换手/年':>9}{'成本%':>8}")
        base_series = {}
        for ag in baseline_agents:
            out = run_agent(frames, ag, cfg, costs_on=True)
            r = out["net"]
            base_series[ag.name] = out["res"].net_equity
            rows.append({"agent": ag.name, "net": r})
            print(f"{ag.name:<24}{r['final_equity']:>13,.0f}"
                  f"{r['total_return']*100:>10.1f}%{r['sharpe']:>9.2f}"
                  f"{r.get('turnover_roundtrips_per_year', 0):>9.1f}"
                  f"{r.get('cost_as_pct_of_initial', 0):>7.1f}%")

        # ---------- 2. 在线学习体超参搜索 ----------
        print(f"\n=== 2. 在线学习体（{n_configs} 组超参，η × 带宽）===")
        print(f"{'配置':<30}{'净终值':>13}{'净收益':>11}{'Sharpe':>9}"
              f"{'换手/年':>9}{'成本%':>8}{'专家熵':>8}")
        configs = []
        best = None
        for eta in ETA_GRID:
            for band in BAND_GRID:
                ag = HedgeEnsemble(syms, cost_rate=rates, eta=eta, band=band)
                out = run_agent(frames, ag, cfg, costs_on=True)
                r = out["net"]
                hist = ag.expert_weight_history()
                ent = float(np.mean(ag.log["entropy"])) if ag.log["entropy"] else 0.0
                rec = {"eta": eta, "band": band, "name": ag.name,
                       "net": r, "insolvent": out["res"].insolvent,
                       "mean_entropy": round(ent, 4),
                       "n_experts": ag.K,
                       "final_expert_weights": dict(zip(ag.expert_names(),
                                                        np.round(ag.p, 4).tolist()))}
                configs.append(rec)
                print(f"η={eta:<5} band={band:<5}{'':<14}{r['final_equity']:>13,.0f}"
                      f"{r['total_return']*100:>10.1f}%{r['sharpe']:>9.2f}"
                      f"{r.get('turnover_roundtrips_per_year', 0):>9.1f}"
                      f"{r.get('cost_as_pct_of_initial', 0):>7.1f}%{ent:>8.3f}")
                if best is None or r["final_equity"] > best["out"]["net"]["final_equity"]:
                    best = {"rec": rec, "out": out, "agent": ag}
        run.log("hyperparameter_search", configs)

        # ---------- 3. 最优配置的完整评估 ----------
        best_ag = best["agent"]
        best_res = best["out"]["res"]
        net_r, net_idx = strategy_returns(best_res)

        print(f"\n=== 3. 最优配置评估：η={best['rec']['eta']} band={best['rec']['band']} ===")
        ev = protocol.evaluate_strategy(net_r, bench_ret, n_trials=n_trials_this,
                                        bars_per_year=BARS_PER_YEAR,
                                        r_index=net_idx, b_index=bench_index)
        ev_total = protocol.evaluate_strategy(net_r, bench_ret, n_trials=total_trials,
                                              bars_per_year=BARS_PER_YEAR,
                                              r_index=net_idx, b_index=bench_index)
        at = ev["alpha_test"]
        print(f"  alpha 年化 {at['alpha_annualized']*100:>8.2f}%   beta {at['beta']:>6.3f}"
              f"   t = {at['t_stat']:>6.2f}   p = {at['p_value']:.4f}")
        print(f"  DSR = {ev['dsr']['dsr']:.4f}    零技能下最大 Sharpe 期望 "
              f"{ev['dsr']['expected_max_sharpe_null']:.4f}   实际 Sharpe "
              f"{ev['dsr']['sharpe']:.4f}")
        print(f"  alpha bootstrap 95% 区间 "
              f"[{ev['alpha_bootstrap']['ci_low']*BARS_PER_YEAR*100:.2f}%, "
              f"{ev['alpha_bootstrap']['ci_high']*BARS_PER_YEAR*100:.2f}%] 年化")
        print(f"  最终判定（本次 {n_trials_this} 次试验惩罚）：{ev['verdict']}")
        print(f"  若按全项目累计 {total_trials} 次试验惩罚：{ev_total['verdict']}"
              f"（DSR {ev_total['dsr']['dsr']:.4f}）")

        # ---------- 4. 分段：持续学习在进步还是退化 ----------
        sr = split_report(best_res)
        print("\n=== 4. 分段表现（持续学习的有效性检验）===")
        for k, v in sr.items():
            print(f"  {k:<14} {v['bars']:>7,} 根   收益 {v['total_return_pct']:>8.2f}%"
                  f"   Sharpe {v['sharpe']:>6.2f}   最大回撤 {v['max_drawdown_pct']:>7.2f}%")

        # ---------- 5. 学到了什么 ----------
        print("\n=== 5. 专家权重（学到了什么）===")
        final_w = best["rec"]["final_expert_weights"]
        for k, v in sorted(final_w.items(), key=lambda kv: -kv[1])[:8]:
            bar = "█" * max(1, int(v * 40))
            print(f"  {k:<14}{v:>7.4f}  {bar}")

        # ---------- 落盘 ----------
        cdir = os.path.join(run.dir, "curves")
        os.makedirs(cdir, exist_ok=True)
        pd.DataFrame({
            "dt": best_res.index, "net": best_res.net_equity,
            "gross": best_res.gross_equity, "cost": best_res.cost_paid,
            "turnover": best_res.turnover_notional,
        }).to_csv(os.path.join(cdir, "hedge_best.csv"), index=False)
        pd.DataFrame({
            "step": best_ag.log["step"],
            "entropy": best_ag.log["entropy"],
            "turnover": best_ag.log["turnover"],
        }).to_csv(os.path.join(cdir, "learning_curve.csv"), index=False)
        wdf = pd.DataFrame(best_ag.expert_weight_history(),
                           columns=best_ag.expert_names())
        wdf.to_csv(os.path.join(cdir, "expert_weights.csv"), index=False)

        run.log("baselines", rows)
        run.log("best_config", {"rec": best["rec"], "split": sr,
                                "evaluation": ev, "evaluation_total_trials": ev_total})
        run.log("holdout_guard", guard.summary())
        run.record_metrics({
            "train_bars": n, "sealed_bars": guard.n - guard.cut,
            "n_configs_searched": n_configs,
            "n_trials_this_run": n_trials_this,
            "n_trials_project_total": total_trials,
            "best_verdict": ev["verdict"],
            "best_verdict_total_trials": ev_total["verdict"],
            "best_dsr": ev["dsr"]["dsr"],
            "best_alpha_annualized": at["alpha_annualized"],
            "best_alpha_t": at["t_stat"],
            "split": sr,
        })
        print(f"\nrun 目录: {os.path.relpath(run.dir, store.project_root())}")


if __name__ == "__main__":
    main()
