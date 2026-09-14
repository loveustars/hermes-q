"""G1 校准闸门：对 100 个纯噪声策略，评估器的假阳性率必须 <= 5%。

这是硬门槛。评估器不过关，后面所有"发现"都是假的，禁止进入自学习阶段（M5）。

本脚本用**真实撮合器**（SimExchange）生成回测结果，不使用近似快路径。
背景：早期版本用向量化权重法做快路径，交叉验证发现它与撮合器偏差高达 134pp
（换仓时点与成交价假设不一致），已整体废弃——评估器的校准不能建立在近似模型上。

校准分两个口径，这是关键：
  (A) 毛口径（零成本）：零技能假设在此成立，才是对统计机制本身的真正校准。
      预期——朴素判定（只看收益为正）假阳性率接近 100%，这就是牛市陷阱；
           剥离 beta 后约 5%；加 DSR 多重试验惩罚后接近 0%。
  (B) 净口径（含成本）：成本把噪声策略的 alpha 系统性推向负值，
      假阳性率应当接近 0，说明"扣费后还能显著"这个门槛有多难。
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
from src.agents.baselines import RandomWeights  # noqa: E402
from src.data import store  # noqa: E402
from src.eval import protocol  # noqa: E402
from src.eval.benchmark import equal_weight_buyhold, strategy_returns  # noqa: E402
from src.registry.runs import Run  # noqa: E402
from src.sim.costs import CostModel  # noqa: E402
from src.sim.exchange import SimConfig, SimExchange  # noqa: E402

BARS_PER_YEAR = 24 * 365
N_TRIALS = 100
REBALANCE_CHOICES = [1, 6, 24, 72, 168, 720]


def load_frames(symbols: list[str], train_only: bool = False,
                holdout_fraction: float = 0.25) -> dict[str, pd.DataFrame]:
    """加载核心池的 1h 现货数据，取公共时间轴。

    `train_only=True` 时只保留前 (1 - holdout_fraction) 段，
    **把封存段排除在协议校准之外**。

    为什么要这个开关：G1 用的是**真实行情上的随机权重策略**，
    所以它的校准窗口默认会覆盖封存段（2023-09-15 ~ 2026-09-14 与封存段
    2025-01-20 ~ 2026-09-14 重叠）。严格说，用封存段来标定协议阈值是
    一种（很轻微的）泄漏。加上这个开关后可以跑"仅训练段"的对照，
    消除这个疑问。默认仍为全窗口，以便与改判据前的数字直接对比。
    """
    frames = {}
    for s in symbols:
        frames[s] = store.load_bars(s, "1h")[
            ["open", "high", "low", "close", "volume", "quote_volume"]]
    idx = None
    for f in frames.values():
        idx = f.index if idx is None else idx.intersection(f.index)
    if train_only:
        idx = idx[:int(len(idx) * (1.0 - holdout_fraction))]
    return {s: f.loc[idx] for s, f in frames.items()}


def naive_verdict(returns: np.ndarray) -> bool:
    """朴素判定：只看平均收益是否显著为正，不剥离 beta。这就是陷阱。"""
    boot = protocol.bootstrap_p_value(returns, block=24, null=0.0)
    return boot["p_value"] < 0.05


def run_calibration(cfg, frames, bench, bench_index, label: str, enabled: bool) -> dict:
    cm_net = CostModel.from_config(cfg, enabled=enabled)
    cm_gross = CostModel.from_config(cfg, enabled=False)
    simcfg = SimConfig(initial_cash=10_000.0, warmup=300, latency_bars=1)

    naive_flags = alpha_flags = dsr_flags = full_flags = 0
    ci_flags = degenerate_flags = 0
    binding_sets: dict[tuple, int] = {}
    results, tstats, rets_for_pbo = [], [], []
    t0 = time.time()
    for i in range(N_TRIALS):
        rb = REBALANCE_CHOICES[i % len(REBALANCE_CHOICES)] * (1 + i // len(REBALANCE_CHOICES))
        ag = RandomWeights(seed=1000 + i, rebalance_every=rb)
        res = SimExchange(frames, cm_gross, cm_net, simcfg).run(ag)
        r, r_idx = strategy_returns(res)
        if len(r) < 100:
            continue

        naive = naive_verdict(r)
        full = protocol.evaluate_strategy(r, bench, n_trials=N_TRIALS,
                                          bars_per_year=BARS_PER_YEAR,
                                          r_index=r_idx, b_index=bench_index)
        a_only = full["alpha_test"]["p_value"] < 0.05
        d_only = full["dsr"]["dsr"] > 0.95
        c_only = full["alpha_bootstrap"]["ci_low"] > 0
        degen = bool(full["distribution"]["degenerate"])
        bs = tuple(full["binding_criteria"])

        naive_flags += int(naive)
        alpha_flags += int(a_only)
        dsr_flags += int(d_only)
        ci_flags += int(c_only)
        degenerate_flags += int(degen)
        binding_sets[bs] = binding_sets.get(bs, 0) + 1
        full_flags += int(full["passed"])
        tstats.append(full["alpha_test"]["t_stat"])
        rets_for_pbo.append(r)
        results.append({
            "trial": i, "rebalance_every": rb,
            "final_equity": round(res.final_net(), 2),
            "total_return_pct": round((res.final_net() / res.initial_cash - 1) * 100, 2),
            "cost_as_pct_of_initial": round(
                float(res.cost_paid.sum()) / res.initial_cash * 100, 2),
            "insolvent_at": (str(res.index[res.insolvent_at])
                             if res.insolvent_at is not None else None),
            "naive_flagged": bool(naive),
            "alpha_t": full["alpha_test"]["t_stat"],
            "alpha_p": full["alpha_test"]["p_value"],
            "beta": full["alpha_test"]["beta"],
            "dsr": full["dsr"]["dsr"],
            "alpha_ci_low": full["alpha_bootstrap"]["ci_low"],
            "kurtosis": full["distribution"]["kurtosis"],
            "degenerate": degen,
            "binding_criteria": list(bs),
            "waived_criteria": full["waived_criteria"],
            "alpha_only_flagged": bool(a_only),
            "dsr_flagged": bool(d_only),
            "alpha_ci_flagged": bool(c_only),
            "verdict": full["verdict"],
        })
        if (i + 1) % 25 == 0:
            print(f"    [{label}] {i+1}/{N_TRIALS} 完成  "
                  f"已耗时 {time.time()-t0:.0f}s", flush=True)

    n = len(results)
    pct = lambda x: round(x / max(n, 1) * 100, 1)
    ts = np.array(tstats) if tstats else np.array([0.0])
    pbo = protocol.pbo_cscv(np.column_stack(rets_for_pbo), n_blocks=16) \
        if len(rets_for_pbo) >= 2 else {"pbo": None}
    return {
        "label": label,
        "costs_enabled": enabled,
        "n": n,
        "fp_rate_pct": {
            "naive_no_beta_removal": pct(naive_flags),
            "alpha_after_beta_removal": pct(alpha_flags),
            "dsr_only": pct(dsr_flags),
            "alpha_ci_only": pct(ci_flags),
            "full_protocol": pct(full_flags),
        },
        "degenerate_pct": pct(degenerate_flags),
        "binding_sets": {"+".join(k): v for k, v in
                         sorted(binding_sets.items(), key=lambda kv: -kv[1])},
        "alpha_t": {"median": round(float(np.median(ts)), 3),
                    "mean": round(float(ts.mean()), 3),
                    "p95": round(float(np.quantile(ts, 0.95)), 3),
                    "p05": round(float(np.quantile(ts, 0.05)), 3)},
        "pbo": pbo.get("pbo"),
        "pbo_combos": pbo.get("n_combinations"),
        "elapsed_s": round(time.time() - t0, 1),
        "results": results,
    }


def print_block(r: dict) -> None:
    fp = r["fp_rate_pct"]
    print(f"\n  【{r['label']}】n={r['n']}  耗时 {r['elapsed_s']}s")
    print(f"    ① 朴素判定（只看收益为正，不剥离 beta） : {fp['naive_no_beta_removal']:>5}%")
    print(f"    ② 剥离 beta 的 alpha (p<0.05)            : {fp['alpha_after_beta_removal']:>5}%"
          f"   ← 门槛 5%")
    print(f"    ③ DSR > 0.95（按 {N_TRIALS} 次试验惩罚）      : {fp['dsr_only']:>5}%")
    print(f"    ③b 仅自助区间下界 > 0（不依赖分布假设）     : {fp['alpha_ci_only']:>5}%")
    print(f"    ④ 协议最终判定（分布感知，见下）           : {fp['full_protocol']:>5}%"
          f"   ← 门槛 5%")
    print(f"    分布退化（峰度 > {protocol.KURT_MAX_WELL_BEHAVED:.0f}）的比例："
          f"{r['degenerate_pct']}%")
    for k, v in r.get("binding_sets", {}).items():
        print(f"      生效判据 {k:<32} : {v:>4} 次")
    t = r["alpha_t"]
    print(f"    alpha t 分布：中位数 {t['median']:>7}  均值 {t['mean']:>7}  "
          f"p05 {t['p05']:>7}  p95 {t['p95']:>7}")
    print(f"    PBO = {r['pbo']}  (组合数 {r['pbo_combos']})")


def main(train_only: bool = False) -> None:
    cfg = cfgmod.load("base")
    cfgmod.guard_frozen(cfg)
    syms = cfg["universe"]["core"]

    frames = load_frames(syms, train_only=train_only)
    n_bars = len(frames[syms[0]])
    bm = equal_weight_buyhold(frames, cfg)
    bench, bench_index = bm.returns, bm.index
    tag = "仅训练段（封存段已排除）" if train_only else "全窗口"
    print(f"校准窗口 [{tag}] {n_bars:,} 根小时线  "
          f"{frames[syms[0]].index[0]:%Y-%m-%d} ~ {frames[syms[0]].index[-1]:%Y-%m-%d}")
    print(f"基准（等权买入持有，同一撮合器零成本）累计收益 "
          f"{bm.cumulative_return() * 100:,.1f}%")

    hypothesis = {
        "question": "评估器会不会把没有预测力的策略判成'有边际'？"
                    "（口径改为分布感知后必须重验，确认假阳性率没被放松）",
        "expected": "毛口径：朴素判定约 100%（牛市陷阱），剥离 beta 约 5%，DSR 惩罚后约 0%；"
                    "净口径：全部接近 0（成本把 alpha 推向负值）",
        "decision_rule": "毛口径下剥离 beta 的假阳性率 > 10% 就地停住修评估器；"
                         "本次改判据后门槛仍为 5%（PLAN §15.9 第 1 项）",
        "window": tag,
        "train_only": bool(train_only),
    }

    name = "m4_g1_calibration_trainonly" if train_only else "m4_g1_calibration"
    with Run(name, cfg, hypothesis) as run:
        print(f"\n=== 校准开始（真实撮合器，{N_TRIALS} 个纯噪声策略）===")
        gross = run_calibration(cfg, frames, bench, bench_index, "毛口径·零成本",
                                enabled=False)
        net = run_calibration(cfg, frames, bench, bench_index, "净口径·含成本",
                              enabled=True)

        print_block(gross)
        print_block(net)

        fp_gross = gross["fp_rate_pct"]["alpha_after_beta_removal"]
        gate_pass = fp_gross <= 5.0
        out = {
            "n_trials": N_TRIALS,
            "window_bars": n_bars,
            "gross": {k: v for k, v in gross.items() if k != "results"},
            "net": {k: v for k, v in net.items() if k != "results"},
            "G1_basis": "毛口径（零技能假设成立）下剥离 beta 的假阳性率",
            "G1_value_pct": fp_gross,
            "G1_threshold_pct": 5.0,
            "G1_gate_pass": bool(gate_pass),
        }
        run.log("g1_results", out)
        run.log("g1_trials_gross", gross["results"])
        run.log("g1_trials_net", net["results"])
        run.record_metrics(out)

        print("\n--- G1 闸门 ---")
        print(f"  判据：毛口径下剥离 beta 的假阳性率 {fp_gross}%  门槛 5%"
              f"  （口径：{tag}）")
        print(f"  结果：{'通过 —— 协议可用' if gate_pass else '不通过 —— 先修评估器'}")
        print(f"  run 目录: {os.path.relpath(run.dir, store.project_root())}")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--train-only", action="store_true",
                    help="只用训练段（排除封存段）做协议校准")
    main(train_only=ap.parse_args().train_only)
