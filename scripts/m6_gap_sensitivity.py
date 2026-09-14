"""缺口敏感性对照：跨缺口成交 vs 跨缺口不成交。

实测缺口：BTC/ETH 各 28 处、累计缺 128 根（0.161%），最长一处 34 小时，
三个标的缺口位置完全一致（交易所停机）。

本脚本回答一个问题：**这个量级会不会改变任何结论？**
对同一批策略跑两种 gap_policy，比较终值、成本、换手与判定。
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src import config as cfgmod  # noqa: E402
from src.agents import baselines as B  # noqa: E402
from src.agents.online import HedgeEnsemble  # noqa: E402
from src.data import store  # noqa: E402
from src.data.quality import gap_report, union_gap_flags  # noqa: E402
from src.eval import metrics, protocol  # noqa: E402
from src.eval.benchmark import equal_weight_buyhold, strategy_returns  # noqa: E402
from src.registry.runs import Run  # noqa: E402
from src.sim.costs import CostModel  # noqa: E402
from src.sim.exchange import SimConfig, SimExchange  # noqa: E402

BARS_PER_YEAR = 24 * 365


def load_frames(symbols):
    frames = {s: store.load_bars(s, "1h")[
        ["open", "high", "low", "close", "volume", "quote_volume"]] for s in symbols}
    idx = None
    for f in frames.values():
        idx = f.index if idx is None else idx.intersection(f.index)
    return {s: f.loc[idx] for s, f in frames.items()}


def main() -> None:
    cfg = cfgmod.load("base")
    syms = cfg["universe"]["core"]
    frames = load_frames(syms)

    idx = frames[syms[0]].index
    print("=== 0. 缺口概况 ===")
    for s in syms:
        r = gap_report(frames[s].index)
        print(f"  {s:<9} 根数 {r['bars']:,}  缺口 {r['n_gaps']} 处  缺 {r['missing_bars']} 根"
              f"（{r['missing_pct']}%）  最长 {r['longest']}  首次 {r['first_gap']}")
    union = union_gap_flags(frames)
    print(f"  三标的缺口并集：{int(union.sum())} 根 bar 被标记")
    print("  说明：BTC 与 ETH 缺口位置完全相同 → 交易所级别停机，不是单标的问题")

    bm = equal_weight_buyhold(frames, cfg)
    rates = np.array([
        0.001 + 0.0001 + float(np.log(frames[s]["close"]).diff().std())
        * np.sqrt(1e4 * 0.1 / float(frames[s]["quote_volume"].median()))
        for s in syms])

    def make_agents():
        return [
            B.SingleAssetBuyHold("BTCUSDT"),
            B.BuyHold(),
            B.Momentum("BTCUSDT", lookback=720, rebalance_every=24),
            B.RandomWeights(seed=11, rebalance_every=24),
            B.RandomWeights(seed=11, rebalance_every=1),
            HedgeEnsemble(syms, cost_rate=rates, eta=0.02, band=0.02),
        ]

    hypothesis = {
        "question": "跨缺口成交（execute）与跨缺口不成交（skip）会不会改变结论？",
        "expected": "缺口仅占 0.157%，差异应以本金的百分点计很小，且不改变任何判定",
        "decision_rule": "若任一策略的差异超过本金的 1 个百分点，或出现判定翻转，"
                         "则必须在报告中显式披露；默认口径保持 execute"
                         "（真实交易者在停机期间无法下单，复牌后即可按复牌开盘价成交）",
    }

    with Run("m6_gap_sensitivity", cfg, hypothesis) as run:
        print("\n=== 1. 两种口径对照 ===")
        print("  差异口径：以**初始本金的百分点**计。相对终值计算会在终值趋近 0 时变成除零噪声。")
        hdr = (f"{'策略':<24}{'execute终值':>14}{'skip终值':>14}{'差异(本金pp)':>13}"
               f"{'跨缺口成交':>12}{'跳过':>8}{'判定':>16}")
        print(hdr)
        rows, flips = [], 0
        for ag_e, ag_s in zip(make_agents(), make_agents()):
            out = {}
            for pol, ag in (("execute", ag_e), ("skip", ag_s)):
                res = SimExchange(
                    frames, CostModel(enabled=False),
                    CostModel.from_config(cfg, enabled=True),
                    SimConfig(initial_cash=10_000.0, warmup=300, gap_policy=pol)
                ).run(ag)
                r, ri = strategy_returns(res)
                ev = protocol.evaluate_strategy(r, bm.returns, n_trials=6,
                                                bars_per_year=BARS_PER_YEAR,
                                                r_index=ri, b_index=bm.index)
                out[pol] = {"res": res, "ev": ev}

            fe = out["execute"]["res"].final_net()
            fs = out["skip"]["res"].final_net()
            dev_pp = (fs - fe) / 10_000.0 * 100        # 初始本金的百分点
            insol = out["execute"]["res"].insolvent or out["skip"]["res"].insolvent
            ve = out["execute"]["ev"]["verdict"]
            vs = out["skip"]["ev"]["verdict"]
            flip = ve != vs
            if flip:
                flips += 1
            rows.append({"agent": ag_e.name, "execute_final": fe, "skip_final": fs,
                         "deviation_pp_of_principal": round(dev_pp, 4),
                         "gap_trades": out["execute"]["res"].n_gap_trades,
                         "gap_skipped": out["skip"]["res"].n_gap_skipped,
                         "insolvent": bool(insol),
                         "verdict_execute": ve, "verdict_skip": vs, "flip": bool(flip)})
            mark = "★翻转" if flip else "一致"
            tag = "（已归零，差异无意义）" if insol else ""
            print(f"{ag_e.name:<24}{fe:>14,.0f}{fs:>14,.0f}{dev_pp:>12.3f}pp"
                  f"{out['execute']['res'].n_gap_trades:>12,}"
                  f"{out['skip']['res'].n_gap_skipped:>8,}{mark:>16}{tag}")

        alive = [r for r in rows if not r["insolvent"]]
        maxdev = max((abs(r["deviation_pp_of_principal"]) for r in alive), default=0.0)
        print(f"\n  存活策略的最大差异 {maxdev:.4f} 个本金百分点   判定翻转 {flips} 处")
        print("  口径结论：默认保持 execute —— 停机期间交易者确实无法下单，"
              "复牌后按复牌开盘价成交才是真实情形；skip 仅作敏感性对照。")
        if maxdev < 1.0 and flips == 0:
            print("  敏感性结论：缺口对结论无实质影响，无需在后续实验中额外处理。")
        else:
            print("  敏感性结论：差异已达本金的 1 个百分点以上，需在报告中显式披露。")

        run.log("gap_sensitivity", rows)
        run.record_metrics({
            "gap_bars": int(union.sum()),
            "gap_pct": round(float(union.mean()) * 100, 4),
            "max_deviation_pp_of_principal": round(maxdev, 4),
            "verdict_flips": flips,
            "default_policy": "execute",
            "note": "差异以初始本金百分点计；相对终值计算在归零情形下是除零噪声",
        })
        print(f"\nrun 目录: {os.path.relpath(run.dir, store.project_root())}")


if __name__ == "__main__":
    main()
