"""D 阶段重跑证据：**公平对照**（同引擎、同成本、同 band 语义）。

背景（WORK_LOG §14 / §15）：
D 阶段原本拿学习体去比 `equal_weight_buyhold` 基准，但那个基准用的是
`BuyHold()`（**只交易一次、不再平衡**）且**零成本**，而学习体每根 bar 都在按
band 再平衡并付真实成本。两者不可直接比 —— 差异里混了"再平衡拖累"这一项。

本脚本把对照换成同口径的：**同一套 SimExchange、同一套成本模型、同样的
band 语义**，只换决策函数。这样才能回答"学习体到底加了多少价值"。

结论（2026-09-15 实测）：学习体输给"等权三币每 bar 再平衡"2.4 倍，
且成本是对方的 2.8 倍。见 WORK_LOG §15.2 ②。

跑法：python3 scripts/d_fair_comparison.py        （约 3-4 分钟）
"""
from __future__ import annotations

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from src import config as cfgmod  # noqa: E402
from src.agents.online import HedgeEnsemble  # noqa: E402
from src.data import store  # noqa: E402
from src.eval.benchmark import equal_weight_buyhold  # noqa: E402
from src.registry.runs import Run  # noqa: E402
from src.sim.costs import CostModel  # noqa: E402
from src.sim.exchange import SimConfig, SimExchange  # noqa: E402
from src.sim.funding import FundingTable  # noqa: E402
from src.sim.margin import MarginConfig  # noqa: E402

BARS_PER_YEAR = 24 * 365


HYPOTHESIS = {
    "claim": ("在**同口径**对照下（同引擎、同成本、同 band 语义），"
              "学习体应当不差于等权再平衡组合；若明显更差，"
              "则说明 18 个专家 + 在线学习没有增加价值。"),
    "why": ("D 阶段的原始基准 `equal_weight_buyhold` 用 `BuyHold()`（只交易一次、"
            "不再平衡）且零成本，与策略不可直接比；差异里混了再平衡拖累。"),
    "falsified_by": "学习体终值显著低于等权再平衡组合（同引擎同成本）。",
    "registered": "2026-09-15",
    "note": ("本对照是在 D 阶段重跑（§15）之后补做的 —— 因为重跑才暴露出"
             "原基准不可比。此处如实记录：这是**事后补充的对照**，"
             "不是事前预注册。它的作用是修正解读，不是产生新结论。"),
}


class FixedWeights:
    """恒定目标权重（每根 bar 都提出同样的目标，靠 band 决定是否真交易）。"""

    def __init__(self, weights: dict, name: str):
        self.weights = weights
        self.name = name

    def decide(self, view):
        return dict(self.weights)


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
    print(f"窗口 {n:,} 根  {fr[syms[0]].index[0]:%Y-%m-%d} ~ {fr[syms[0]].index[-1]:%Y-%m-%d}\n")

    # 各标的自身的买入持有倍数 —— 解释"基准为什么是 158 倍"
    print("各标的 买入持有倍数（数据自带的起点/终点）")
    for s in syms:
        p0 = float(fr[s]["close"].iloc[0])
        p1 = float(fr[s]["close"].iloc[-1])
        print(f"  {s:<9} {p0:>12,.2f} → {p1:>12,.2f}   {p1/p0:>7.1f}x")
    print()

    funding = FundingTable.load(
        os.path.join(store.project_root(), "data", "funding.csv"))
    rates = np.array([0.0012, 0.0013, 0.0014])
    # 保证金参数：由实际杠杆推出（本脚本 max_gross / max_exposure = 1.0 ⇒ k = 1.0）。
    # 2026-09-15 修：此前硬编码 k=0.5（= 假设 2x）会给 1x 仓位加上与杠杆无关的
    # −44.4% 强平线；band 又阻止重入，于是静态规则被"停损"后永不返场。
    # 本脚本 2026-09-15 的那次 commit 产物是用 k=0.5 跑的，其数字受此影响
    # （见 FRAMEWORK_AUDIT.md 问题二的影响量化表）。
    EXPOSURE = 1.0
    margin_cfg = MarginConfig.for_leverage(
        EXPOSURE,
        maintenance_margin_ratio=0.1,
        topup_trigger_ratio=0.5,
    )
    net_cost = CostModel.from_config(cfg, enabled=True)
    gross_cost = CostModel(enabled=False)

    def run(ag, tag: str) -> dict:
        res = SimExchange(fr, gross_cost, net_cost,
                          SimConfig(initial_cash=10_000.0, warmup=300,
                                    max_gross=EXPOSURE, max_exposure_per_symbol=EXPOSURE,
                                    margin=margin_cfg, allow_short=True,
                                    instrument="perp")).run(ag)
        eq = res.net_equity
        rets = np.diff(eq) / eq[:-1]
        rets = rets[np.isfinite(rets)]
        return {
            "tag": tag, "final": float(eq[-1]),
            "cost": float(res.cost_paid.sum()),
            "sharpe": round(float(np.mean(rets) / np.std(rets) * np.sqrt(BARS_PER_YEAR)), 4)
            if len(rets) > 1 and np.std(rets) > 0 else 0.0,
            "max_dd": round(float(np.min(eq / np.maximum.accumulate(eq)) - 1.0), 4),
            "n_trades": res.n_trades,
            "bankrupt": res.bankrupt,
        }

    k = len(syms)
    cands = [
        (FixedWeights({s: 1.0 / k for s in syms}, "ew_all_rebalanced"),
         "等权三币 每bar再平衡（★公平对照）"),
        (FixedWeights({"BTCUSDT": 0.5, "ETHUSDT": 0.5, "BNBUSDT": 0.0}, "ew_btc_eth"),
         "BTC+ETH 各半 再平衡"),
        (FixedWeights({"BTCUSDT": 1.0, "ETHUSDT": 0.0, "BNBUSDT": 0.0}, "btc_only"),
         "纯 BTC 满仓"),
    ]

    with Run("d_fair_comparison", {
        "script": "scripts/d_fair_comparison.py",
        "bars": n, "universe": list(syms),
        "window": f"{fr[syms[0]].index[0]:%Y-%m-%d}~{fr[syms[0]].index[-1]:%Y-%m-%d}",
        "engine": "SimExchange(instrument=perp, allow_short=True, max_gross=1.0)",
        "margin": {"initial": margin_cfg.initial_margin_ratio,
                   "maintenance": margin_cfg.maintenance_margin_ratio,
                   "topup_trigger": margin_cfg.topup_trigger_ratio,
                   "derived_from": f"leverage={EXPOSURE}",
                   "long_liquidation_ratio": margin_cfg.long_liquidation_ratio()},
    }, HYPOTHESIS) as run_ctx:
        rows = []
        for ag, lab in cands:
            r = run(ag, ag.name)
            rows.append({**r, "label": lab})
            print(f"  {lab:<32} 终值 {r['final']:>12,.0f}  成本 {r['cost']:>10,.0f}  "
                  f"sharpe {r['sharpe']:>6.3f}  回撤 {r['max_dd']*100:>6.1f}%")

        # 学习体最优组（D 阶段 18 组里 sharpe 最高的那组超参）
        ag = HedgeEnsemble(syms, cost_rate=rates, eta=0.20, band=0.20,
                           max_exposure=1.0, funding=funding)
        r = run(ag, "hedge_best_d")
        r["label"] = "★ 学习体 η=0.20 band=0.20"
        rows.append(r)
        print(f"  {r['label']:<32} 终值 {r['final']:>12,.0f}  成本 {r['cost']:>10,.0f}  "
              f"sharpe {r['sharpe']:>6.3f}  回撤 {r['max_dd']*100:>6.1f}%")

        # 学习体的平均逐标的权重（解释它为什么跑不过基准）
        w = np.stack(ag.log["mixed_weight"], axis=0)
        expo = {s: {"mean_w": round(float(w[:, i].mean()), 4),
                    "mean_abs_w": round(float(np.abs(w[:, i]).mean()), 4),
                    "long_pct": round(float((w[:, i] > 1e-9).mean()), 4),
                    "short_pct": round(float((w[:, i] < -1e-9).mean()), 4)}
                for i, s in enumerate(syms)}
        print("\n学习体平均逐标的权重")
        for s, d in expo.items():
            print(f"  {s:<9} mean_w {d['mean_w']:>7.3f}  mean|w| {d['mean_abs_w']:>6.3f}  "
                  f"long {d['long_pct']*100:>5.1f}%  short {d['short_pct']*100:>5.1f}%")

        # 项目旧的基准口径（零成本 + 不再平衡）—— 只作参照，说明它为什么不可比
        bm = equal_weight_buyhold(fr, cfg)
        bh = 10_000 * (1 + bm.cumulative_return())
        print(f"\n参照：脚本旧基准口径（零成本 + 不再平衡）净终值 {bh:,.0f}  ← 与上表不可直接比")

        run_ctx.record_metrics({
            "portfolio": {r["label"]: {kk: vv for kk, vv in r.items() if kk != "label"}
                          for r in rows},
            "strategy_exposure": expo,
            "buyhold_incomparable": round(float(bh), 2),
            "verdict": ("学习体输给等权再平衡组合；差异主因是 BNB 敞口"
                        "（学习体 mean_w_BNB=%.3f vs 等权 %.3f）"
                        % (expo["BNBUSDT"]["mean_w"], 1.0 / k)),
        })
        run_ctx.note(
            f"学习体终值 {rows[-1]['final']:,.0f} vs 等权再平衡 "
            f"{rows[0]['final']:,.0f}（比值 {rows[-1]['final']/rows[0]['final']:.2f}x）；"
            f"成本 {rows[-1]['cost']:,.0f} vs {rows[0]['cost']:,.0f}。"
            f"假设按 falsified_by 被证伪：学习未增加价值。")
        run_ctx.note(
            f"旧基准口径（零成本+不再平衡）净终值 {bh:,.0f} —— "
            f"比等权再平衡高 {bh/rows[0]['final']:.1f}x，差额即再平衡拖累，"
            f"不可归因于策略。")
        print(f"\n证据目录：{run_ctx.dir}")


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"\n耗时 {time.time() - t0:.1f}s")
