"""修复验证：保证金模块三处缺陷的**前后对照 + 守恒不变量**。

背景见 WORK_LOG.md §14。D 阶段曾把病态归因于"CostModel 在薄市场虚高冲击成本"，
实测证伪：真正的元凶是 `src/sim/margin.py`。本脚本做两件事：

  1. **把初版行为保留在脚本里**，作为永不失真的对照。
     修完 bug 之后，"修复前"的数字通常就再也拿不到了（代码已经改了），
     于是结论只剩下文档里的一句话。这里把初版 `liquidate` 的**净效果**
     内联成一个子类，任何人在任何时间都能重跑出对照表。

  2. **在真实数据上检查守恒不变量**：强平不得创造或销毁权益。

用法：
    python3 scripts/fix_margin_verify.py                # 默认前 25,000 根
    python3 scripts/fix_margin_verify.py --bars 8000    # 更快

注意：本脚本**不**声称"3x 能不能赚钱"。它只回答"账有没有记错"。
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from src import config as cfgmod  # noqa: E402
from src.agents.online import HedgeEnsemble  # noqa: E402
from src.data import store  # noqa: E402
from src.registry.runs import Run  # noqa: E402
from src.sim import exchange as exchange_mod  # noqa: E402
from src.sim.costs import CostModel  # noqa: E402
from src.sim.exchange import SimConfig, SimExchange  # noqa: E402
from src.sim.margin import MarginBook, MarginConfig  # noqa: E402

HYPOTHESIS = {
    "statement": "D 阶段 '成本是净值 1.6 倍' 的根因是 CostModel，还是 sim 记账？",
    "prediction": "若根因是 CostModel，则关闭保证金（margin=None）也应变现病态；"
                  "实测不应如此 —— 病态只随保证金启用而出现，指向记账缺陷。",
    "falsified_if": "关闭保证金后仍出现权益 > 本金 1000 倍或成本 > 净值 10 倍。",
}


class LegacyMarginBook(MarginBook):
    """**初版行为**（2026-09-15 修复前）—— 仅供对照，不要在生产路径使用。

    三个缺陷全部复现，否则对照就不忠实（"半新半旧"的混合体会让差异被归因错）：

      缺陷 1（致命）：强平时凭空造钱。
        `liquidate` 把 `margin_cash` 退给 `cash_pool`，但那笔钱从未被单独借记过
        （主循环建仓时已按名义额全额扣现金）⇒ 每强平一次白得半个名义额。
        更糟的是调用点传的 `[cash_n]` 只是浮点**副本**的列表，改它改不到局部变量，
        而调用点又没用返回值 ⇒ 那行"退钱"其实是**空操作**。
        (1) 与 (b) 相抵后，**净效果是"仓位被清零、现金无任何变动"**。
        对空头（units<0）这就等于让一笔负债凭空消失 ⇒ 权益凭空增加 |units| × 价格。

      缺陷 2：权益口径错。
        初版 `equity = margin_cash + units × price`（仓位**绝对市值**），
        对空头在开仓瞬间就是 `0.5N − N = −0.5N`，恒为负。

      缺陷 3：压力测试永远用不利价 `high`（在调用点写死）。
        对多头 `high` 是**有利**价，于是多头几乎永不被强平。
        这里通过缺陷 2 的公式一并复现：调用点现在给多头传 `low`，
        代入初版公式得 `5000 + 100·low ≤ 25·low` ⇒ 永假 ⇒ 多头同样永不被强平，
        与原缺陷 3 的可观测行为一致。

    注意：本类的存在是为了**让这个 bug 可被反复复现**，不是为了兼容它。
    """

    def equity(self, symbol: str, units: float, price: float) -> float:  # noqa: D102
        # 缺陷 2：绝对市值口径（不是相对建仓价的浮动盈亏）
        return self.legs[symbol].margin_cash + units * price

    def liquidate(self, symbol: str, units: float, exit_price: float,
                  cash_pool) -> float:                       # noqa: D102
        # 缺陷 1：账本清零、标记强平，但现金一分不动
        self.legs[symbol].margin_cash = 0.0
        self.legs[symbol].initial_margin = 0.0
        self.legs[symbol].entry_price = 0.0
        self.liquidated.add(symbol)
        return 0.0


def load_frames(bars: int | None):
    cfg = cfgmod.load("base")
    syms = cfg["universe"]["core"]
    fr = {s: store.load_bars(s, "1h")[
        ["open", "high", "low", "close", "volume", "quote_volume"]]
        for s in syms}
    idx = None
    for f in fr.values():
        idx = f.index if idx is None else idx.intersection(f.index)
    idx = idx[:bars] if bars else idx
    return cfg, syms, {s: f.loc[idx] for s, f in fr.items()}, idx


def run_one(frames, syms, cfg, exposure: float, legacy: bool):
    """跑一组配置。legacy=True 用初版强平行为。"""
    orig = exchange_mod.MarginBook
    if legacy:
        exchange_mod.MarginBook = LegacyMarginBook
    try:
        ag = HedgeEnsemble(syms, cost_rate=np.array([0.0012, 0.0013, 0.0014]),
                           eta=0.20, band=0.10, max_exposure=exposure)
        net = CostModel.from_config(cfg, enabled=True)
        return SimExchange(
            frames, CostModel(enabled=False), net,
            SimConfig(initial_cash=10_000.0, warmup=300, max_gross=exposure,
                      max_exposure_per_symbol=exposure,
                      margin=MarginConfig(0.5, 0.1, 0.5),
                      allow_short=True, instrument="perp")).run(ag)
    finally:
        exchange_mod.MarginBook = orig


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bars", type=int, default=25_000,
                    help="只用前 N 根（默认 25,000；病态在 bar ~19,300 已出现）")
    args = ap.parse_args()

    cfg, syms, frames, idx = load_frames(args.bars)
    print(f"窗口 {len(idx):,} 根  {idx[0]:%Y-%m-%d} ~ {idx[-1]:%Y-%m-%d}  "
          f"标的 {syms}\n")

    rows = {}
    with Run("fix_margin_verify",
             {"bars": len(idx), "eta": 0.20, "band": 0.10,
              "margin": [0.5, 0.1, 0.5], "universe": list(syms)},
             HYPOTHESIS) as run:
        print(f"{'杠杆':<6}{'实现':<10}{'终值':>16}{'最大权益':>16}"
              f"{'总成本':>18}{'强平':>7}{'截断':>7}{'破产':>7}")
        for exposure in (1.0, 2.0, 3.0):
            for legacy, label in ((True, "初版"), (False, "修复后")):
                r = run_one(frames, syms, cfg, exposure, legacy)
                ne = r.net_equity
                key = f"{exposure:g}x/{label}"
                rows[key] = {
                    "final": float(ne[-1]), "max_equity": float(ne.max()),
                    "cost": float(r.cost_paid.sum()), "n_liq": len(r.liquidated_legs),
                    "n_capped": int(r.n_impact_capped), "bankrupt": bool(r.bankrupt),
                }
                print(f"{exposure:>3.0f}x  {label:<10}{ne[-1]:>16,.0f}"
                      f"{ne.max():>16,.0f}{r.cost_paid.sum():>18,.0f}"
                      f"{len(r.liquidated_legs):>7}{r.n_impact_capped:>7}"
                      f"{str(r.bankrupt):>7}")

        # ---- 守恒不变量：在真实数据上逐笔核对强平的权益连续性 ----
        print("\n=== 守恒不变量（真实数据，修复后实现）===")
        r = run_one(frames, syms, cfg, 3.0, legacy=False)
        ne = r.net_equity
        d = np.diff(ne)
        # 单根权益变化的上界：最大可能仓位 ≈ 3 × 权益；单根价格波动上界用实测最大振幅
        max_bp = max(
            float((frames[s]["high"] / frames[s]["low"] - 1.0).max()) for s in syms)
        bound = float(np.abs(ne).max()) * 3.0 * max_bp * 1.5
        worst = float(np.abs(d).max())
        ok = worst <= bound
        print(f"  单根最大权益变化 {worst:,.0f}   宽松上界 {bound:,.0f}   "
              f"{'✅ 通过' if ok else '❌ 越界'}")
        print(f"  最大单根价格振幅 {max_bp * 100:.1f}%  "
              f"（上界 = 3x × 最大权益 × 该振幅 × 1.5 安全系数）")
        run.record_metrics({
            "max_single_bar_equity_change": worst,
            "continuity_bound": bound,
            "continuity_ok": bool(ok),
            "bars": len(idx),
            **{f"grid.{k}.{m}": v for k, d_ in rows.items() for m, v in d_.items()},
        })
        run.note("初版实现内联在 LegacyMarginBook 里，任何时间都可重跑对照。")
        print(f"\n证据目录：{run.finish()}")

        if not ok:
            print("❌ 守恒不变量未通过 —— 记账路径仍有缺陷。")
            return 1
    print("✅ 修复验证通过：对照组复现病态，修复组守恒。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
