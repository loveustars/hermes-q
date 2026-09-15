"""C 阶段基线 —— 3x 杠杆 + 逐腿保证金/强平。

目的：
  - 复用 B' baseline 的结构（funding 已接、14 专家含 3 做空、专家满仓与 max_exposure 解耦）
  - 加 MarginConfig：开启逐腿保证金账户和强平判定
  - 与 B' baseline 对比：alpha 是否还那么差？是否触发强平？
  - 不更新 run registry（diagnostic，不算新策略）

要看的事：
  - **C 必触发强平** —— B' 真实数据上 bar 1,420 触发了破产，但 sim 没逐腿强平。
    C 应该捕捉到强平事件（liquidated_legs 不为空），强平后该腿清零，agent 不能再建仓。
  - **杠杆 sim 现在接近实盘**：破产前会被逐腿强平打断
  - **C 的关键问题**：Hedge 长期 99% 在 long_all，加 3x + 强平后 early 段被强平，
    后段不再有 BTC 仓位 → 错过反弹

不做：
  - DSR / 多重试验惩罚
  - 调参（保持 A'/B' 同样的超参，隔离 C 这一变量）
  - 封存段评估
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
from src.eval.benchmark import equal_weight_buyhold  # noqa: E402
from src.sim.costs import CostModel  # noqa: E402
from src.sim.exchange import SimConfig, SimExchange  # noqa: E402
from src.sim.funding import FundingTable  # noqa: E402
from src.sim.margin import MarginConfig  # noqa: E402

BARS_PER_YEAR = 24 * 365
LEVERAGE = 3.0    # C 阶段：3x 杠杆 + 逐腿强平


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
    print(f"基准（等权买入持有，零成本）累计收益 {bm.cumulative_return()*100:,.1f}%\n")

    rates = np.array([0.0012, 0.0013, 0.0014])

    eta, band = 0.05, 0.05
    funding = FundingTable.load(
        os.path.join(store.project_root(), "data", "funding.csv"))
    print(f"funding 数据 {len(funding.rates_by_hour.get('BTCUSDT', {}))} 条")
    print(f"杠杆上限 = {LEVERAGE}x  （agent max_exposure + sim max_gross）")

    # 关键：margin 配置 + per-symbol 杠杆 + 总杠杆 配齐
    # 2026-09-15 修：k 由实际杠杆推出（k = 1/LEVERAGE）而不是硬编码 0.5。
    # 硬编码 0.5 等于假设 2x 杠杆 ⇒ 无论 LEVERAGE 填几，强平阈值都是 −44.4%
    # （阈值只由 (k, m) 决定），"3x 强平风险更高"在代码层面根本不成立。
    # 见 src/sim/margin.py 模块 docstring 的第二次修复。
    margin_cfg = MarginConfig.for_leverage(
        LEVERAGE,
        maintenance_margin_ratio=0.1,    # 10% 维持保证金
        topup_trigger_ratio=0.5,
    )
    print(f"margin: {margin_cfg.liquidation_description()}\n")
    ag = HedgeEnsemble(syms, cost_rate=rates, eta=eta, band=band,
                       max_exposure=LEVERAGE, funding=funding)

    res = SimExchange(fr, CostModel(enabled=False),
                      CostModel.from_config(cfg, enabled=True),
                      SimConfig(initial_cash=10_000.0, warmup=300,
                                max_gross=LEVERAGE,
                                max_exposure_per_symbol=LEVERAGE,
                                margin=margin_cfg,
                                allow_short=True,
                                instrument="perp")).run(ag)

    print(f"=== C 阶段基线（HedgeEnsemble × {ag.K} 专家，{LEVERAGE}x 杠杆 + 逐腿强平）===")
    print(f"η={eta}  band={band}  杠杆 {LEVERAGE}x  专家数 {ag.K}")
    print(f"成交笔数 {res.n_trades:,}   归零={res.insolvent}")
    print(f"毛终值 {res.final_gross():>14,.0f}   净终值 {res.final_net():>14,.0f}"
          f"   累计成本 {res.cost_paid.sum():>10,.0f}")
    print(f"换手名义额 {res.turnover_notional.sum():>16,.0f}"
          f"  = 本金 {res.turnover_notional.sum()/10000:,.1f} 倍")
    print(f"**强平腿** {res.liquidated_legs}（{len(res.liquidated_legs)} 个）")
    if res.insolvent:
        print(f"  ⚠ 破产时刻：bar {res.insolvent_at:,}\n")
    else:
        print()

    bh_final = 10_000 * (1 + bm.cumulative_return())
    print(f"基准（buy_hold）净终值  {bh_final:>14,.0f}")
    print(f"C agent  净终值         {res.final_net():>14,.0f}"
          f"   alpha {(res.final_net()/bh_final - 1)*100:>+10.2f}%\n")

    # ---------- 杠杆 + 强平后的关键指标 ----------
    mixed_log = ag.log["mixed_weight"]
    mixed = np.stack(mixed_log, axis=0) if len(mixed_log) > 1 else mixed_log[0][None, :]
    n_bars = mixed.shape[0]

    print(f"=== 杠杆 + 强平后的权重范围（{n_bars:,} 根 bar）===")
    print(f"  max_exposure = {LEVERAGE}, max_gross = {LEVERAGE}")
    print(f"  min(w) 历史最低：{mixed.min():>+.4f}")
    print(f"  max(w) 历史最高：{mixed.max():>+.4f}")
    print(f"  min(Σ|w|) 历史最低：{np.abs(mixed).sum(axis=1).min():>+.4f}")
    print(f"  max(Σ|w|) 历史最高：{np.abs(mixed).sum(axis=1).max():>+.4f}"
          f"  （应 ≤ {LEVERAGE}）\n")

    # ---------- 强平段定位 ----------
    print(f"=== 强平段（liquidated_legs 按时序）===")
    for i, s in enumerate(res.liquidated_legs):
        # 找强平时刻（接近 bar 1,420 那种早期）
        # 我们通过 net_eq 序列反推强平时刻
        # 简化：直接给个提示，让用户去看 curves CSV
        print(f"  {i+1}. {s}  （看 runs/c_margin_baseline/curves/hedge.csv 定位时刻）")

    # ---------- 落盘 ----------
    out_dir = os.path.join(store.project_root(), "runs", "c_margin_baseline", "curves")
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
                  "gross": [np.abs(m).sum() for m in mixed_log],
                  "turnover": ag.log["turnover"],
                  "entropy": ag.log["entropy"]}).to_csv(
        os.path.join(out_dir, "learning_curve.csv"), index=False)
    print(f"\n落盘到 {os.path.relpath(out_dir, store.project_root())}")


if __name__ == "__main__":
    main()
