"""B 阶段基线 —— 3x 杠杆。

目的：
  - 复用 A' baseline 的结构（funding 已接、14 专家含 3 做空）
  - 把 max_gross_exposure 从 1.0 提到 3.0（agent 层 + sim 层配齐）
  - 不改其他参数（η=0.05, band=0.05 保持不变），隔离"杠杆"这一个变量
  - 不更新 run registry（diagnostic，不算新策略）

要看的事：
  - 杠杆后净终值 vs A' baseline 净终值（应放大盈亏，但** sim 不会逐腿爆仓**）
  - 段 1/2/3/4/5 各自的"杠杆放大"是否对称
  - **重点**：C 阶段没做 → sim 失真，看 insolvent_at 是否被触发
    （B 阶段 sim 仍然只在 net_eq < 1% 时标 insolvent，不是逐腿强平）

不做：
  - DSR / 多重试验惩罚
  - 调参（保持 A' 同样的超参，隔离杠杆变量）
  - 封存段评估

已知风险（PLAN §19.6）：
  - **C 阶段没做** → sim 不会逐腿强平，alpha 看起来会比实盘乐观
  - 终态 long_all ≈ 1.0 → 加 3x 后多头仓位 = 3x，** BTC 跌 33% 净资产归零**
  - 但 sim 只看 net_eq < 1% 才标 insolvent，所以可能**触发不到**真正的破产路径
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

BARS_PER_YEAR = 24 * 365
LEVERAGE = 3.0    # B 阶段的核心变量


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
    print(f"杠杆上限 = {LEVERAGE}x  （agent max_exposure + sim max_gross）\n")

    # 关键：agent 层 max_exposure 与 sim 层 max_gross 必须同时改
    ag = HedgeEnsemble(syms, cost_rate=rates, eta=eta, band=band,
                       max_exposure=LEVERAGE, funding=funding)

    res = SimExchange(fr, CostModel(enabled=False),
                      CostModel.from_config(cfg, enabled=True),
                      SimConfig(initial_cash=10_000.0, warmup=300,
                                max_gross=LEVERAGE)).run(ag)

    print(f"=== B 阶段基线（HedgeEnsemble × {ag.K} 专家，{LEVERAGE}x 杠杆）===")
    print(f"η={eta}  band={band}  杠杆 {LEVERAGE}x  专家数 {ag.K}")
    print(f"成交笔数 {res.n_trades:,}   归零={res.insolvent}  "
          f"（注意：C 阶段没做，sim 不会逐腿爆仓）")
    print(f"毛终值 {res.final_gross():>14,.0f}   净终值 {res.final_net():>14,.0f}"
          f"   累计成本 {res.cost_paid.sum():>10,.0f}")
    print(f"换手名义额 {res.turnover_notional.sum():>16,.0f}"
          f"  = 本金 {res.turnover_notional.sum()/10000:,.1f} 倍")
    if res.insolvent:
        print(f"  ⚠ 破产时刻：bar {res.insolvent_at:,}\n")
    else:
        print()

    bh_final = 10_000 * (1 + bm.cumulative_return())
    print(f"基准（buy_hold）净终值  {bh_final:>14,.0f}")
    print(f"B agent  净终值         {res.final_net():>14,.0f}"
          f"   alpha {(res.final_net()/bh_final - 1)*100:>+10.2f}%\n")

    # ---------- 杠杆后的关键指标 ----------
    mixed_log = ag.log["mixed_weight"]
    mixed = np.stack(mixed_log, axis=0) if len(mixed_log) > 1 else mixed_log[0][None, :]
    n_bars = mixed.shape[0]

    print(f"=== 杠杆后的权重范围（{n_bars:,} 根 bar）===")
    print(f"  max_exposure = {LEVERAGE}, max_gross = {LEVERAGE}")
    print(f"  min(w) 历史最低：{mixed.min():>+.4f}")
    print(f"  max(w) 历史最高：{mixed.max():>+.4f}")
    print(f"  min(Σ|w|) 历史最低：{np.abs(mixed).sum(axis=1).min():>+.4f}")
    print(f"  max(Σ|w|) 历史最高：{np.abs(mixed).sum(axis=1).max():>+.4f}"
          f"  （应 ≤ {LEVERAGE}）")
    gross_violation = int((np.abs(mixed).sum(axis=1) > LEVERAGE + 1e-6).sum())
    per_asset_violation = int((np.abs(mixed) > LEVERAGE + 1e-6).sum())
    print(f"  gross 超限 bar：{gross_violation}  单标的超限 bar：{per_asset_violation}\n")

    # ---------- 杠杆 vs 不杠杆（与 A' 段 1/2/3/4/5 对比） ----------
    print("=== 分段表现（每段的 gross exposure + 做空 bar 占比）===")
    print("注：C 阶段没做 → 杠杆下爆仓不会真发生；alpha 是 sim 估值，非实盘预期。\n")
    rt = np.diff(np.log(fr[syms[0]]["close"].to_numpy()))
    seg_len = len(mixed) // 5
    for i in range(5):
        seg_mixed = mixed[i * seg_len:(i + 1) * seg_len]
        if len(seg_mixed) == 0:
            continue
        ps = i * seg_len + 1
        pe = (i + 1) * seg_len + 1
        ret = float(np.exp(rt[ps:pe].sum()) - 1)
        active = int((seg_mixed.min(axis=1) < -1e-6).sum())
        max_gross_seg = float(np.abs(seg_mixed).sum(axis=1).max())
        print(f"  段 {i + 1}/5  区间收益 {ret*100:+7.2f}%  "
              f"做空 bar {active:>4}/{len(seg_mixed)} ({active/len(seg_mixed)*100:>5.1f}%)  "
              f"max gross {max_gross_seg:>5.2f}")

    # ---------- 专家终态 ----------
    print("\n=== 专家权重终态（按 p 降序，前 8）===")
    for k, v in sorted(zip(ag.expert_names(), ag.p), key=lambda kv: -kv[1])[:8]:
        bar = "█" * max(1, int(v * 50))
        print(f"  {k:<16}{v:>7.4f}  {bar}")

    # ---------- 落盘 ----------
    out_dir = os.path.join(store.project_root(), "runs", "b_leverage_baseline", "curves")
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
