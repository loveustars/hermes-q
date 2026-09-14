"""M2 验收：撮合与成本模型 —— 复现冲击表，并画出扣费前/后的权益曲线。

验收标准（PLAN.md §2.1）：
  1. 用真实数据 + 实测点差，复现"BTC 上冲击占比很小、薄盘上冲击主导"的结论
  2. 对若干基线策略，给出毛/净两本账的对照，量化成本拖累
  3. 换手率与成本拖累的关系表

本脚本不训练任何智能体——目的只是先看清摩擦成本。
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
from src.data import store  # noqa: E402
from src.eval import metrics  # noqa: E402
from src.registry.runs import Run  # noqa: E402
from src.sim.costs import CostModel  # noqa: E402
from src.sim.exchange import SimConfig, SimExchange  # noqa: E402

BARS_PER_YEAR = 24 * 365
ORDER_USD = 10_000.0


def load_aligned(symbols: list[str], interval: str = "1h") -> dict[str, pd.DataFrame]:
    frames = {}
    for s in symbols:
        df = store.load_bars(s, interval)
        frames[s] = df[["open", "high", "low", "close", "volume", "quote_volume"]]
    idx = None
    for s, f in frames.items():
        idx = f.index if idx is None else idx.intersection(f.index)
    return {s: f.loc[idx] for s, f in frames.items()}


def impact_table(frames: dict[str, pd.DataFrame], spreads: dict,
                 sigma_window: int = 168, vol_window: int = 168) -> list[dict]:
    """用真实数据算出往返成本拆解，单位基点。"""
    out = []
    for s, f in frames.items():
        sigma = float(np.log(f["close"] / f["close"].shift()).dropna().std())
        v_med = float(f["quote_volume"].median())
        cm = CostModel(spread_bp={s: spreads.get(s, 1.0)}, use_bnb_discount=False)
        d = cm.round_trip_bp(s, ORDER_USD, sigma, v_med)
        d.update(symbol=s, sigma_pct=round(sigma * 100, 4),
                 vol_median_M=round(v_med / 1e6, 3),
                 participation_pct=round(ORDER_USD / v_med * 100, 4),
                 impact_share_pct=round(d["impact_bp"] / d["total_bp"] * 100, 1)
                 if d["total_bp"] else 0.0)
        out.append(d)
    return out


def ascii_curve(eq: np.ndarray, width: int = 72, height: int = 10, label: str = "") -> str:
    e = np.asarray(eq, dtype=float)
    if len(e) < 2:
        return f"{label}: 数据不足"
    idx = np.linspace(0, len(e) - 1, width).astype(int)
    y = e[idx]
    lo, hi = float(y.min()), float(y.max())
    span = hi - lo or 1.0
    rows = [[" "] * width for _ in range(height)]
    for x, v in enumerate(y):
        r = int((hi - v) / span * (height - 1))
        rows[r][x] = "*"
    lines = [f"{label}  min={lo:,.0f}  max={hi:,.0f}  final={e[-1]:,.0f}"]
    for r, row in enumerate(rows):
        val = hi - span * r / (height - 1)
        lines.append(f"{val:>10,.0f} |" + "".join(row))
    return "\n".join(lines)


def main() -> None:
    cfg = cfgmod.load("base")
    cfgmod.guard_frozen(cfg)
    syms = cfg["universe"]["core"]

    spreads_path = os.path.join(store.project_root(), "configs", "spreads.json")
    spreads = {}
    if os.path.exists(spreads_path):
        with open(spreads_path, encoding="utf-8") as f:
            spreads = json.load(f).get("spread_bp", {}) or {}
    spreads = {k: v for k, v in spreads.items() if v is not None}

    frames = load_aligned(syms, cfg["data"]["primary_interval"])
    n = len(frames[syms[0]])
    print(f"载入 {len(syms)} 个标的，对齐后 {n:,} 根小时线  "
          f"{frames[syms[0]].index[0]:%Y-%m-%d} ~ {frames[syms[0]].index[-1]:%Y-%m-%d}")

    hypothesis = {
        "question": "在核心池上，摩擦成本究竟吃掉多少收益？冲击占比是否如预期很小？",
        "expected": "手续费为成本主项；冲击占比 < 25%；高换手策略被成本显著侵蚀",
        "decision_rule": "若某基线策略换手后净收益相对毛收益损失超过 50%，则必须在强化信号中强制换手约束",
    }

    with Run("m2_cost_baseline", cfg, hypothesis) as run:
        # ---------- 1. 冲击表 ----------
        print("\n=== 1. 往返成本拆解（1 万 U 单次下单，费率 0.1%/边，实测点差）===")
        hdr = (f"{'标的':<10}{'σ(小时)':>10}{'成交额中位':>12}{'参与率':>10}"
               f"{'手续费':>10}{'点差':>9}{'冲击':>9}{'合计':>9}{'冲击占比':>10}")
        print(hdr)
        tbl = impact_table(frames, spreads)
        for r in tbl:
            print(f"{r['symbol']:<10}{r['sigma_pct']:>9.4f}%{r['vol_median_M']:>10,.1f}M"
                  f"{r['participation_pct']:>9.4f}%{r['fee_bp']:>9.1f}bp{r['spread_bp']:>8.1f}bp"
                  f"{r['impact_bp']:>8.2f}bp{r['total_bp']:>8.1f}bp{r['impact_share_pct']:>9.1f}%")
        run.log("impact_table", tbl)

        # ---------- 2. 基线策略毛/净对照 ----------
        cm_net = CostModel.from_config(cfg, enabled=True)
        cm_net.spread_bp = spreads
        cm_gross = CostModel.from_config(cfg, enabled=False)
        simcfg = SimConfig(initial_cash=10_000.0, warmup=300,
                           latency_bars=cfg["costs"]["latency_bars"])

        agents = [
            B.Cash(),
            B.SingleAssetBuyHold("BTCUSDT"),
            B.SingleAssetBuyHold("ETHUSDT"),
            B.SingleAssetBuyHold("BNBUSDT"),
            B.BuyHold(),
            B.Momentum("BTCUSDT", lookback=720, rebalance_every=24),
            B.RandomWeights(seed=0, rebalance_every=24),
            B.RandomWeights(seed=0, rebalance_every=1),
        ]

        print("\n=== 2. 基线策略：毛收益 vs 净收益（本金统一 10,000 USDT）===")
        rows, curves = [], {}
        hdr2 = (f"{'策略':<24}{'毛终值':>12}{'净终值':>12}{'成本':>9}{'换手/年':>9}"
                f"{'吃掉的毛利润':>13}{'毛Sharpe':>10}{'净Sharpe':>10}{'状态':>8}")
        print(hdr2)
        for ag in agents:
            res = SimExchange(frames, cm_gross, cm_net, simcfg).run(ag)
            g = metrics.summary(res.gross_equity, res.turnover_notional,
                                None, BARS_PER_YEAR, initial=res.initial_cash)
            nn = metrics.summary(res.net_equity, res.turnover_notional,
                                 res.cost_paid, BARS_PER_YEAR, initial=res.initial_cash)
            cmp = metrics.compare(g, nn)
            cmp["agent"] = ag.name
            cmp["gross"] = g
            cmp["net"] = nn
            cmp["insolvent_at"] = (str(res.index[res.insolvent_at])
                                   if res.insolvent_at is not None else None)
            rows.append(cmp)
            curves[ag.name] = pd.DataFrame({
                "dt": res.index, "net": res.net_equity, "gross": res.gross_equity,
                "cost": res.cost_paid, "turnover": res.turnover_notional,
            })
            status = "归零" if res.insolvent_at is not None else ""
            print(f"{ag.name:<24}{g['final_equity']:>12,.0f}{nn['final_equity']:>12,.0f}"
                  f"{nn.get('cost_as_pct_of_initial', 0):>8.1f}%"
                  f"{nn.get('turnover_roundtrips_per_year', 0):>9.1f}"
                  f"{cmp['share_of_gross_profit_eaten_pct']:>12.1f}%"
                  f"{g['sharpe']:>10.2f}{nn['sharpe']:>10.2f}{status:>8}")
            if cmp["insolvent_at"]:
                print(f"{'':<24}  ↑ 净账本在 {cmp['insolvent_at']} 归零")
        run.log("baseline_compare", rows)

        # ---------- 3. 权益曲线落盘 ----------
        cdir = os.path.join(run.dir, "curves")
        os.makedirs(cdir, exist_ok=True)
        for name, df in curves.items():
            df.to_csv(os.path.join(cdir, f"{name}.csv"), index=False)

        # ---------- 4. 换手率 ↔ 成本拖累 ----------
        print("\n=== 3. 换手率与成本拖累（理论值，往返 0.2%）===")
        theory = []
        for n_rt in [10, 25, 50, 100, 200, 500, 1000]:
            drag = n_rt * 0.002
            theory.append({"roundtrips_per_year": n_rt,
                           "cost_drag_pct_per_year": round(drag * 100, 1)})
            print(f"  年换手 {n_rt:>5} 次往返 -> 成本拖累 {drag*100:>6.1f}%/年"
                  f"   （若目标年化 20%，则需毛收益 {20 + drag*100:>6.1f}%）")
        run.log("turnover_drag_theory", theory)

        # ---------- 5. ASCII 权益曲线 ----------
        print("\n=== 4. 权益曲线（毛 vs 净）===")
        for name in ["bh_BNBUSDT", "bh_BTCUSDT", "random_rb24_s0", "random_rb1_s0"]:
            if name in curves:
                c = curves[name]
                print()
                print(ascii_curve(c["gross"].to_numpy(), label=f"[毛] {name}"))
                print(ascii_curve(c["net"].to_numpy(), label=f"[净] {name}"))

        run.record_metrics({
            "impact_table": tbl,
            "agents": [r["agent"] for r in rows],
            "worst_profit_eaten_pct": max(r["share_of_gross_profit_eaten_pct"] for r in rows),
        })
        print(f"\nrun 目录: {os.path.relpath(run.dir, store.project_root())}")
        print(f"曲线 CSV: {os.path.relpath(cdir, store.project_root())}/")


if __name__ == "__main__":
    main()
