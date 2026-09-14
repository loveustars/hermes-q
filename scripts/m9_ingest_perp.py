"""M9-1 永续 K 线采集 + 基差分析。

carry 侦察（M8）只算了资金费现金流，明确声明忽略了 basis P&L。
这里把它补上，并量化 carry 空头腿的关键风险：

  1. 基差水平与波动（永续价相对现货价的偏离）
  2. **二阶 basis P&L**：固定单位数的「现货多 + 永续空」在整段窗口的漂移损益
  3. **空头腿的挤空风险**：永续价的滚动最大上涨幅度 —— 这决定了保证金要留多少
     （2021-05、2022-11 那种暴涨是 carry 空头腿的典型杀手）

只读公开行情（fapi/v1/klines），不需要任何账号或 key。
"""
from __future__ import annotations

import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src import config as cfgmod  # noqa: E402
from src.data import sources, store  # noqa: E402
from src.registry.runs import Run  # noqa: E402
from src.sim.funding import load_default  # noqa: E402

START_MS = int(datetime(2019, 8, 1, tzinfo=timezone.utc).timestamp() * 1000)


def ingest(symbol: str, interval: str = "1h") -> dict:
    bars = sources.perp_klines_parallel(symbol, interval=interval, start_ms=START_MS)
    if not bars:
        return {"symbol": symbol, "rows": 0, "error": "空数据"}
    path = store.save_bars(bars, symbol, interval, "binanceperp")
    df = store.load_bars(symbol, interval, source="binanceperp")
    h = store.health(df)
    h.update(symbol=symbol, interval=interval, path=os.path.relpath(path, store.project_root()))
    return h


def main() -> None:
    cfg = cfgmod.load("base")
    cfgmod.guard_frozen(cfg)
    syms = cfg["universe"]["core"]

    hypothesis = {
        "question": "基差（永续 vs 现货）的二阶损益有多大？空头腿需要多少保证金才不会被挤爆？",
        "expected": "基差在 0 附近小幅波动；二阶 P&L 相对资金费收入可忽略；"
                    "但存在足以爆掉低保证金空头的极端上冲",
        "decision_rule": "若二阶 P&L 与资金费收入同量级，则 M8 的结论不成立，需重做；"
                         "若滚动最大上冲超过保证金的某个倍数，必须在策略里显式建模强平",
    }

    with Run("m9_ingest_perp", cfg, hypothesis) as run:
        print("=== 1. 采集永续 1h K 线 ===")
        info = {}
        for s in syms:
            h = ingest(s)
            info[s] = h
            print(f"  [{s}] {h.get('rows', 0):,} 根  {h.get('start')} ~ {h.get('end')}"
                  f"  缺口={h.get('gaps')}", flush=True)

        ft = load_default()
        fund_start = pd.Timestamp(max(min(d) for d in ft.rates_by_hour.values() if d),
                                  unit="ms", tz="UTC")
        print(f"\n资金费公共起点 {fund_start:%Y-%m-%d}（基差与 carry 分析的窗口起点）")

        print("\n=== 2. 基差水平（永续 − 现货)/现货，基点）===")
        print(f"{'标的':<9}{'中位':>9}{'均值':>9}{'标准差':>9}{'p1':>9}"
              f"{'p99':>9}{'最大':>9}{'最小':>9}{'contango占比':>13}")
        basis_rows = {}
        aligned = {}
        for s in syms:
            spot = store.load_bars(s, "1h")
            perp = store.load_bars(s, "1h", source="binanceperp")
            idx = spot.index.intersection(perp.index)
            idx = idx[idx >= fund_start]
            a = pd.DataFrame({"spot": spot.loc[idx, "close"],
                              "perp": perp.loc[idx, "close"]})
            a["basis_bp"] = (a["perp"] - a["spot"]) / a["spot"] * 1e4
            aligned[s] = a
            b = a["basis_bp"]
            basis_rows[s] = {
                "bars": int(len(a)), "median_bp": round(float(b.median()), 3),
                "mean_bp": round(float(b.mean()), 3), "std_bp": round(float(b.std()), 3),
                "p1_bp": round(float(b.quantile(0.01)), 3),
                "p99_bp": round(float(b.quantile(0.99)), 3),
                "max_bp": round(float(b.max()), 3), "min_bp": round(float(b.min()), 3),
                "contango_share": round(float((b > 0).mean()), 4)}
            print(f"{s:<9}{b.median():>9.2f}{b.mean():>9.2f}{b.std():>9.2f}"
                  f"{b.quantile(0.01):>9.2f}{b.quantile(0.99):>9.2f}"
                  f"{b.max():>9.1f}{b.min():>9.1f}{(b > 0).mean()*100:>12.1f}%")

        print("\n=== 3. 二阶 basis P&L（固定单位数「现货多 + 永续空」，本金 10,000）===")
        print("  这是 M8 明确声明忽略掉的那一项。")
        print(f"{'标的':<9}{'年数':>7}{'基差漂移P&L':>14}{'年化':>10}"
              f"{'同期资金费':>13}{'比值':>9}")
        basis_pnl = {}
        for s in syms:
            a = aligned[s]
            years = len(a) / (24 * 365)
            N = 10_000.0
            su = N / a["spot"].iloc[0]          # 现货多头单位数（固定）
            pu = -N / a["perp"].iloc[0]         # 永续空头单位数（固定）
            pnl = su * (a["spot"].iloc[-1] - a["spot"].iloc[0]) + \
                pu * (a["perp"].iloc[-1] - a["perp"].iloc[0])
            ser = pd.Series(ft.rates_by_hour.get(s, {})).sort_index()
            ser.index = pd.to_datetime(ser.index, unit="ms", utc=True)
            ser = ser[ser.index >= fund_start]
            fund_income = float(ser.sum()) * N
            ratio = abs(pnl) / fund_income if fund_income else float("inf")
            basis_pnl[s] = {"years": round(years, 2), "basis_pnl": round(float(pnl), 2),
                            "annualized": round(float(pnl) / years / N, 6),
                            "funding_income": round(fund_income, 2),
                            "ratio_abs_pnl_over_funding": round(float(ratio), 4)}
            print(f"{s:<9}{years:>7.2f}{pnl:>13,.0f}{pnl/years/N*100:>9.2f}%"
                  f"{fund_income:>13,.0f}{ratio:>9.4f}")

        print("\n=== 4. 空头腿挤空风险：永续价的滚动最大上冲 ===")
        print("  空头腿的亏损随价格上涨而扩大，这个数决定了保证金要留多少。")
        print(f"{'标的':<9}{'全期最大上冲':>14}{'7日最大':>11}{'30日最大':>11}"
              f"{'90日最大':>11}")
        spike = {}
        for s in syms:
            px = aligned[s]["perp"]
            rec = {}
            for label, win in [("max_up_all", None), ("7d", 24 * 7),
                               ("30d", 24 * 30), ("90d", 24 * 90)]:
                if win is None:
                    run_max = float((px / px.cummin() - 1).max())
                else:
                    run_max = float((px / px.rolling(win, min_periods=2).min() - 1).max())
                rec[label] = round(run_max, 4)
            spike[s] = rec
            print(f"{s:<9}{rec['max_up_all']*100:>13.1f}%{rec['7d']*100:>10.1f}%"
                  f"{rec['30d']*100:>10.1f}%{rec['90d']*100:>10.1f}%")
        print("  → 以 100% 保证金（2x 资本）的 carry 结构为例：现货腿全额占用不动，")
        print("     永续腿保证金承受上冲。上冲超过保证金率即触发强平。")

        print("\n=== 5. 保证金需求 ⇒ 折算到资本后的真实收益 ===")
        print("  这是 M8 那个 6.80% 最脆弱的地方：它假设保证金只需 100%。")
        print("  结构：资本 = 现货腿 N + 永续腿保证金 m×N，资金费收入按 N 计。")
        print("  所以资本回报率 = 名义额回报率 / (1 + m)。")
        print(f"\n{'标的':<9}{'名义额年化':>12}{'m=1.0':>10}{'m=1.35':>10}"
              f"{'m=2.7':>10}{'需 m 撑过30日':>15}{'需 m 撑过90日':>15}")
        m_rows = {}
        for s in syms:
            ser = pd.Series(ft.rates_by_hour.get(s, {})).sort_index()
            ser.index = pd.to_datetime(ser.index, unit="ms", utc=True)
            ser = ser[ser.index >= fund_start]
            years = len(aligned[s]) / (24 * 365)
            ann = float(ser.sum()) / years
            need_30 = spike[s]["30d"]
            need_90 = spike[s]["90d"]
            m_rows[s] = {
                "annualized_on_notional": round(ann, 6),
                "return_at_m_1_0": round(ann / 2.0, 6),
                "return_at_m_1_35": round(ann / 2.35, 6),
                "return_at_m_2_7": round(ann / 3.7, 6),
                "required_m_for_30d": round(need_30, 4),
                "required_m_for_90d": round(need_90, 4),
                "return_at_required_m_30d": round(ann / (1 + need_30), 6),
                "return_at_required_m_90d": round(ann / (1 + need_90), 6),
            }
            print(f"{s:<9}{ann*100:>11.2f}%{ann/2*100:>9.2f}%{ann/2.35*100:>9.2f}%"
                  f"{ann/3.7*100:>9.2f}%{need_30*100:>14.1f}%{need_90*100:>14.1f}%")
        print("\n  按各标的**实际所需**保证金折算后的资本回报率：")
        for s in syms:
            r30 = m_rows[s]["return_at_required_m_30d"]
            r90 = m_rows[s]["return_at_required_m_90d"]
            print(f"    {s:<9} 撑过 30 日窗口需 m={m_rows[s]['required_m_for_30d']:.2f}"
                  f" → 年化 {r30*100:>5.2f}%   "
                  f"撑过 90 日需 m={m_rows[s]['required_m_for_90d']:.2f}"
                  f" → 年化 {r90*100:>5.2f}%")
        print("\n  → **保证金需求把 6.80% 压到 4%~6% 甚至更低，取决于再平衡频率与补保频率。**")
        print("  → 更关键的是：现货腿的盈利**不能**自动补到永续腿的保证金里（普通账户），")
        print("     所以价格不动方向也能被单独爆掉永续腿。这是 carry 的第一号风险，不是方向风险。")

        run.log("perp_ingest", info)
        run.log("basis_stats", basis_rows)
        run.log("basis_pnl", basis_pnl)
        run.log("short_squeeze", spike)
        run.log("margin_requirement", m_rows)
        run.record_metrics({"basis_stats": basis_rows, "basis_pnl": basis_pnl,
                            "short_squeeze": spike, "margin_requirement": m_rows})
        print(f"\nrun 目录: {os.path.relpath(run.dir, store.project_root())}")


if __name__ == "__main__":
    main()
