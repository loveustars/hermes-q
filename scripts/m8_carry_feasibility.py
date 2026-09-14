"""资金费 carry 可行性侦察 —— 先量有没有，再决定建不建策略。

两条路线的收益结构完全不同：
  (A) 纯永续按费率择方向：带方向敞口，carry 与价格风险混在一起。
  (B) 现货多 + 永续空（基差/cash-and-carry）：理论上方向中性，只收 carry。

本脚本只回答"结构上有没有钱"，先不计入我们的撮合器（那一步留给策略实现）：
  1. 连续持有空头收到的资金费（= 多头付出的那一笔，符号相反）
  2. 按资本占用折算 —— 这是最容易被高估的地方，必须显式假设
  3. 择时版本（只在正费率时持有）能加多少，以及切换成本吃掉多少
  4. 分年度稳定性 —— 只有年年为正才叫结构，某一年暴赚不算

资本假设（显式写出，便于质疑）：
  - 基差交易：现货腿全额占用 + 永续腿保证金。保证金按 100% 保守估，
    即总占用 = 2 倍名义额；另给出 50% 保证金（总占用 1.5 倍）的对照。
  - 单腿永续：只占用 1 倍名义额。
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src import config as cfgmod  # noqa: E402
from src.data import store  # noqa: E402
from src.registry.runs import Run  # noqa: E402
from src.sim.costs import CostModel  # noqa: E402
from src.sim.funding import load_default  # noqa: E402

SETTLEMENTS_PER_YEAR = 3 * 365
ROUND_TRIP_COST_BP = 22.0     # 单腿往返：手续费 20bp + 点差/冲击约 2bp（见 M2 实测）


def funding_series(ft, symbol, start_ms, end_ms):
    d = ft.rates_by_hour.get(symbol, {})
    s = pd.Series(d).sort_index()
    s.index = pd.to_datetime(s.index, unit="ms", utc=True)
    return s[(s.index >= pd.Timestamp(start_ms, unit="ms", tz="UTC")) &
             (s.index <= pd.Timestamp(end_ms, unit="ms", tz="UTC"))]


def carry_path(ser: pd.Series) -> pd.Series:
    """空头持有收到的资金费累计路径。

    符号约定：正费率下**多头付出、空头收取**，所以空头的现金流是 +rate。
    （第一版这里写成 -rate，把"收到"算成了"成本"，整个结论方向颠倒。）
    """
    return ser.cumsum()


def underwater_share(cum: pd.Series) -> float:
    """累计 carry 路径中处于历史高位以下的时段占比 —— carry 的"回撤时间"。"""
    if len(cum) == 0:
        return 0.0
    peak = cum.cummax()
    return float((cum < peak - 1e-12).mean())


def main() -> None:
    cfg = cfgmod.load("base")
    cfgmod.guard_frozen(cfg)
    syms = cfg["universe"]["core"]
    ft = load_default()
    if not ft.rates_by_hour:
        raise SystemExit("缺资金费数据，先跑 scripts/m7_ingest_funding.py")

    frames = {s: store.load_bars(s, "1h")[
        ["open", "high", "low", "close", "volume", "quote_volume"]] for s in syms}
    idx = None
    for f in frames.values():
        idx = f.index if idx is None else idx.intersection(f.index)
    frames = {s: f.loc[idx] for s, f in frames.items()}
    start_ms = max(min(d) for d in ft.rates_by_hour.values() if d)
    end_ms = int(idx[-1].timestamp() * 1000)
    years = (end_ms - start_ms) / 1000 / 86400 / 365
    print(f"窗口 {pd.Timestamp(start_ms, unit='ms', tz='UTC'):%Y-%m-%d} ~ "
          f"{pd.Timestamp(end_ms, unit='ms', tz='UTC'):%Y-%m-%d}  ({years:.2f} 年)")
    print("符号：正费率下多头付出、空头收取 ⇒ 空头的现金流是 +rate。")

    hypothesis = {
        "question": "资金费 carry 在真实数据里是否结构成立？折算到资本占用后还剩多少？",
        "expected": "BTC/ETH 连续做空应年年收到正 carry（正费率占比约 86%）；"
                    "按 2 倍资本占用折算后年化约 5%~7%",
        "decision_rule": "若折算到资本后低于无风险利率量级，或分年度出现大额负年，"
                         "则 carry 不值得建策略，转其他信号源",
    }

    with Run("m8_carry_feasibility", cfg, hypothesis) as run:
        rows = []
        print("\n=== 1. 连续持有空头（不择时）===")
        print(f"{'标的':<9}{'累计carry':>11}{'年化(名义)':>12}{'年化@2x资本':>13}"
              f"{'年化@1.5x':>11}{'负费率时段':>11}{'水下时间':>10}")
        series_cache = {}
        for s in syms:
            ser = funding_series(ft, s, start_ms, end_ms)
            series_cache[s] = ser
            cum = carry_path(ser)
            total_nom = float(cum.iloc[-1])
            ann_nom = total_nom / years
            ann_2x = ann_nom / 2.0
            ann_15x = ann_nom / 1.5
            neg = float((ser < 0).mean())
            uw = underwater_share(cum)
            rows.append({"symbol": s, "years": round(years, 3),
                         "cumulative_carry_on_notional": round(total_nom, 4),
                         "annualized_on_notional": round(ann_nom, 4),
                         "annualized_at_2x_capital": round(ann_2x, 4),
                         "annualized_at_1_5x_capital": round(ann_15x, 4),
                         "negative_rate_share": round(neg, 4),
                         "underwater_share": round(uw, 4)})
            print(f"{s:<9}{total_nom*100:>10.2f}%{ann_nom*100:>11.2f}%"
                  f"{ann_2x*100:>12.2f}%{ann_15x*100:>10.2f}%"
                  f"{neg*100:>10.1f}%{uw*100:>9.1f}%")

        print("\n=== 2. 分年度 carry：只有年年为正才叫结构 ===")
        yearly, pos_years = {}, {}
        for s in syms:
            ser = series_cache[s]
            by = ser.groupby(ser.index.year).sum()
            yearly[s] = {int(k): round(float(v), 4) for k, v in by.items()}
            pos_years[s] = f"{int((by > 0).sum())}/{len(by)}"
            print(f"  {s:<9}" + "  ".join(f"{k}:{v*100:>+7.2f}%" for k, v in by.items())
                  + f"   正年数 {pos_years[s]}")

        print("\n=== 3. 择时版本：只在费率 > 0 时持有 ===")
        print(f"  切换成本按单腿往返 {ROUND_TRIP_COST_BP:.0f}bp 计")
        tim = []
        base = {r["symbol"]: r["cumulative_carry_on_notional"] for r in rows}
        for s in syms:
            ser = series_cache[s]
            pos = (ser > 0)
            switches = int(pos.astype(int).diff().abs().fillna(0).sum())
            raw = float(ser[pos].sum())
            cost = switches * ROUND_TRIP_COST_BP / 1e4
            net_nom = raw - cost
            tim.append({"symbol": s, "raw_on_notional": round(raw, 4),
                        "switches": switches, "cost_on_notional": round(cost, 4),
                        "net_on_notional": round(net_nom, 4),
                        "net_annualized_2x": round(net_nom / years / 2, 4),
                        "delta_vs_always": round(net_nom - base[s], 4)})
            print(f"  {s:<9} 裸收 {raw*100:>7.2f}%  切换 {switches:>4} 次  "
                  f"成本 {cost*100:>7.2f}%  净 {net_nom*100:>7.2f}%  "
                  f"相对不择时 {net_nom - base[s]:+.2%}")

        print("\n=== 4. 判定 ===")
        best = max(rows, key=lambda r: r["annualized_at_2x_capital"])
        b = best["annualized_at_2x_capital"]
        sym = best["symbol"]
        print(f"  最优标的 {sym}：名义额年化 {best['annualized_on_notional']*100:.2f}%，"
              f"折算到 2x 资本年化 {b*100:.2f}%（1.5x 资本 "
              f"{best['annualized_at_1_5x_capital']*100:.2f}%）")
        print(f"  分年度为正：{pos_years[sym]}    水下时间占比 {best['underwater_share']*100:.1f}%")
        if b <= 0.05:
            print("  → 折算后不高于无风险利率量级，朴素 carry 不构成足够吸引的边际。")
        else:
            print("  → 折算后仍显著高于无风险利率量级，且方向中性，值得进入策略实现与正式评估。")
        print("  未计入的真实摩擦：跨市场资金划转、永续保证金机会成本、")
        print("    现货腿无法加杠杆而永续腿可加杠杆造成的资本结构差异、对手方与清算风险。")

        run.log("carry_always", rows)
        run.log("carry_yearly", yearly)
        run.log("carry_timed", tim)
        run.record_metrics({
            "window_years": round(years, 3),
            "best_symbol": sym,
            "best_annualized_on_notional": best["annualized_on_notional"],
            "best_annualized_at_2x_capital": b,
            "positive_years": pos_years,
            "timed_net_annualized_2x": {t["symbol"]: t["net_annualized_2x"] for t in tim},
        })
        print(f"\nrun 目录: {os.path.relpath(run.dir, store.project_root())}")


if __name__ == "__main__":
    main()
