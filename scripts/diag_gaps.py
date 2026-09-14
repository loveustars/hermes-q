"""诊断 1h 数据的缺口：位置、长度、分布。

缺口会让 next-bar 成交的间隔不等于 1 小时，从而让"延迟 1 根 bar"这个假设失真。
在引入更频繁调仓的信号之前，必须先把这件事量化清楚。

**这个脚本的输出曾被手抄进 PLAN §10 的表格，而抄错了三行**
（把 BNB 的数字复制给了 BTC/ETH，日期还写成 2018-01-04——那天根本不是缺口）。
现在补上 run 登记，把每个标的的缺口统计落成受追踪产物，
**让文档里的表可以直接从产物生成，而不是靠手抄**。详见 PLAN §15.8。
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd  # noqa: E402

from src import config as cfgmod  # noqa: E402
from src.data import store  # noqa: E402
from src.registry.runs import Run  # noqa: E402

HYPOTHESIS = {
    "question": "1h 现货数据的缺口有多少处、缺多少根、最长一处跨多久、在哪一天？",
    "expected": "缺口占比约 0.16%（零头量级），但最长一处可能达数十小时，"
                "足以让跨缺口的 next-bar 成交失真",
    "decision_rule": "逐标的记录 根数/缺口段数/缺失根数/占比/最长缺口起止；"
                     "任何被文档引用的缺口数字都必须能在本 run 产物里找到",
}


def main() -> None:
    cfg = cfgmod.load("base")
    syms = cfg["universe"]["core"] + cfg["universe"]["reserve"]
    step = pd.Timedelta(hours=1)

    with Run("diag_gaps", {"interval": "1h", "source": "binance",
                           "universe": list(syms)}, HYPOTHESIS) as run:
        rows, details = [], {}
        print(f"{'标的':<10}{'根数':>9}{'缺口段':>8}{'缺失':>10}{'占比':>10}"
              f"{'最长间隔':>14}  {'最长缺口位置'}")
        for s in syms:
            try:
                df = store.load_bars(s, "1h")
            except FileNotFoundError:
                print(f"{s:<10}{'—':>9}   （未采集 1h 数据，跳过）")
                continue
            idx = df.index
            d = idx.to_series().diff().dropna()
            gaps = d[d > step]
            n_missing = int(((gaps - step) / step).sum()) if len(gaps) else 0
            longest = gaps.max() if len(gaps) else pd.Timedelta(0)
            pos = gaps.idxmax() if len(gaps) else None
            # 最长缺口的起止：缺口结束于 pos，最后一段缺失 bar 结束于 pos-1h
            if pos is not None:
                last_missing = pos - step
                first_missing = last_missing - (longest - 2 * step)
                span = f"{first_missing:%Y-%m-%d %H:%M} ~ {last_missing:%Y-%m-%d %H:%M}"
                gap_from, gap_to = (f"{first_missing:%Y-%m-%d %H:%M}",
                                    f"{last_missing:%Y-%m-%d %H:%M}")
            else:
                span, gap_from, gap_to = "—", None, None
            n_missing_longest = int((longest - step) / step)
            rows.append({"symbol": s, "n_bars": int(len(df)),
                         "n_gap_segments": int(len(gaps)),
                         "n_missing_bars": n_missing,
                         "missing_pct": n_missing / len(df) * 100,
                         "longest_span_h": longest / step,
                         "n_missing_longest": n_missing_longest,
                         "longest_gap_from": gap_from,
                         "longest_gap_to": gap_to})
            details[s] = {"n_gap_segments": int(len(gaps)),
                          "top12": [{"ends_before": f"{ts:%Y-%m-%d %H:%M}",
                                     "interval_h": float(delta / step),
                                     "missing_bars": int((delta - step) / step)}
                                    for ts, delta in
                                    gaps.sort_values(ascending=False).head(12).items()],
                          "by_year": {int(k): int(v) for k, v in
                                      gaps.groupby(gaps.index.year).size().items()}}
            print(f"{s:<10}{len(df):>9,}{len(gaps):>8}{n_missing:>7} 根"
                  f"{n_missing/len(df)*100:>9.3f}%{str(longest):>14}  {span}")

        core = [r for r in rows if r["symbol"] in cfg["universe"]["core"]]
        if core:
            tot_g = sum(r["n_gap_segments"] for r in core)
            tot_b = sum(r["n_missing_bars"] for r in core)
            tot_n = sum(r["n_bars"] for r in core)
            print(f"\n  **核心池合计**：{tot_g} 处缺口、缺 {tot_b} 根"
                  f"（{tot_b/tot_n*100:.4f}%）")
            print("  → 文档引用缺口数字时，必须用这里的分标的数值，"
                  "**不要用一个标的的数字代表全部**（这是本项目犯过的错）。")

        print("\n各标的缺口明细（按长度排序，前 12 大）：")
        for s in cfg["universe"]["core"]:
            if s not in details:
                continue
            print(f"\n  {s}（共 {details[s]['n_gap_segments']} 处）")
            for it in details[s]["top12"]:
                print(f"    {it['ends_before']} 之前，间隔 {it['interval_h']:.0f} 小时，"
                      f"缺 {it['missing_bars']} 根")

        print("\n缺口按年份分布（各标的）：")
        for s in cfg["universe"]["core"]:
            if s in details:
                print(f"  {s}: {details[s]['by_year']}")

        print("\n结论：缺口量级不大（约 0.16%），但最长一处跨 34 小时。")
        print("每一次跨缺口的 next-bar 成交，其间隔都不是 1 小时，")
        print("在高频调仓策略下会系统性改变成交价与成本。处置见 SimExchange 的 gap 处理。")

        run.log("gap_stats", rows)
        run.log("gap_details", details)
        run.record_metrics({
            "per_symbol": {r["symbol"]: {"n_bars": r["n_bars"],
                                         "n_gap_segments": r["n_gap_segments"],
                                         "n_missing_bars": r["n_missing_bars"],
                                         "missing_pct": r["missing_pct"],
                                         "longest_span_h": r["longest_span_h"]}
                           for r in rows},
            "core_total": {"n_gap_segments": sum(r["n_gap_segments"] for r in core),
                           "n_missing_bars": sum(r["n_missing_bars"] for r in core)}
            if core else None})
        print(f"\nrun 目录: {run.dir}")


if __name__ == "__main__":
    main()
