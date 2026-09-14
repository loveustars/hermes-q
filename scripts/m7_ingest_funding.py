"""M7-1 资金费采集：拉取核心池的永续资金费历史。

只读公开行情（fapi/v1/fundingRate），不需要任何账号或 key。
产出 data/funding.csv，并打印各标的的资金费统计——这决定了做多的持仓成本量级。
"""
from __future__ import annotations

import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from src import config as cfgmod  # noqa: E402
from src.data import sources, store  # noqa: E402
from src.registry.runs import Run  # noqa: E402
from src.sim.funding import FundingTable, funding_path  # noqa: E402

START_MS = int(datetime(2019, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)


def main() -> None:
    cfg = cfgmod.load("base")
    cfgmod.guard_frozen(cfg)
    syms = cfg["universe"]["core"] + cfg["universe"]["reserve"]

    hypothesis = {
        "question": "永续资金费的年化量级有多大？覆盖率是否足以支撑回测？",
        "expected": "核心池年化 5%~10% 的持仓成本；覆盖率应接近 100%",
        "decision_rule": "若某标的覆盖率 < 90%，则该标的的资金费结论不可用，"
                         "需要在报告中标注而非强行使用",
    }

    with Run("m7_ingest_funding", cfg, hypothesis) as run:
        table = FundingTable()
        raw_dir = store.data_dir("raw")
        summaries, records = {}, []

        for s in syms:
            try:
                rows = sources.funding_rate_history(s, START_MS)
            except Exception as e:
                print(f"  [{s}] 拉取失败: {e}")
                continue
            for r in rows:
                table.add(s, r["funding_time"], r["funding_rate"])
            # 也落一份原始 CSV 便于审计
            p = os.path.join(raw_dir, f"binance_funding_{s}.csv")
            with open(p, "w", encoding="utf-8") as f:
                f.write("funding_time,mark_price,funding_rate,rate_type\n")
                for r in rows:
                    f.write(f"{r['funding_time']},{r['mark_price']},"
                            f"{r['funding_rate']},{r['rate_type']}\n")
            store.update_manifest(p, s, "funding", "binance_futures", len(rows))

            first = datetime.fromtimestamp(rows[0]["funding_time"] / 1000, tz=timezone.utc)
            last = datetime.fromtimestamp(rows[-1]["funding_time"] / 1000, tz=timezone.utc)
            st = table.summary(s)
            try:
                idx = store.load_bars(s, "1h").index
                cov = table.coverage(s, idx)
            except FileNotFoundError:
                cov = {"coverage": None}
            st.update(symbol=s, n_records=len(rows),
                      first=f"{first:%Y-%m-%d}", last=f"{last:%Y-%m-%d}",
                      coverage=cov.get("coverage"))
            summaries[s] = st
            print(f"  [{s}] {len(rows):>6,} 条  {st['first']} ~ {st['last']}  "
                  f"覆盖率 {cov.get('coverage')}", flush=True)

        out = table.save(funding_path("funding.csv"))
        print(f"\n已保存 -> {os.path.relpath(out, store.project_root())}")

        print("\n=== 资金费统计（正费率＝多头付出）===")
        print(f"{'标的':<10}{'条数':>8}{'均值/次':>12}{'年化均值':>11}"
              f"{'年化中位':>11}{'正费率占比':>11}{'最大':>11}{'最小':>11}")
        for s, st in summaries.items():
            print(f"{s:<10}{st['n']:>8,}{st['mean_per_settlement']*100:>11.4f}%"
                  f"{st['annualized_mean']*100:>10.2f}%{st['median_annualized']*100:>10.2f}%"
                  f"{st['positive_share']*100:>10.1f}%{st['max']*100:>10.4f}%"
                  f"{st['min']*100:>10.4f}%")

        run.log("funding_summary", summaries)
        run.record_metrics({
            "symbols": list(summaries),
            "annualized_mean": {s: v["annualized_mean"] for s, v in summaries.items()},
            "min_coverage": min((v["coverage"] or 0) for v in summaries.values()),
        })
        print(f"\nrun 目录: {os.path.relpath(run.dir, store.project_root())}")


if __name__ == "__main__":
    main()
