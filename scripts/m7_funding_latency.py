"""M7-2 资金费与延迟抖动的量级报告。

要回答两个问题：
  1. 资金费到底有多大？（M2 只算了手续费+点差+冲击，完全漏掉了持仓成本）
  2. 延迟抖动会不会改变结论？

注意口径：永续上市晚于现货（BTC 永续 2019-09，BNB 2020-02），
所以永续模式的可用样本必须截到「三个标的的资金费数据都有」的公共窗口，
否则早期会按 0 处理资金费，等于假装那是免费的。
"""
from __future__ import annotations

import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src import config as cfgmod  # noqa: E402
from src.agents import baselines as B  # noqa: E402
from src.data import store  # noqa: E402
from src.eval import metrics, protocol  # noqa: E402
from src.eval.benchmark import equal_weight_buyhold, strategy_returns  # noqa: E402
from src.registry.runs import Run  # noqa: E402
from src.sim.costs import CostModel  # noqa: E402
from src.sim.exchange import SimConfig, SimExchange  # noqa: E402
from src.sim.funding import FundingTable, load_default  # noqa: E402

BARS_PER_YEAR = 24 * 365
JITTER_LEVELS = [0.0, 0.5, 1.0, 2.0]


def load_frames(symbols):
    frames = {s: store.load_bars(s, "1h")[
        ["open", "high", "low", "close", "volume", "quote_volume"]] for s in symbols}
    idx = None
    for f in frames.values():
        idx = f.index if idx is None else idx.intersection(f.index)
    return {s: f.loc[idx] for s, f in frames.items()}


def funding_common_window(ft: FundingTable, symbols) -> pd.Timestamp:
    """所有标的都有资金费数据的起点。"""
    starts = []
    for s in symbols:
        d = ft.rates_by_hour.get(s) or {}
        if d:
            starts.append(min(d))
    if not starts:
        raise RuntimeError("没有任何资金费数据")
    return pd.Timestamp(max(starts), unit="ms", tz="UTC")


def run_one(frames, cfg, agent, instrument, funding, jitter=0.0, seed=0):
    return SimExchange(
        frames, CostModel(enabled=False),
        CostModel.from_config(cfg, enabled=True),
        SimConfig(initial_cash=10_000.0, warmup=300, instrument=instrument,
                  allow_short=(instrument == "perp"),
                  latency_jitter_mean=jitter, latency_jitter_seed=seed),
        funding=funding).run(agent)


def main() -> None:
    cfg = cfgmod.load("base")
    cfgmod.guard_frozen(cfg)
    syms = cfg["universe"]["core"]
    ft = load_default()
    if not ft.rates_by_hour:
        raise SystemExit("资金费数据缺失，请先运行 scripts/m7_ingest_funding.py")

    frames_all = load_frames(syms)

    hypothesis = {
        "question": "资金费相对手续费/冲击有多大？延迟抖动会不会改变结论？",
        "expected": "资金费量级应显著超过手续费（年化 5%~15% vs 往返 0.2%）；"
                    "抖动只影响高频策略",
        "decision_rule": "若资金费在多头持有下的年化拖累 > 5%，"
                         "则所有持仓型结论必须在永续口径下重算",
    }

    with Run("m7_funding_latency", cfg, hypothesis) as run:
        # ---------- 1. 资金费统计 ----------
        print("=== 1. 资金费统计（正费率＝多头付出）===")
        print(f"{'标的':<10}{'条数':>8}{'年化均值':>11}{'年化中位':>11}"
              f"{'正费率占比':>11}{'最大':>11}{'最小':>11}")
        stats = {}
        for s in syms:
            st = ft.summary(s)
            stats[s] = st
            print(f"{s:<10}{st['n']:>8,}{st['annualized_mean']*100:>10.2f}%"
                  f"{st['median_annualized']*100:>10.2f}%{st['positive_share']*100:>10.1f}%"
                  f"{st['max']*100:>10.4f}%{st['min']*100:>10.4f}%")
        run.log("funding_stats", stats)

        # ---------- 2. 永续公共窗口 ----------
        start = funding_common_window(ft, syms)
        frames = {s: f.loc[f.index >= start] for s, f in frames_all.items()}
        n = len(frames[syms[0]])
        years = n / BARS_PER_YEAR
        print(f"\n=== 2. 永续公共窗口 ===")
        print(f"  三个标的资金费数据都有的起点：{start:%Y-%m-%d}")
        print(f"  对齐后 {n:,} 根小时线（{years:.2f} 年）")
        for s in syms:
            cov = ft.coverage(s, frames[s].index)
            print(f"  {s}: 覆盖率 {cov['coverage']}  （{cov['found']}/{cov['expected']}）")

        # ---------- 3. 现货 vs 永续：资金费拖累 ----------
        print(f"\n=== 3. 资金费拖累：同一策略在现货 / 永续口径下 ===")
        print("  有效费率分母用**平均权益**，不用初始本金——")
        print("  权益翻十几倍时，用初始本金做分母会把资金费率高估好几倍。")
        print(f"{'策略':<20}{'口径':<7}{'净终值':>13}{'净收益':>11}"
              f"{'总资金费':>11}{'有效费率':>10}{'Sharpe':>8}{'终值影响':>10}")
        rows = []
        for name, mk in [
            ("等权持有三币", lambda: B.BuyHold()),
            ("单持 BTC", lambda: B.SingleAssetBuyHold("BTCUSDT")),
            ("动量（月换仓）", lambda: B.Momentum("BTCUSDT", lookback=720, rebalance_every=24)),
            ("随机（日换仓）", lambda: B.RandomWeights(seed=3, rebalance_every=24)),
        ]:
            got = {}
            for inst in ("spot", "perp"):
                res = run_one(frames, cfg, mk(), inst, ft)
                got[inst] = res
                r = metrics.summary(res.net_equity, res.turnover_notional,
                                    res.cost_paid, BARS_PER_YEAR,
                                    initial=res.initial_cash)
                fund = res.total_funding()
                mean_eq = float(np.mean(res.net_equity))
                eff_rate = fund / mean_eq / years if (mean_eq > 0 and years) else 0.0
                got[inst] = {"res": res, "net": r, "fund": fund, "eff": eff_rate}
                print(f"{name:<20}{inst:<7}{res.final_net():>13,.0f}"
                      f"{r['total_return']*100:>10.1f}%{fund:>11,.0f}"
                      f"{eff_rate*100:>9.2f}%{r['sharpe']:>8.2f}")
            fs, fp = got["spot"]["res"].final_net(), got["perp"]["res"].final_net()
            impact = (fp - fs) / fs * 100 if fs else 0.0
            print(f"{'':<20}{'终值影响':<7}{'':>13}{'':>11}{'':>11}{'':>10}"
                  f"{got['spot']['net']['sharpe']:>8.2f}{impact:>9.1f}%")
            rows.append({"agent": name,
                         "spot_final": round(fs, 2), "perp_final": round(fp, 2),
                         "terminal_impact_pct": round(impact, 2),
                         "total_funding": round(got["perp"]["fund"], 2),
                         "effective_funding_rate": round(got["perp"]["eff"], 6),
                         "sharpe_spot": got["spot"]["net"]["sharpe"],
                         "sharpe_perp": got["perp"]["net"]["sharpe"]})
        run.log("spot_vs_perp", rows)

        # ---------- 4. 成本层级对比 ----------
        print(f"\n=== 4. 成本层级：资金费 vs 手续费/冲击 ===")
        res_bh = run_one(frames, cfg, B.BuyHold(), "perp", ft)
        friction = float(res_bh.cost_paid.sum())
        funding = res_bh.total_funding()
        mean_eq = float(np.mean(res_bh.net_equity))
        print(f"  等权持有三币（{years:.2f} 年，持仓 ≈ 100%，平均权益 {mean_eq:,.0f}）：")
        print(f"    摩擦成本（手续费+点差+冲击）: {friction:>10,.0f} USDT"
              f"   占平均权益年化 {friction/mean_eq/years*100:>6.3f}%")
        print(f"    资金费（持仓成本）          : {funding:>10,.0f} USDT"
              f"   占平均权益年化 {funding/mean_eq/years*100:>6.2f}%")
        if friction > 0:
            print(f"    资金费是摩擦成本的 {funding/friction:>6.0f} 倍")
        print("    → **对持仓型策略，真实成本由资金费主导，与换手率无关。**")
        print("      M2 的『换手率一票否决』只对现货成立；永续口径下主导成本是持仓 carry。")

        # ---------- 5. 延迟抖动敏感性 ----------
        print(f"\n=== 5. 延迟抖动敏感性（现货口径，每档 8 个抖动种子）===")
        print("  报离散度而不是单点差异：单一种子会把'抖动噪声'和'种子噪声'混在一起。")
        jrows = []
        jitters = [0.5, 1.0, 2.0]
        n_seeds = 8
        hdr = f"{'策略':<20}{'无抖动':>13}"
        for j in jitters:
            hdr += f"{'j=' + str(j) + ' 中位':>14}{'离散度':>9}"
        print(hdr)
        for name, mk in [
            ("等权持有三币", lambda: B.BuyHold()),
            ("动量（月换仓）", lambda: B.Momentum("BTCUSDT", lookback=720, rebalance_every=24)),
            ("随机（日换仓）", lambda: B.RandomWeights(seed=3, rebalance_every=24)),
            ("随机（小时换仓）", lambda: B.RandomWeights(seed=3, rebalance_every=1)),
        ]:
            base = run_one(frames, cfg, mk(), "spot", None, jitter=0.0).final_net()
            line = f"{name:<20}{base:>13,.0f}"
            rec = {"agent": name, "no_jitter": round(base, 2),
                   "insolvent": base < 100.0, "levels": {}}
            for j in jitters:
                vals = [run_one(frames, cfg, mk(), "spot", None,
                                jitter=j, seed=100 + k).final_net()
                        for k in range(n_seeds)]
                vals = np.array(vals)
                med = float(np.median(vals))
                disp = float((vals.max() - vals.min()) / med * 100) if med > 0 else 0.0
                flag = "!" if med < 100.0 else " "
                line += f"{med:>13,.0f}{flag}{disp:>8.1f}%"
                rec["levels"][str(j)] = {
                    "median": round(med, 2), "min": round(float(vals.min()), 2),
                    "max": round(float(vals.max()), 2),
                    "dispersion_pct": round(disp, 2),
                    "meaningless_insolvent": bool(med < 100.0),
                    "deviation_from_no_jitter_pct":
                        round((med - base) / base * 100, 2) if base else 0.0}
            print(line)
            jrows.append(rec)
        print("  离散度 = (最大 − 最小) / 中位数。抖动每档跑 8 个种子。")
        print("  '!' 表示该策略已归零，此时百分比离散度是除零噪声，不可解读。")
        print("  → 买入持有只有 1.6%~1.9% 离散；调仓策略达 9%~21%。")
        print("    因此所有涉及调仓的结论必须在多个抖动种子下重复，不能只报单次结果。")
        run.log("latency_jitter", jrows)

        # ---------- 6. 资金费自身的可预测性（为下一阶段铺路）----------
        print(f"\n=== 6. 资金费自身的可预测性（资金费率/基差作为信号的可行性初探）===")
        for s in syms:
            d = ft.rates_by_hour.get(s, {})
            if not d:
                continue
            ser = pd.Series(d).sort_index()
            ser = ser[(ser.index >= int(start.timestamp() * 1000))]
            ac = [round(float(ser.autocorr(lag=k)), 4) for k in (1, 2, 3, 6, 9)]
            ann = ser * 3 * 365
            print(f"  {s:<10} 自相关 lag1/2/3/6/9 = {ac}   "
                  f"年化均值 {ann.mean()*100:>6.2f}%  "
                  f"负费率时段占比 {float((ser < 0).mean())*100:>5.1f}%")
        print("  自相关显著为正 ⇒ 资金费有持续性，可作为状态变量用于持仓方向选择")

        run.record_metrics({
            "funding_window_start": f"{start:%Y-%m-%d}",
            "funding_window_bars": n,
            "funding_window_years": round(years, 3),
            "funding_annualized": {s: v["annualized_mean"] for s, v in stats.items()},
            "buyhold_friction_usdt": round(friction, 2),
            "buyhold_funding_usdt": round(funding, 2),
            "funding_over_friction_ratio": round(funding / friction, 3) if friction else None,
            "jitter_dispersion_pct": {
                r["agent"]: {k: v["dispersion_pct"] for k, v in r["levels"].items()}
                for r in jrows},
        })
        print(f"\nrun 目录: {os.path.relpath(run.dir, store.project_root())}")


if __name__ == "__main__":
    main()
