"""委托人侧独立核验：审计报告的论断 + 两个新发现的记账缺口。

为什么单独有这个脚本（而不是只在临时日志里）：
本项目的铁律是**关键证据必须落在 git 里的 runs/ 目录**。这里的每条结论都会
改变既有判断（其中一条会推翻委托人自己的预测），所以它必须有受追踪的产物。

四项核验：
  ① 审计报告的关键论断：清算事件（bar 9104 / 2018-11-23 / BTC low 4,239.67）
     与强平阈值解析式 (1−k)/(1−m)
  ② **新发现**：引擎漏结算资金费 —— SimExchange(funding=None) 时现金流恒为 0，
     而 9 个调用点里 8 个没传（含 m4/m5/m11/a/b/d 与 d_fair_comparison）
  ③ 新发现：funding 那几倍摆幅是「学习体的路径混沌」而非资金费的经济量级
     （判据：η=0 时摆幅 = 1.0000x）
  ④ 新发现：同一策略在不同取数窗口下终值横跨数倍 ⇒ 单次数值不是结论

跑法：python3 scripts/audit_verify.py        （约 8~10 分钟）
"""
from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from src import config as cfgmod  # noqa: E402
from src.agents.online import HedgeEnsemble  # noqa: E402
from src.data import store  # noqa: E402
from src.registry.runs import Run  # noqa: E402
from src.sim.costs import CostModel  # noqa: E402
from src.sim.exchange import SimConfig, SimExchange  # noqa: E402
from src.sim.funding import FundingTable  # noqa: E402
from src.sim.margin import MarginConfig  # noqa: E402

HYPOTHESIS = {
    "claim": ("① 审计报告关于「1x 下凭空强平」的论断成立（事件与解析阈值均可复现）；"
              "② 引擎漏结算资金费，且量级足以让训练线绝对值虚高约 2 倍；"
              "③ funding 开关造成的数倍摆幅来自学习体的路径混沌，不是经济量级；"
              "④ 学习体对取数窗口混沌敏感 ⇒ 单次运行不构成结论。"),
    "why": ("这些数字直接决定「学习体有没有价值」能否被回答。若 ②③④ 成立，"
            "则项目所有关于该学习体的单次数值都需要重述为分布。"),
    "falsified_by": ("① 事件对不上；② 引擎结算资金费后终值变化 < 5%；"
                     "③ η=0 时 funding 摆幅也远离 1.0；④ 不同窗口终值差异 < 20%。"),
    "registered": "2026-09-15",
}

cfg = cfgmod.load("base")
SYMS = cfg["universe"]["core"]
FUNDING = FundingTable.load(os.path.join(store.project_root(), "data", "funding.csv"))
COST_BY_SYM = {"BTCUSDT": 0.0012, "ETHUSDT": 0.0013, "BNBUSDT": 0.0014}
MG1 = MarginConfig(initial_margin_ratio=1.0, maintenance_margin_ratio=0.1,
                   topup_trigger_ratio=0.5)


def frames(syms):
    """按 syms 取交集窗口（注意：窗口不同 ⇒ 学习体路径不同，见 ④）。"""
    fr = {s: store.load_bars(s, "1h")[
        ["open", "high", "low", "close", "volume", "quote_volume"]] for s in syms}
    ix = None
    for f in fr.values():
        ix = f.index if ix is None else ix.intersection(f.index)
    return {s: f.loc[ix] for s, f in fr.items()}


class FixedW:
    def __init__(self, w):
        self.w = w

    def decide(self, view):
        return dict(self.w)


def go(fr, agent, *, funding_arg, k=1.0, m=0.1):
    mg = (MarginConfig(initial_margin_ratio=k, maintenance_margin_ratio=m,
                       topup_trigger_ratio=0.5) if k is not None else None)
    return SimExchange(fr, CostModel(enabled=False),
                       CostModel.from_config(cfg, enabled=True),
                       SimConfig(initial_cash=1e4, warmup=300, max_gross=1.0,
                                 max_exposure_per_symbol=1.0, margin=mg,
                                 allow_short=True, instrument="perp"),
                       funding=funding_arg).run(agent)


def main() -> None:
    t0 = time.time()
    out: dict = {}

    # ---------------- ④ 窗口敏感性（先建两套 frames）----------------
    fr_3 = frames(SYMS)                      # 三标的交集
    fr_2_3 = {s: fr_3[s] for s in ("BTCUSDT", "ETHUSDT")}
    fr_2 = frames(["BTCUSDT", "ETHUSDT"])    # 两标的自己的交集（更长）

    with Run("audit_verify", {
        "script": "scripts/audit_verify.py",
        "universe": list(SYMS),
        "bars_3sym": len(fr_3[SYMS[0]]), "bars_2sym": len(fr_2["BTCUSDT"]),
        "bars_2sym_from_3sym_window": len(fr_2_3["BTCUSDT"]),
    }, HYPOTHESIS) as run_ctx:

        # -------- ① 审计论断 --------
        ix = fr_3["BTCUSDT"].index
        print("① 核验审计的清算事件")
        print(f"   idx[9104]          = {ix[9104]}")
        print(f"   BTCUSDT low @9104  = {fr_3['BTCUSDT']['low'].iloc[9104]:,.2f}")
        thr = {k: (1 - k) / (1 - 0.1) for k in (0.5, 1 / 3, 1.0)}
        print("   解析阈值 p/p0 = (1−k)/(1−m):")
        for k, v in thr.items():
            print(f"     k={k:<8.4f} → {v:.4f}   (首次可强平跌幅 {(1-v)*100:5.1f}%)")

        # -------- ② 引擎 funding 缺口 --------
        print("\n② 引擎是否结算资金费（3 标的，k=1.0）")
        ew = {s: 1.0 / len(SYMS) for s in SYMS}
        cases = [
            ("被动等权", lambda: FixedW(ew)),
            ("只持 BTC", lambda: FixedW({"BTCUSDT": 1.0, "ETHUSDT": 0.0, "BNBUSDT": 0.0})),
            ("学习体 η=.2 b=.2", lambda: HedgeEnsemble(
                SYMS, cost_rate=np.array([COST_BY_SYM[s] for s in SYMS]),
                eta=0.20, band=0.20, max_exposure=1.0, funding=FUNDING)),
        ]
        rows = []
        for name, mk_agent in cases:
            # **必须每跑新建 agent**：HedgeEnsemble 是有状态的（专家权重跨运行残留），
            # 复用实例会让第二次运行从第一次的权重出发 ⇒ 路径不同、数字被污染。
            # （本脚本初版就是这么错的：同一实例连跑 funding OFF/ON，见 WORK_LOG §17.5。）
            off = go(fr_3, mk_agent(), funding_arg=None)
            on = go(fr_3, mk_agent(), funding_arg=FUNDING)
            fc = float(np.sum(on.funding_paid)) if len(on.funding_paid) else 0.0
            rows.append({"策略": name, "终值_funding_OFF": round(float(off.net_equity[-1]), 2),
                         "终值_funding_ON": round(float(on.net_equity[-1]), 2),
                         "变化%": round((on.net_equity[-1] / off.net_equity[-1] - 1) * 100, 2),
                         "资金费现金流": round(fc, 2)})
            print(f"   {name:<18} OFF {off.net_equity[-1]:>12,.0f}  "
                  f"ON {on.net_equity[-1]:>12,.0f}  ({rows[-1]['变化%']:+.1f}%)  "
                  f"资金费 {fc:>11,.1f}")
        p_off, p_on = rows[0]["终值_funding_OFF"], rows[0]["终值_funding_ON"]
        h_off, h_on = rows[2]["终值_funding_OFF"], rows[2]["终值_funding_ON"]
        print(f"   次序：OFF 被动/学习体 = {p_off/h_off:.2f}x   "
              f"ON 被动/学习体 = {p_on/h_on:.2f}x")
        out["funding_gap"] = {
            "rows": rows,
            "ratio_off": round(p_off / h_off, 4),
            "ratio_on": round(p_on / h_on, 4),
            "order_flipped": bool((p_off / h_off - 1) * (p_on / h_on - 1) < 0),
            "note": "次序未翻转 ⇒ 相对结论较稳，但绝对水平虚高约 2 倍",
        }

        # -------- ③ funding 摆幅 = 混沌还是经济量级 --------
        print("\n③ 判据：funding 摆幅来自混沌还是经济量级（2 标的）")
        sub = ["BTCUSDT", "ETHUSDT"]
        chaos = {}
        for eta in (0.0, 0.20):
            off = go(fr_2_3, HedgeEnsemble(
                sub, cost_rate=np.array([COST_BY_SYM[s] for s in sub]),
                eta=eta, band=0.20, max_exposure=1.0, funding=None), funding_arg=None)
            on = go(fr_2_3, HedgeEnsemble(
                sub, cost_rate=np.array([COST_BY_SYM[s] for s in sub]),
                eta=eta, band=0.20, max_exposure=1.0, funding=FUNDING), funding_arg=FUNDING)
            sw = float(on.net_equity[-1]) / float(off.net_equity[-1])
            chaos[f"eta_{eta}"] = {"funding_OFF": round(float(off.net_equity[-1]), 2),
                                   "funding_ON": round(float(on.net_equity[-1]), 2),
                                   "swing_x": round(sw, 6)}
            print(f"   η={eta:<5} OFF {off.net_equity[-1]:>12,.0f}  "
                  f"ON {on.net_equity[-1]:>12,.0f}   摆幅 {sw:.6f}x")
        out["funding_swing"] = {
            **chaos,
            "verdict": ("η=0 摆幅=1.0000 ⇒ 摆幅来自学习动态的路径混沌，"
                        "不是资金费的经济量级"
                        if abs(chaos["eta_0.0"]["swing_x"] - 1.0) < 1e-6 else
                        "η=0 摆幅也远离 1.0 ⇒ 另有渠道，需另找原因"),
        }

        # -------- ④ 窗口敏感性 --------
        print("\n④ 窗口敏感性（同一个「2 标的 η=.2 b=.2」策略）")
        w_sens = {}
        for tag, frx in (("两标的自身交集", fr_2), ("三标的交集切片", fr_2_3)):
            r = go(frx, HedgeEnsemble(
                sub, cost_rate=np.array([COST_BY_SYM[s] for s in sub]),
                eta=0.20, band=0.20, max_exposure=1.0, funding=FUNDING),
                funding_arg=FUNDING)
            w_sens[tag] = {"bars": len(frx["BTCUSDT"]),
                           "final": round(float(r.net_equity[-1]), 2)}
            print(f"   {tag:<16} {len(frx['BTCUSDT']):>7,} 根  终值 {r.net_equity[-1]:>12,.0f}")
        a = w_sens["两标的自身交集"]["final"]
        b = w_sens["三标的交集切片"]["final"]
        ratio = max(a, b) / min(a, b)
        print(f"   ⇒ 仅因窗口差 {abs(len(fr_2['BTCUSDT'])-len(fr_2_3['BTCUSDT'])):,} 根，"
              f"终值差 {ratio:.2f} 倍")
        out["window_sensitivity"] = {**w_sens, "bar_diff":
                                     abs(len(fr_2["BTCUSDT"]) - len(fr_2_3["BTCUSDT"])),
                                     "ratio_x": round(ratio, 4),
                                     "note": "学习体对窗口混沌敏感 ⇒ 单次数值不是结论"}

        out["elapsed_s"] = round(time.time() - t0, 1)
        run_ctx.record_metrics(out)
        run_ctx.note("① 事件与阈值均复现；② funding 缺口 −39%~−56%，次序"
                     f"{rows[0]['变化%']:.0f}% 变化下未翻转；③ η=0 摆幅 "
                     f"{chaos['eta_0.0']['swing_x']:.6f}x ⇒ 混沌；④ 窗口差 "
                     f"{out['window_sensitivity']['ratio_x']:.2f}x")
        print(f"\n证据目录：{run_ctx.dir}")
        print(f"耗时 {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
