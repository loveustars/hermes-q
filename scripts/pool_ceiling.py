"""专家池**上界**测量 —— 把"天花板在哪"从看不见变成一行数字。

**为什么需要它**（FRAMEWORK_AUDIT.md §3「如果只能做一件事」）

在知道上界之前，任何后续投入（调 η/band、写 RL/反思体、换成本模型）都无法判断
值不值得：它们都不会改变池子的上界。本脚本把上界量出来，并顺带把**强平通道**
的贡献与**保证金参数修正**的效果直接摆在同一张表上。

具体做三件事：

  (a) **池内事后最优**：把 `default_experts(syms)` 的 18 条规则**逐条单独**过引擎，
      按净终值与 Sharpe 两个口径分别给最优成员（含名次）。
  (b) **相对基准的差额**：每条规则与"**同 band 语义**的等权买入持有"的差
      （同引擎、同成本、同 band ⇒ 差额只归因于决策函数）。
  (c) **学习体的位置**：`HedgeEnsemble(η=0.20, band=0.20)` 落在池内第几名、分位多少。

**三套保证金口径同表对照**（这是本脚本与审计前文的接口）：

  - `off`            —— 关闭保证金（`margin=None`）
  - `on_k1overL`     —— `MarginConfig.for_leverage(1.0)` ⇒ `k = 1/L = 1.0`
  - `legacy_k0.5`    —— 修正前的硬编码参数 `k=0.5`（**已作废**，仅作对照）

在 `max_gross = 1.0` 下 `k=1` 意味着"多头没有借款 ⇒ 永不强平"，所以**只能做多的规则**
在 `on` 与 `off` 下的每一格都应当**完全相同**。脚本会按"是否提出过负权重"分类，
对多头候选逐格断言 —— 它是保证金参数修好了的运行时证据，也顺带证明"强平通道"在
1x 多头上的贡献**恰好为零**（修正前它高达 −94.6%：`only_BTCUSDT` 5,469 vs 101,251）。

**空头腿不同，且这是对的**：空头卖的是借来的资产，强平通道经济上本来就该存在。
`k=1/L` 下空头阈值是 `(1+1/L)/(1+m) = 1.818`（涨 82% 强平），与「1x 空头 + 10%
维持保证金」一致。所以含空的规则在 `on/off` 之间不一致是**预期行为**，
脚本把它单列成 `short_side_channel` 而不是当成缺陷。

**第三条缺陷（不是参数，本次未修）**：`exchange.py:258-269` 只在
「空→有仓 / 有仓→空 / 多空翻转」时同步保证金账本。仓位被**调大调小却不穿过零**
时，`margin_cash`/`entry_price` 仍是旧仓位的锚 ⇒ 权益式不再是 `|u_现|·p`，
于是连 `k=1`（本该永不强平）都会凭空触发强平。实测：只做多的 `inv_vol`
被强平，全窗口 `n_stale_anchor_bars = 228,591` 腿-bar。脚本把它单列成
`② 只做多但会调仓的规则`，并在 `pool_table` 里给出每条候选的
`stale_anchor_bars` —— **这些行的 on 臂数字是被污染的，上界请看 off 臂**。
（修它要动 `exchange.py`，不在本次改动范围；见 `src/sim/margin.py` 的
`n_stale_anchor_bars` 与 `tests/test_margin_leverage_semantics.py` 第 6 节。）

**资金费口径（2026-09-15 加的第二维）**：`SimExchange(..., funding=None)` 是默认值，
而仓库里 9 个调用点有 8 个没传 ⇒ `use_funding=False` ⇒ **引擎的资金费现金流恒为 0**，
只有 agent 内部按 funding 估成本（错位）。本脚本把 `fund_off / fund_on` 与
`margin off / on(k=1/L) / legacy` 交叉成 6 套口径同表报出。**★结论口径 =
`on_k1overL|fund_on`**（正确的保证金参数 + 引擎真实结算资金费），其余只作对照差异。

**band 语义**（务必与真专家一致）：`HedgeEnsemble` 是**开环**的 —— 它不读成交
回报、不看权益、不知道自己被强平，所以强平之后它会一直重发同一个目标、被 band
滤掉、**永不重入**。因此包一层与它相同的 `|Δw|₁ < band ⇒ return None` 才是同语义
的对照；否则"每 bar 重发目标"会在强平后立刻重入，得到的是另一个策略。

跑法：python3 scripts/pool_ceiling.py                 （全窗口，约 12~15 分钟）
      python3 scripts/pool_ceiling.py --no-legacy     （跳过已作废参数那套）
      python3 scripts/pool_ceiling.py --bars 20000    （快速冒烟，看流程不看结论）
"""
from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from src import config as cfgmod  # noqa: E402
from src.agents.online import HedgeEnsemble, default_experts  # noqa: E402
from src.data import store  # noqa: E402
from src.registry.runs import Run  # noqa: E402
from src.sim.costs import CostModel  # noqa: E402
from src.sim.exchange import SimConfig, SimExchange  # noqa: E402
from src.sim.funding import FundingTable  # noqa: E402
from src.sim.margin import MarginConfig  # noqa: E402

BARS_PER_YEAR = 24 * 365
MAX_GROSS = 1.0          # 与 D 阶段一致：池内专家与学习体都在 1x gross 上限下
BAND = 0.20              # 与 D 阶段最优组一致（也是学习体的 band）
ETA = 0.20               # 同上
MAINT = 0.1

# 铁律 5：假设与判定规则**在跑之前**写好，随 Run 的 meta.json 一起落盘。
HYPOTHESIS = {
    "claim": ("(1) 池内事后最优就是**等权买入持有**（池内的 `long_all`，其 weights "
              "恰为 1/n 等权 ⇒ 与同 band 语义的等权买入持有逐点相同），即"
              "「池子的上界 = 零智能」；(2) 学习体（η=0.20, band=0.20）**不在**池内"
              "前 30%；(3) 在 max_gross=1.0 且 k=1/L=1.0 下，**只能做多的规则**上"
              "margin ON 与 OFF 的每一格**完全相同**（1x 无借款 ⇒ 不该有任何强平"
              "通道）。"),
    "why": ("审计已给出三条前置事实：① 18 条规则里没有一条 carry/basis 规则，"
            "funding 只作为 payoff 项加在既有方向性规则上；② 学习体在 D 阶段"
            "收敛到 `long_all`（p_long_all≈0.96）但把权重摊在一堆 on/off 规则上"
            "（mean_gross≈0.65），于是既没拿满等权也不停付成本；③ 修正前旧参数"
            "给 1x 仓位强加了 −44.4% 的隐形停损，且 band 让它永不重入。"),
    "claim_3_scope": ("(3) 只对**多头**成立，这一点是跑之前就划定的：空头腿卖的是"
                      "借来的资产，强平通道在经济上**本来就该存在**"
                      "（k=1/L 下空头阈值 (1+1/L)/(1+m) = 1.818 ⇒ 涨 82% 强平，"
                      "与「1x 空头 + 10% 维持保证金」一致）。所以 on/off 在空头规则上"
                      "不一致是**预期行为**，不是缺陷；脚本按 `long_only` 分类断言。"),
    "claim_3_status": ("**跑全窗口之前已被冒烟跑（3,000 根）证伪过一次**（如实记录，不事后"
                       "改假设）：`inv_vol`（`iv/Σiv × gross`，恒正的只做多规则）在 k=1 下"
                       "**仍被强平**，终值 23,400 vs 关保证金的 29,516。机制不是参数，而是"
                       "**第三条缺陷**：主循环只在「空→有仓 / 有仓→空 / 多空翻转」时同步"
                       "保证金账本（`exchange.py:258-269`），仓位被**调大调小却不穿过零**时"
                       "`margin_cash`/`entry_price` 仍是旧仓位的锚 ⇒ 权益式 "
                       "`margin_cash + (p−p₀)·|u_现|` 不再等于 `|u_现|·p`，连 k=1 都会凭空"
                       "触发强平。全窗口实测 `n_stale_anchor_bars` 量级见 pool_table 的 "
                       "`stale_anchor_bars` 列（inv_vol 一条规则就有 22.9 万腿-bar）。"),
    "claim_4": ("(4) 「锚点过期」在池子里是**普遍状态**而不是个别边角：任何会在不穿零的前提下"
                "调整仓位的规则（`inv_vol`、以及几乎总在调权的 `HedgeEnsemble`）都受影响；"
                "而只交易一次的静态规则（`long_all` / `only_*` / 等权买入持有）不受影响"
                "（仓位从不变化 ⇒ 锚点始终有效），它们的 on/off 必然逐格相同。"),
    "claim_5_funding": ("(5) 引擎**真实结算资金费**（`SimExchange(funding=...)`）会显著压低"
                        "所有绝对水平，但**不翻转池内次序**；且**资金费的方向由平均净持仓"
                        "决定，不由「是否做空」决定** —— 学习体平均是净多头，所以它是净支付"
                        "方，方向与永远满仓的等权买入持有相同。"
                        "（委托人在本脚本之外已实测过同一件事并给出了数字：等权 ×0.56、"
                        "学习体 ×0.61，次序未翻转。本条是**独立复核**，不是新预注册："
                        "如实标注它晚于委托人的报告。）"),
    "falsified_by": ("任一条：出现某规则在**同 band 语义**下净终值显著高于等权买入"
                     "持有（池内真有 alpha，上界不是买入持有）；或学习体进入池内前 "
                     "30%；或**仓位从不变化的静态多头规则**在 on/off 两套之间出现差异"
                     "（那才是参数没修好的证据 —— 锚点过期不在这一类里，单列）。"),
    "decision_rule": ("（FRAMEWORK_AUDIT.md §3 预登记）若学习体不在池内前 30%，"
                      "**停止参数调整**，按 G2 转信号源研究（最直接的一条是把已验证的 "
                      "funding carry 做成专家，但需要引擎能表达双腿仓）。"),
    "registered": "2026-09-15",
    "note": ("本脚本的 (a) 是**事后**最优，有前视偏差；它的用途是当「天花板」，"
             "不是当目标。任何「学习体赢不了它」的解读都必须先声明这一点。"),
}


# ==========================================================================
# 与 HedgeEnsemble **同语义**的 band 包装
# ==========================================================================
class BandedRule:
    """恒定目标权重 + 与 `HedgeEnsemble` 相同的 band 过滤。

    第一根 bar 必发（`last_emitted is None`），此后 `|Δw|₁ < band` 一律 `None`。
    ⇒ 对静态规则（等权、只押一个标的）这就是"只交易一次"的买入持有。
    """

    def __init__(self, weights: dict, symbols: list[str], name: str,
                 band: float = BAND):
        self.weights = {s: float(weights.get(s, 0.0)) for s in symbols}
        self.symbols = list(symbols)
        self.name = name
        self.band = band
        self.last: np.ndarray | None = None
        self.min_weight_seen = 0.0
        self.n_emits = 0

    def decide(self, view):
        w = np.array([self.weights[s] for s in self.symbols], dtype=float)
        self.min_weight_seen = min(self.min_weight_seen, float(w.min()))
        if self.last is not None and float(np.abs(w - self.last).sum()) < self.band:
            return None
        self.last = w
        self.n_emits += 1
        return {s: float(x) for s, x in zip(self.symbols, w)}


class OneExpert:
    """把一条 `Expert` 单独包成 agent —— **band 语义必须与 `HedgeEnsemble` 一致**。

    真专家是开环的：不读成交、不看权益、不知道被强平。所以包一层
    `|Δw|₁ < band ⇒ None` 之后，强平过的规则会一直重发同一目标、被滤掉、
    **永不重入**（这正是旧参数下 5,469 vs 101,251 的差异来源）。用"每 bar 重发"
    的无 band 包装会得到另一个策略，不可比。
    """

    def __init__(self, expert, symbols: list[str], band: float = BAND,
                 gross: float = MAX_GROSS):
        self.expert = expert
        self.symbols = list(symbols)
        self.band = band
        self.gross = gross
        self.name = expert.name
        self.last: np.ndarray | None = None
        self.min_weight_seen = 0.0
        self.n_emits = 0

    def decide(self, view):
        w = np.asarray(self.expert.weights(view, self.symbols, self.gross),
                       dtype=float)
        self.min_weight_seen = min(self.min_weight_seen, float(w.min()))
        if self.last is not None and float(np.abs(w - self.last).sum()) < self.band:
            return None
        self.last = w
        self.n_emits += 1
        return {s: float(x) for s, x in zip(self.symbols, w)}


def holds_short(ag) -> bool:
    """该 agent 是否在回测中提出过负权重（⇒ 有借券腿，强平通道经济上成立）。

    真专家的 `min_weight_seen` 由上面的包装类记录；`HedgeEnsemble` 的记录在
    `ag.log["mixed_weight"]`（每根 bar 的混合权重）。
    """
    if hasattr(ag, "min_weight_seen"):
        return float(ag.min_weight_seen) < -1e-9
    mix = getattr(ag, "log", {}).get("mixed_weight")
    if mix:
        return any(float(np.min(w)) < -1e-9 for w in mix)
    return False


def is_static(ag) -> bool:
    """该 agent 是否**从不调整目标权重**（只发一次目标，之后持仓自然漂移）。

    静态规则（`long_all` / `only_*` / 等权买入持有）的仓位从不变化 ⇒
    保证金锚点始终有效 ⇒ 它们的 on/off **必须**逐格相同；这是"参数修好了"的
    干净检验。会调权的规则（`inv_vol` / 学习体）则受"锚点过期"这条独立缺陷影响。
    """
    if hasattr(ag, "n_emits"):
        return ag.n_emits <= 1
    tr = getattr(ag, "log", {}).get("turnover")
    if tr:
        return sum(1 for t in tr if t > 0) <= 1
    return False


# ==========================================================================
def summarize(res) -> dict:
    """一次回测 → 一行指标（净终值为主口径，毛终值与成本并列）。"""
    eq = res.net_equity
    geq = res.gross_equity
    out = {
        "final_net": float(eq[-1]) if len(eq) else 0.0,
        "final_gross": float(geq[-1]) if len(geq) else 0.0,
        "total_return": float(eq[-1] / 10_000.0 - 1.0) if len(eq) else 0.0,
        "cost": float(res.cost_paid.sum()),
        "turnover": float(res.turnover_notional.sum()),
        "n_trades": int(res.n_trades),
        "n_liquidated": len(res.liquidated_legs),
        # 引擎实际结算的资金费现金流（正 = 策略净支付给交易所）。
        # 引擎不传 funding 时恒为 0（见 SimConfig 与 use_funding 判定）。
        "funding_paid": (float(res.funding_paid.sum())
                         if len(res.funding_paid) else 0.0),
        "n_funding_events": int(res.n_funding_events),
        "bankrupt": bool(res.bankrupt),
        "bars_recorded": int(len(eq)),
        "sharpe": 0.0, "max_dd": 0.0,
    }
    if len(eq) > 2:
        r = np.diff(eq) / eq[:-1]
        r = r[np.isfinite(r)]
        if len(r) > 1 and np.std(r) > 0:
            out["sharpe"] = float(np.mean(r) / np.std(r) * np.sqrt(BARS_PER_YEAR))
        peak = np.maximum.accumulate(eq)
        safe = np.where(peak > 0, peak, 1.0)
        out["max_dd"] = float(np.min((eq - peak) / safe))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bars", type=int, default=None,
                    help="只用前 N 根（冒烟用；默认全窗口 77,503）")
    ap.add_argument("--no-legacy", action="store_true",
                    help="跳过已作废的 k=0.5 那一套（省 ~1/3 机时）")
    ap.add_argument("--no-learner", action="store_true",
                    help="跳过学习体（每套 72 秒）")
    ap.add_argument("--out", default="pool_ceiling", help="run 名字")
    args = ap.parse_args()

    cfg = cfgmod.load("base")
    syms = cfg["universe"]["core"]
    fr = {s: store.load_bars(s, "1h")[
        ["open", "high", "low", "close", "volume", "quote_volume"]] for s in syms}
    idx = None
    for f in fr.values():
        idx = f.index if idx is None else idx.intersection(f.index)
    idx = idx[:args.bars] if args.bars else idx
    fr = {s: f.loc[idx] for s, f in fr.items()}
    n = len(fr[syms[0]])
    print(f"窗口 {n:,} 根  {fr[syms[0]].index[0]:%Y-%m-%d} ~ "
          f"{fr[syms[0]].index[-1]:%Y-%m-%d}  标的 {list(syms)}")

    funding = FundingTable.load(
        os.path.join(store.project_root(), "data", "funding.csv"))
    n_funding_pts = len(funding.rates_by_hour.get(syms[0], {}))
    print(f"funding 数据 {n_funding_pts:,} 条（{syms[0]}）")
    rates = np.array([0.0012, 0.0013, 0.0014])
    net_cost = CostModel.from_config(cfg, enabled=True)
    gross_cost = CostModel(enabled=False)

    # ---- 保证金口径（k=1/L 由 MAX_GROSS 推出，不再手写常数）----
    margin_arms = [
        ("off", None),
        ("on_k1overL", MarginConfig.for_leverage(
            MAX_GROSS, maintenance_margin_ratio=MAINT, topup_trigger_ratio=0.5)),
    ]
    if not args.no_legacy:
        margin_arms.append(("legacy_k0.5", MarginConfig(
            initial_margin_ratio=0.5, maintenance_margin_ratio=MAINT,
            topup_trigger_ratio=0.5)))
    # ---- 资金费口径：引擎是否真的结算资金费 ----
    # 2026-09-15 加：此前 9 个 SimExchange 调用点里 8 个没传 `funding`，
    # `use_funding` 恒为 False ⇒ **引擎的资金费现金流恒为 0**，
    # 而 agent 内部仍按 funding 估计成本（错位）。两套都报，让这个缺口可见。
    funding_arms = [("fund_off", None), ("fund_on", funding)]
    arms = [(f"{mn}|{fn}", mcfg, fobj)
            for mn, mcfg in margin_arms for fn, fobj in funding_arms]
    # 结论口径（委托人指定）：正确保证金参数 + 引擎真实结算资金费
    PRIMARY, CONTROL = "on_k1overL|fund_on", "off|fund_off"

    # ---- 候选：18 条专家（逐条单独跑）+ 两个参照 + 学习体 ----
    experts = default_experts(syms)
    n_pool = len(experts)
    k_n = len(syms)

    def expert_factory(ex):
        return lambda: OneExpert(ex, syms)

    def ew_factory(band):
        return lambda: BandedRule({s: 1.0 / k_n for s in syms}, syms,
                                  "ew_buyhold_band" if band > 0 else "ew_rebalanced",
                                  band=band)

    cands: list[tuple[str, str, object]] = [
        (ex.name, "pool", expert_factory(ex)) for ex in experts
    ]
    cands.append(("ew_buyhold_band", "ref", ew_factory(BAND)))
    cands.append(("ew_rebalanced", "ref", ew_factory(0.0)))
    if not args.no_learner:
        cands.append(("hedge_eta0.20_band0.20", "learner",
                      lambda: HedgeEnsemble(syms, cost_rate=rates, eta=ETA,
                                            band=BAND, max_exposure=MAX_GROSS,
                                            funding=funding)))

    print(f"专家 {n_pool} 条 + 参照 2 条" +
          ("" if args.no_learner else " + 学习体 1 条"))
    print(f"口径矩阵 = 保证金 {[m for m, _ in margin_arms]} × 资金费 "
          f"{[f for f, _ in funding_arms]} = {len(arms)} 套")
    print(f"band={BAND}  max_gross={MAX_GROSS}  成本 net=on / gross=off")
    print(f"★ 结论口径 {PRIMARY}；对照口径 {CONTROL}\n")

    with Run(args.out, {
        "script": "scripts/pool_ceiling.py",
        "bars": n,
        "window": f"{fr[syms[0]].index[0]:%Y-%m-%d}~{fr[syms[0]].index[-1]:%Y-%m-%d}",
        "universe": list(syms),
        "engine": ("SimExchange(instrument=perp, allow_short=True, max_gross=1.0, "
                   "max_exposure_per_symbol=1.0, warmup=300, latency_bars=1)"),
        "band_semantics": "|Δw|₁ < 0.20 ⇒ None（与 HedgeEnsemble 相同；开环、强平后不重入）",
        "cost": "net=CostModel.from_config(cfg, enabled=True)；gross=CostModel(enabled=False)",
        "margin_arms": {
            name: (None if m is None else {
                "initial_margin_ratio": m.initial_margin_ratio,
                "maintenance_margin_ratio": m.maintenance_margin_ratio,
                "long_liquidation_ratio": m.long_liquidation_ratio(),
                "description": m.liquidation_description()})
            for name, m in margin_arms},
        "funding_arms": {
            "fund_off": "SimExchange(funding=None) ⇒ use_funding=False ⇒ 现金流恒为 0",
            "fund_on": (f"SimExchange(funding=FundingTable.load(data/funding.csv), "
                        f"{n_funding_pts} 个 8h 结算点/标的)")},
        "primary_arm": PRIMARY,
        "control_arm": CONTROL,
        "pool_size": n_pool,
        "learner": None if args.no_learner else {
            "class": "HedgeEnsemble", "eta": ETA, "band": BAND,
            "max_exposure": MAX_GROSS, "agent_funding": "data/funding.csv"},
    }, HYPOTHESIS) as run:
        rows: list[dict] = []
        t0 = time.time()
        for arm_name, margin_cfg, fund_obj in arms:
            desc = "" if margin_cfg is None else "  " + margin_cfg.liquidation_description()
            print(f"=== 口径 {arm_name}{desc} ===   "
                  f"（引擎结算资金费：{'是' if fund_obj is not None else '否'}）")
            for cname, kind, factory in cands:
                t1 = time.time()
                ag = factory()
                sim = SimExchange(
                    fr, gross_cost, net_cost,
                    SimConfig(initial_cash=10_000.0, warmup=300,
                              max_gross=MAX_GROSS,
                              max_exposure_per_symbol=MAX_GROSS,
                              margin=margin_cfg, allow_short=True,
                              instrument="perp"),
                    funding=fund_obj)
                res = sim.run(ag)
                mb = sim.margin_book
                row = {"arm": arm_name, "candidate": cname, "kind": kind,
                       "long_only": not holds_short(ag),
                       "static": is_static(ag),
                       "stale_anchor_bars": (0 if mb is None
                                             else int(mb.n_stale_anchor_bars)),
                       "n_liquidated_stale": (0 if mb is None
                                              else len(mb.liquidated_stale)),
                       **summarize(res), "seconds": round(time.time() - t1, 1)}
                rows.append(row)
                print(f"  {cname:<24} 净终值 {row['final_net']:>12,.0f}  "
                      f"毛终值 {row['final_gross']:>12,.0f}  "
                      f"sharpe {row['sharpe']:>6.3f}  回撤 {row['max_dd']*100:>6.1f}%  "
                      f"成本 {row['cost']:>10,.0f}  笔数 {row['n_trades']:>6}  "
                      f"强平 {row['n_liquidated']:>3}  "
                      f"{'多头' if row['long_only'] else '含空'}"
                      f"{'静态' if row['static'] else '调权'}"
                      f"  锚点过期 {row['stale_anchor_bars']:>7}"
                      f"  {row['seconds']:>5.1f}s")
            run.log(f"pool_table_partial_{arm_name}", rows)   # 每套落一次盘，超时不丢
            print(f"  （累计 {time.time() - t0:.0f}s，已落盘中间结果）\n")

        # ---------------- 汇总 ----------------
        def arm_rows(arm):
            return [r for r in rows if r["arm"] == arm]

        def rank_of(arm, metric, name):
            """名次与分位：只在**池内 18 条专家**里排名（name 可以是池外候选）。"""
            pool = [r for r in arm_rows(arm) if r["kind"] == "pool"]
            vals = sorted((r[metric] for r in pool), reverse=True)
            me = next(r[metric] for r in arm_rows(arm) if r["candidate"] == name)
            n_better = sum(1 for v in vals if v > me)
            return {"rank": 1 + n_better, "of": len(pool),
                    "beats_pct": (sum(1 for v in vals if v < me) / (len(pool) - 1) * 100.0)}

        report: dict = {"per_arm": {}, "on_else_off_identical": {}}
        arm_names = [a[0] for a in arms]
        print("=== (a) 池内事后最优（18 条专家；★=结论口径）===")
        for arm_name in arm_names:
            pool = [r for r in arm_rows(arm_name) if r["kind"] == "pool"]
            best_net = max(pool, key=lambda r: r["final_net"])
            best_shp = max(pool, key=lambda r: r["sharpe"])
            # 与"同 band 语义的等权买入持有"比较（基准也在池内：long_all 权重即 1/n）
            bench = next(r for r in arm_rows(arm_name)
                         if r["candidate"] == "ew_buyhold_band")
            beaten = [r["candidate"] for r in pool
                      if r["final_net"] > bench["final_net"] * 1.005]
            report["per_arm"][arm_name] = {
                "best_by_net": {k: best_net[k] for k in
                                ("candidate", "final_net", "sharpe", "max_dd")},
                "best_by_sharpe": {k: best_shp[k] for k in
                                   ("candidate", "final_net", "sharpe", "max_dd")},
                "benchmark_ew_buyhold": {
                    "final_net": bench["final_net"], "sharpe": bench["sharpe"],
                    "max_dd": bench["max_dd"], "cost": bench["cost"]},
                "n_pool_rules_beating_ew_buyhold": len(beaten),
                "rules_beating_ew_buyhold": beaten,
            }
            star = "★" if arm_name == PRIMARY else " "
            print(f" {star}[{arm_name:<22}] 净终值最优 {best_net['candidate']:<14} "
                  f"{best_net['final_net']:>12,.0f}    "
                  f"Sharpe 最优 {best_shp['candidate']:<14} {best_shp['sharpe']:>6.3f}")
            print(f"               等权买入持有（同 band） {bench['final_net']:>12,.0f}  "
                  f"sharpe {bench['sharpe']:>6.3f}    "
                  f"池内跑赢它的规则：{len(beaten)}/{len(pool)} 条 {beaten}")

        # ---------------- (b) 每条规则相对 et 基准的差额 ----------------
        print("\n=== (b) 相对「同 band 语义的等权买入持有」的差额（★ 与对照两套）===")
        deltas: dict[str, list] = {}
        for arm_name in arm_names:
            bench = next(r for r in arm_rows(arm_name)
                         if r["candidate"] == "ew_buyhold_band")
            lst = []
            for r in [x for x in arm_rows(arm_name) if x["kind"] == "pool"]:
                lst.append({"candidate": r["candidate"],
                            "delta_net": r["final_net"] - bench["final_net"],
                            "ratio": (r["final_net"] / bench["final_net"]
                                      if bench["final_net"] > 0 else float("nan")),
                            "delta_sharpe": r["sharpe"] - bench["sharpe"]})
            lst.sort(key=lambda d: -d["delta_net"])
            deltas[arm_name] = lst
        for arm_show in (PRIMARY, CONTROL):
            if arm_show not in deltas:
                continue
            print(f"  —— {arm_show} ——")
            for d in deltas[arm_show][:5]:
                print(f"     {d['candidate']:<24} Δ净终值 {d['delta_net']:>+13,.0f}  "
                      f"×{d['ratio']:>5.3f}  Δsharpe {d['delta_sharpe']:>+7.3f}")
        print("  …（全部 18 行 × 各口径见 run 产物 deltas_vs_ew_buyhold.json）")

        # ---------------- (c) 学习体的分位 ----------------
        learner_info = None
        if not args.no_learner:
            lname = "hedge_eta0.20_band0.20"
            print("\n=== (c) 学习体落在池内第几分位（★=结论口径）===")
            learner_info = {}
            for arm_name in arm_names:
                rn = rank_of(arm_name, "final_net", lname)
                rs = rank_of(arm_name, "sharpe", lname)
                row = next(r for r in arm_rows(arm_name) if r["candidate"] == lname)
                learner_info[arm_name] = {
                    "final_net": row["final_net"], "sharpe": row["sharpe"],
                    "funding_paid": row["funding_paid"],
                    "rank_by_net": rn, "rank_by_sharpe": rs}
                star = "★" if arm_name == PRIMARY else " "
                print(f" {star}[{arm_name:<22}] 净终值 {row['final_net']:>12,.0f} "
                      f"→ 第 {rn['rank']}/{rn['of']} 名（优于池内 {rn['beats_pct']:.0f}%）  "
                      f"Sharpe {row['sharpe']:>6.3f} → 第 {rs['rank']}/{rs['of']} 名"
                      f"（{rs['beats_pct']:.0f}%）  资金费 {row['funding_paid']:>+11,.0f}")
            report["learner"] = learner_info

        # ---------------- 资金费通道：引擎结算 vs 不结算 ----------------
        print("\n=== 资金费通道：引擎真实结算（fund_on）vs 不结算（fund_off）===")
        funding_channel: dict[str, dict] = {}
        for margin_name in [m for m, _ in margin_arms]:
            a = f"{margin_name}|fund_off"
            b = f"{margin_name}|fund_on"
            if a not in arm_names or b not in arm_names:
                continue
            per_cand = {}
            for r_off in arm_rows(a):
                r_on = next(r for r in arm_rows(b)
                            if r["candidate"] == r_off["candidate"])
                per_cand[r_off["candidate"]] = {
                    "net_no_settle": r_off["final_net"], "net_settle": r_on["final_net"],
                    "delta": r_on["final_net"] - r_off["final_net"],
                    "funding_paid": r_on["funding_paid"],
                    "n_funding_events": r_on["n_funding_events"]}
            funding_channel[margin_name] = per_cand
            print(f"  —— margin={margin_name} ——")
            show = [c for c in ("ew_buyhold_band", "ew_rebalanced", "long_all",
                                "only_BTCUSDT", "inv_vol", "mom_168",
                                "hedge_eta0.20_band0.20") if c in per_cand]
            for c in show:
                d = per_cand[c]
                ratio = (d["net_settle"] / d["net_no_settle"]
                         if d["net_no_settle"] else float("nan"))
                print(f"     {c:<24} {d['net_no_settle']:>12,.0f} → "
                      f"{d['net_settle']:>12,.0f}（×{ratio:>5.3f}）  资金费现金流 "
                      f"{d['funding_paid']:>+12,.0f}  结算点 {d['n_funding_events']:>7,}")
        report["funding_channel"] = funding_channel
        # 次序是否被翻转（委托人在实验外报告"未翻转"，这里独立复核）
        if not args.no_learner and arms:
            for margin_name in [m for m, _ in margin_arms]:
                a, b = f"{margin_name}|fund_off", f"{margin_name}|fund_on"
                if a not in arm_names:
                    continue
                ra = {r["candidate"]: r["final_net"] for r in arm_rows(a)}
                rb = {r["candidate"]: r["final_net"] for r in arm_rows(b)}
                order_a = sorted(ra, key=lambda c: -ra[c])
                order_b = sorted(rb, key=lambda c: -rb[c])
                same_order = order_a == order_b
                print(f"  次序（margin={margin_name}）：资金费结算前后"
                      f"{'完全相同' if same_order else '**发生翻转**'}"
                      f"；前 3 名 {order_a[:3]} → {order_b[:3]}")
                report.setdefault("funding_order_check", {})[margin_name] = {
                    "identical_order": bool(same_order),
                    "top3_no_settle": order_a[:3], "top3_settle": order_b[:3]}

        # ---------------- on == off 的逐格断言（按缺陷来源分类） ----------------
        for fund_name in [f for f, _ in funding_arms]:
            arm_off, arm_on = f"off|{fund_name}", f"on_k1overL|{fund_name}"
            if arm_on not in arm_names or arm_off not in arm_names:
                continue
            print(f"\n=== 保证金参数修正的运行时证据：{arm_on} vs {arm_off} ===")
            mism, resized, short_diffs, n_static = [], [], [], 0
            for r_off in arm_rows(arm_off):
                r_on = next(r for r in arm_rows(arm_on)
                            if r["candidate"] == r_off["candidate"])
                same = (abs(r_on["final_net"] - r_off["final_net"])
                        <= 1e-6 * max(1.0, abs(r_off["final_net"])))
                item = (r_off["candidate"], r_off["final_net"], r_on["final_net"],
                        r_on["n_liquidated"], r_on["n_liquidated_stale"],
                        r_off["stale_anchor_bars"])
                # 分类优先级：先看有没有空头腿（有 ⇒ 强平通道经济上成立），
                # 再看是不是"仓位从不变化"（那才是参数修正的干净检验），
                # 剩下的是"只做多但会调仓" ⇒ 锚点过期。初版把 short_all /
                # short_mom_168 这类**静态空头**规则归进了多头那一类，是错的。
                if not r_off["long_only"]:
                    if not same:
                        short_diffs.append(item)
                elif r_off["static"]:
                    n_static += 1
                    report["on_else_off_identical"].setdefault(
                        fund_name, {})[r_off["candidate"]] = bool(same)
                    if not same:
                        mism.append(item)
                elif not same:
                    resized.append(item)
            print(f"  ① 只能做多且仓位从不变化 {n_static} 条（参数修正的干净检验）："
                  f"不一致 {len(mism)} 条")
            if mism:
                print("     ❌ 静态多头规则在 on/off 间出现差异 ⇒ 参数修正没关掉强平通道")
                for c, a, b, nl, nls, sa in mism:
                    print(f"        {c}: off {a:,.0f} vs on {b:,.0f}（强平 {nl}）")
            else:
                print("     ✅ 全部逐格相同 ⇒ k=1/L 下静态多头**没有任何**强平通道，"
                      "贡献恰好为 0")
            if resized:
                resized.sort(key=lambda t: t[1] - t[2])
                print(f"  ② 只做多但**会调仓**的规则 {len(resized)} 条出现差异 —— "
                      f"**不是参数问题**，是第三条缺陷「锚点过期」"
                      f"（`exchange.py` 只在穿越零点时同步保证金账本）：")
                for c, a, b, nl, nls, sa in resized:
                    print(f"        {c:<22} off {a:>10,.0f} → on {b:>10,.0f}"
                          f"（Δ {b - a:>+10,.0f}，强平 {nl} 次，其中 {nls} 次发生在"
                          f"锚点过期状态）")
                print("     ⇒ on 臂上这些行的数字**被污染**，上界应看 off 臂或等修复后重测。")
            report.setdefault("stale_anchor_contamination",
                              {})[fund_name] = {
                c: {"off": a, "on": b, "n_liquidated_on": nl,
                    "n_liquidated_while_stale": nls, "stale_anchor_bars": sa}
                for c, a, b, nl, nls, sa in resized}
            if short_diffs:
                short_diffs.sort(key=lambda t: t[1] - t[2])
                print(f"  ③ 含空候选 {len(short_diffs)} 条在两套间有差异 —— **空头腿卖的是"
                      f"借来的资产，强平通道经济上成立**；其中「发生在锚点过期状态」的"
                      f"次数单列，供判断有多少是第三种缺陷贡献的：")
                for c, a, b, nl, nls, sa in short_diffs[:8]:
                    print(f"        {c:<22} off {a:>10,.0f} → on {b:>10,.0f}"
                          f"（Δ {b - a:>+10,.0f}，强平 {nl} 次，过期态 {nls} 次）")
            report.setdefault("short_side_channel", {})[fund_name] = {
                c: {"off": a, "on": b, "n_liquidated_on": nl,
                    "n_liquidated_while_stale": nls}
                for c, a, b, nl, nls, sa in short_diffs}
            leg_arm = f"legacy_k0.5|{fund_name}"
            if leg_arm in arm_names:
                worst = None
                for r_leg in arm_rows(leg_arm):
                    r_off = next(r for r in arm_rows(arm_off)
                                 if r["candidate"] == r_leg["candidate"])
                    d = r_leg["final_net"] - r_off["final_net"]
                    if worst is None or d < worst[1]:
                        worst = (r_leg["candidate"], d, r_off["final_net"],
                                 r_leg["long_only"])
                print(f"  对照（已作废参数 k=0.5，同资金费口径）：受损最重的候选 {worst[0]}"
                      f"（{'多头' if worst[3] else '含空'}），Δ {worst[1]:,.0f}"
                      f"（off {worst[2]:,.0f}）")
                report.setdefault("legacy_worst_damage", {})[fund_name] = {
                    "candidate": worst[0], "delta_net": worst[1],
                    "final_off": worst[2], "long_only": bool(worst[3])}

        # ---------------- 落盘 ----------------
        run.log("pool_table", rows)
        run.log("deltas_vs_ew_buyhold", deltas)
        run.record_metrics(report)
        rule = HYPOTHESIS["decision_rule"]
        if learner_info and PRIMARY in learner_info:
            r0 = learner_info[PRIMARY]["rank_by_net"]
            r_ctl = learner_info.get(CONTROL, {}).get("rank_by_net", r0)
            in_top30 = r0["rank"] <= 0.30 * r0["of"] + 1e-9
            run.note(
                f"★结论口径 {PRIMARY}：学习体净终值第 {r0['rank']}/{r0['of']} 名"
                f"（优于池内 {r0['beats_pct']:.0f}% 成员），"
                f"资金费现金流 {learner_info[PRIMARY]['funding_paid']:,.0f}；"
                f"对照口径 {CONTROL} 第 {r_ctl['rank']}/{r_ctl['of']} 名。"
                f"预登记判定：{'在' if in_top30 else '不在'}池内前 30% ⇒ "
                f"{'形式上不必停止调参' if in_top30 else '按 G2 停止参数调整、转信号源'}"
                f"（但池内上界 = 等权买入持有，名次本身不是价值证据）。")
            print(f"\n预登记判定（{rule}）")
            print(f"  → 学习体在 ★{PRIMARY} 口径下"
                  f"{'在' if in_top30 else '**不在**'}池内前 30%"
                  f"（第 {r0['rank']}/{r0['of']} 名）")
        if "funding_channel" in report and PRIMARY in arm_names:
            fc = report["funding_channel"]
            for margin_name, per in fc.items():
                for c in ("ew_buyhold_band", "hedge_eta0.20_band0.20"):
                    if c in per:
                        d = per[c]
                        run.note(
                            f"资金费缺口（margin={margin_name}，{c}）：引擎不结算 "
                            f"{d['net_no_settle']:,.0f} → 真实结算 {d['net_settle']:,.0f}"
                            f"（×{d['net_settle'] / d['net_no_settle']:.3f}），"
                            f"资金费现金流 {d['funding_paid']:+,.0f}"
                            f"（正 = 净支付），{d['n_funding_events']:,} 次结算。")
        run.note("池内最优为事后口径（含前视偏差），仅作上界使用，不可当作目标。")
        run.note("★ 结论口径 = 保证金 k=1/L（`MarginConfig.for_leverage(1.0)`）"
                 "+ 引擎真实结算资金费；其余口径只作对照差异。")
        print(f"\n证据目录：{run.dir}")

    print(f"\n总耗时 {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
