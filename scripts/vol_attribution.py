"""波动口径的收益归因层 —— 回答「这些标的的波动里能不能提取正期望」。

## 为什么需要这一层

项目至今的标尺是「终值倍数」，而终值倍数完全由**对照物的选择**决定。同一个
学习体（η=0.20 band=0.20）实测得到过 2.45x / 1.55x / 7.61x / 4.86x 四个不同
结论，只因为对照物换了。倍数回答不了「输在哪儿」，于是每次换对照物就换一个
结论。本脚本把标尺从「倍数」换成**波动/β 口径**：

    不是「追上了多少倍」，而是「在同等市场暴露下，它多拿了还是少拿了」。

## 指标定义（全部按 1h 口径年化，BARS_PER_YEAR = 8760）

记号：`E` = 某次 `SimExchange.run` 的净权益曲线（首元素在 warmup+latency 之后，
所以一律以**初始本金 10,000** 为基数，不用 `E[0]`）；`r_t = E_t/E_{t-1} - 1`。

1. **基准 BM1 = 同池等权「每 bar 再平衡」**
   `FixedWeights({s: 1/k})` + `band=0`（每根 bar 都重发同一目标）+ 同一台
   `SimExchange` + **同一成本/资金费设置** + 与候选**同一保证金档**。
   同一台引擎、同一成本模型、同一时间轴，只换决策函数与 band。

2. **β / α / R²**（numpy 手写 OLS，无 scipy）
   OLS：`r_c = α + β·r_bm + ε`
       β   = Cov(r_c, r_bm)/Var(r_bm)（= OLS 斜率，带截距）
       α   = mean(r_c) − β·mean(r_bm)                      （每 bar 口径）
       α_ann = expm1(8760 · log1p(α))                       （年化，指数复合）
       R²  = 1 − SSR/SST
   显著性用 **Newey–West（HAC, lags=24）三明治标准误**：
   `V = (X'X)^{-1} S (X'X)^{-1}`，`S = Σ_l κ_l Σ_t w_t w_{t-l}'`，`w_t = X_t·u_t`，
   `κ_l = 1 − l/(L+1)`。`t_α = α̂ / sqrt(V[0,0])`。同时报普通 OLS 标准误的 t 作对照。
   **基准对自己必须 β=1、α=0、R²=1** —— 脚本里有自检并写进 run 目录。

3. **波动归一口径**
       ann_vol   = std(r_c, ddof=1) · sqrt(8760)
       sharpe    = mean(r_c)/std(r_c) · sqrt(8760)          （项目既有约定，rf=0）
       CAGR      = expm1( log(E_T / 10000) / (T/8760) )
       vol_norm  = CAGR / ann_vol   ←「vol 归一后收益」：每单位年化波动拿到的年化收益
       max_dd    = min(E / cummax(E) − 1)，基数含 10,000

4. **β 匹配对照（β-matched passive）**
   把基准**缩放到与该候选相同的 β**，再比较：
       E_β(t) = 10000 · Π_{s<=t} (1 + β · r_bm,s)
   这是「一个同等市场暴露、无技术含量、无摩擦」的被动持有。意义：
   **「它到底有没有超越一个同等暴露的被动持有」，而不是「它有没有追上 BNB」。**
   β_g 由「候选的无摩擦（毛账本）路径」对「基准毛账本」回归得到 ——
   毛账本既不计交易成本也不计资金费（`exchange.py:14-15` 明写资金费只进净账本），
   所以 β_g 是纯暴露量，不被摩擦污染。（同时报 β_n：实际净路径对基准净路径的 β。）

5. **四块归因分解**（对数空间，**严格可加**）

   需要同一候选的三次仿真（见 ENGINE_VARIANTS）：
       A = 保证金 k1 + **引擎结算资金费**          ← 一切结论的基准
       B = 保证金 OFF + 引擎结算资金费              ← 只关强平通道
       C = 保证金 OFF + 引擎不结算资金费            ← 再关资金费
   每次都产出净/毛两个账本（同一次遍历，`exchange.py` 的双账本设计）。

   定义 `G_total := log(E_c^net,A) − log(E_β_g^gross)`，则

       G_total = G_exp + G_trade + G_fund + G_liq

       G_exp   = log(E_c^gross,C) − log(E_β_g^gross)     ← 敞口缺口（暴露路径 vs 同 β 恒定暴露）
       G_trade = log(E_c^net,C)   − log(E_c^gross,C)     ← 交易成本（手续费+点差+冲击）
       G_fund  = log(E_c^net,B)   − log(E_c^net,C)       ← 资金费（永续持有成本）
       G_liq   = log(E_c^net,A)   − log(E_c^net,B)       ← 强平通道

   四项**逐项相消**，和恒等于 `G_total`（脚本内断言 `|Σ − G_total| < 1e-9`）。
   解读：`G_exp>0` 暴露路径胜过同 β 恒定暴露（有择时价值）；`G_trade<0` 摩擦；
   `G_fund<0` 持有成本；`G_liq<0` 强平通道在伤害它。
   分母口径差异另有一项 `G_bcost`（被动的自身成本×β），**不属于策略差异**，单列。

   **为什么资金费必须单列**：`M7` 阶段测过资金费是其它摩擦成本合计的 2,627 倍；
   把它并进一个「成本」数会把最大的那一项藏起来。

6. **引擎参数（2026-09-15 委托人补充）**
   `SimExchange(..., funding=...)` 是可选第 5 参数、默认 None，仓库里 9 个调用点有 8 个
   没传它（含 `d_param_search` / `d_fair_comparison` / `m5_online` / `m11_walkforward`），
   于是**引擎从不结算资金费**。本脚本**所有仿真都传 funding 表**；把它关掉的那一档
   只作为敏感性对照（`C_nofund` / `k1_noengfund` / `k05_nofund`）。

   **保证金参数**：`k1` = `initial_margin_ratio = 1.0/max_gross`（与真实杠杆一致），
   `k05` = 0.5（仓库现状，已知错误），`k1m005` = k=1/L 且维持保证金率换成 `carry.py`
   用的现实量级 0.005。结论一律建在 `A = k1 + 资金费 ON` 上。

   **强平阈值的推广式（本脚本推出、比既有审计更一般）**
   既有审计的阈值 `p/p₀ ≤ (1−k)/(1−m)` 只在「`margin_cash` 与当前持仓量成比例」时成立。
   但 `MarginBook.open_leg` **只在「从 0 开仓」或「多空翻转」时重锚**，逐 bar 再平衡的
   仓位在两次重锚之间 `margin_cash` 固定而 `|units|` 在变。设
   `a := margin_cash/|units_now|`，多头阈值变成

       p/p₀ ≤ (1 − a/p₀) / (1 − m),      a/p₀ = k · |units_anchor| / |units_now|

   持仓涨大 ⇒ `a/p₀` 变小 ⇒ **阈值上移，越赚越容易被强平**。实测：等权三币逐 bar
   再平衡，`k=1.0, m=0.1` 时 BTC 在 `p/p₀ = 0.757`（预测 0.7578）被强平、ETH 在 0.679
   —— **`k=1/L` 并没有关掉这条通道，只是把它从 −44.4% 挪到 −24.3% 之类的位置。**
   只有「发了目标就不动」的候选（band 开环、持仓量冻结）才真的 `ON ≡ OFF`（逐位一致，
   脚本里有 `exp:only_BTCUSDT` 的 101,251 = 101,251 作证）。

## 单一标的依赖的稳健性检查

同一套比较在四个池上重做：`be`(BTC+ETH) / `btc`(仅 BTC) / `full`(+BNB) /
`be_raw`(BTC+ETH，但**不裁到三币交集**，79,440 根 —— 用来量化「取数窗口」这个扰动)。
四个池都裁到同一窗口（除 `be_raw`），于是「去掉 BNB 后结论翻转」不会被窗口差异污染。

## 敏感度（学习体是混沌敏感的）

同一「2 标的 η=.2 b=.2」的策略，仅因配置差异就横跨一个数量级。所以脚本最后打一张
**变体敏感度表**：把同一候选在全部保证金/资金费/窗口变体下的终值列出来，报 max/min
比值。**单次运行的数字不是结论。**

## 跑法

    cd /home/nick/workspace/quant
    python3 scripts/vol_attribution.py               # 全量（约 25 分钟）
    python3 scripts/vol_attribution.py --list-plan
    python3 scripts/vol_attribution.py --quick
    python3 scripts/vol_attribution.py --deadline-min 20   # 到点不再起新 run（断点可续）

断点：每跑完一个 (候选 × 变体) 就往 `--ckpt`（默认 /tmp）落一条，重跑会自动跳过已完成的。

## 合规

只读本地已落盘数据（`data/raw/*.csv`）。不联网、不接账号、不存 key、不下单。
本脚本新建的产物只有 `runs/<Run id>/`。
"""
from __future__ import annotations

import argparse
import json
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
INITIAL_CASH = 10_000.0
WARMUP = 300
MAX_GROSS = 1.0                 # 与 d_fair_comparison / d_param_search 一致
BAND_EXPERT = 0.20              # 与真专家（HedgeEnsemble 默认 band）相同
HAC_LAGS = 24
COST_RATES = np.array([0.0012, 0.0013, 0.0014])
FUNDING_CSV = os.path.join("data", "funding.csv")


# ==========================================================================
# 事前登记（铁律 5：hypothesis 必须在跑之前写好）
# ==========================================================================
HYPOTHESIS = {
    "claim": (
        "H1（波动口径）: 本池 18 条专家规则中，没有任何一条在剥离 β 之后取得显著的"
        "正年化 α（HAC |t|<1.96）；且没有任何候选赢过「与其同 β 的无摩擦被动持有」。"
        "H2（单一标的依赖）: 「等权池里谁最强」的结论由池内是否含 BNB 决定 —— "
        "去掉 BNB 后池内最优候选的名字发生改变。"
        "H3（强平通道）: 在 k=1/L（与真实杠杆一致）下，band 开环的买入持有型候选"
        "通道贡献恒为 0（ON 与 OFF 逐位一致）；但任何逐 bar 再平衡的候选仍有非零通道，"
        "因为 margin_cash 只在「从 0 开仓/多空翻转」时重锚、与当前持仓量脱钩 —— "
        "即 k=1/L 只修掉了 k=0.5 的一半。"
        "H4（冻结参数对照）: 委托人引用的「关强平 1.62 倍 / 2.45x→1.55x」在 k=1/L 下显著缩小。"
        "H5（资金费，2026-09-15 追加）: 引擎结算资金费后，资金费是四块里**单块最大**的"
        "负贡献项（|G_fund| > |G_trade|），且被动等权的终值至少 −30%。"
        "H6（混沌敏感）: 同一候选在不同「等价」配置下的终值 max/min ≥ 2，"
        "所以单次运行的数字不能作为结论。"),
    "why": (
        "项目至今用终值倍数做标尺，而倍数随对照物翻转（同一学习体 2.45/1.55/7.61/4.86）。"
        "没有归因层就回答不了「输在哪儿」，也回答不了「波动里有没有正期望」；"
        "而不结算资金费会让「成本」这一块漏掉项目自己认定的最大项。"),
    "falsified_by": {
        "H1": "出现 ≥1 个候选 HAC t_α ≥ 1.96 且 α_ann > 0；或 ≥1 个候选 G_total > 0。",
        "H2": "be 池与 full 池的「池内净终值最强候选」是同一个名字。",
        "H3": "某个 band=0.2 的纯多头买入持有候选在 k=1/L 下 n_liquidated > 0；或全部候选在 k=1/L 下 G_liq 恒为 0。",
        "H4": "k=1/L 下的 OFF/ON 比值与 k=0.5 下的比值相差 < 5 个百分点。",
        "H5": "|G_fund| < |G_trade| 的候选占多数；或被动的终值变化 > −30%。",
        "H6": "全部候选的变体终值 max/min < 2。",
    },
    "registered": "2026-09-15",
    "note": (
        "H1~H4 在本脚本第一次运行（2026-09-15T054036）之前写定，该次运行因发现引擎未"
        "结算资金费而被中止（委托人 out-of-band 指出）。H5/H6 是中止后、在委托人明确"
        "要求下追加的 —— **如实记录：这两条是事后补充的，不是事前预注册**。"
        "其余假设的阈值未被改动，以便结果出来后用同一把尺子判定。"),
}


# ==========================================================================
# 引擎变体：保证金档 × 是否结算资金费 × agent 是否相信资金费
# ==========================================================================
MARGIN_TABLE = {
    "OFF": None,
    "k1": MarginConfig(initial_margin_ratio=1.0 / MAX_GROSS,
                       maintenance_margin_ratio=0.1, topup_trigger_ratio=0.5),
    "k05": MarginConfig(initial_margin_ratio=0.5,
                        maintenance_margin_ratio=0.1, topup_trigger_ratio=0.5),
    "k1m005": MarginConfig(initial_margin_ratio=1.0 / MAX_GROSS,
                           maintenance_margin_ratio=0.005, topup_trigger_ratio=0.5),
}
MARGIN_DOC = {
    "OFF": "margin=None：完全没有强平通道",
    "k1": "k=1/max_gross=1.0, m=0.1：与真实杠杆一致的「正确值」",
    "k05": "k=0.5, m=0.1：仓库现状（另一个 subagent 正在修的 bug）",
    "k1m005": "k=1.0, m=0.005：再把维持保证金率换到 carry.py 的现实量级",
}

# variant -> (margin_key, 引擎是否结算资金费, agent 是否把资金费放进 payoff)
ENGINE_VARIANTS = {
    "A_head":        ("k1",     True,  True),
    "B_fund":        ("OFF",    True,  True),
    "C_nofund":      ("OFF",    False, True),
    "k1_noengfund":  ("k1",     False, True),
    "k05_nofund":    ("k05",    False, True),
    "k05_head":      ("k05",    True,  True),
    "k1m005_head":   ("k1m005", True,  True),
    "A_agentNoFund": ("k1",     True,  False),
}
CORE_VARIANTS = ["A_head", "B_fund", "C_nofund"]
HEADLINE = "A_head"


# ==========================================================================
# 候选
# ==========================================================================
class FixedWeights:
    """恒定目标权重 + band（band=0 即每根 bar 重发 = 逐 bar 再平衡）。"""

    def __init__(self, weights: dict, band: float, name: str):
        self.weights = weights
        self.band = band
        self.name = name
        self.last = None
        self.emit = 0
        self.skip = 0
        self.t_weighted_gross = 0.0
        self._prev_step = None
        self._prev_gross = 0.0

    def _acc(self, step: int, gross: float) -> None:
        if self._prev_step is not None and step > self._prev_step:
            self.t_weighted_gross += self._prev_gross * (step - self._prev_step)
        self._prev_step = step
        self._prev_gross = gross

    def decide(self, view):
        syms = view.symbols()
        d = {s: float(self.weights.get(s, 0.0)) for s in syms}
        v = np.array([d[s] for s in syms], dtype=float)
        gross = float(np.abs(v).sum())
        if self.band > 0 and self.last is not None \
                and float(np.abs(v - self.last).sum()) < self.band:
            self.skip += 1
            self._acc(view.step, self._prev_gross)
            return None
        self.last = v
        self.emit += 1
        self._acc(view.step, gross)
        return d

    def mean_target_gross(self) -> float:
        if self._prev_step is None or self._prev_step <= 0:
            return float("nan")
        return self.t_weighted_gross / self._prev_step

    def trade_bar_share(self) -> float:
        return self.emit / max(self.emit + self.skip, 1)


class OneExpert:
    """单条专家规则过引擎 —— band 语义与 HedgeEnsemble 内完全一致。

    `HedgeEnsemble.decide` 的过滤是 `|mixed − last_emitted|₁ < band → 返回 None`，
    且 **`last_emitted` 只在真的发单时更新**（`src/agents/online.py:269-277`）。
    于是它是**开环**的：强平后它不知道自己被强平了，继续重发同一目标、
    被 band 滤掉、永不重入。本类逐字复刻这条语义。
    """

    def __init__(self, expert, symbols: list[str], band: float,
                 max_exposure: float = MAX_GROSS):
        self.expert = expert
        self.symbols = list(symbols)
        self.band = band
        self.max_exposure = max_exposure
        self.name = f"exp_{expert.name}_b{band}"
        self.last = None
        self.emit = 0
        self.skip = 0
        self.gross_sum = 0.0
        self.n_bars = 0

    def decide(self, view):
        w = np.asarray(self.expert.weights(view, self.symbols, self.max_exposure),
                       dtype=float)
        self.n_bars += 1
        self.gross_sum += float(np.abs(w).sum())
        if self.last is not None and \
                float(np.abs(w - self.last).sum()) < self.band:
            self.skip += 1
            return None
        self.last = w
        self.emit += 1
        return {s: float(x) for s, x in zip(self.symbols, w)}

    def mean_target_gross(self) -> float:
        return self.gross_sum / self.n_bars if self.n_bars else float("nan")

    def trade_bar_share(self) -> float:
        return self.emit / max(self.emit + self.skip, 1)


class HedgeAgent:
    """HedgeEnsemble 的薄包装 —— 把 mean_gross / 发单率取出来。"""

    def __init__(self, symbols, funding, eta, band):
        self.symbols = list(symbols)
        self.ag = HedgeEnsemble(symbols, cost_rate=COST_RATES[:len(symbols)],
                                eta=eta, band=band,
                                max_exposure=MAX_GROSS, funding=funding)
        self.name = f"hedge_eta{eta:.2f}_b{band:.2f}"
        self.eta = eta
        self.band = band

    def decide(self, view):
        return self.ag.decide(view)

    def mean_target_gross(self) -> float:
        w = self.ag.log["mixed_weight"]
        if not w:
            return float("nan")
        return float(np.abs(np.stack(w, axis=0)).sum(axis=1).mean())

    def trade_bar_share(self) -> float:
        t = self.ag.log["turnover"]
        if not t:
            return float("nan")
        return float((np.asarray(t, dtype=float) > 0).mean())


def candidate_factories(pool: list[str], funding):
    """[(key, label, factory, kind)]；factory(agent_funding: bool) 必须给出**全新** agent。"""
    k = len(pool)
    base = {s: 1.0 / k for s in pool}
    out = [
        ("ew_rebal", "等权池 每bar再平衡（★基准 BM1）",
         lambda _af=True: FixedWeights(base, 0.0, "ew_rebal"), "ref"),
        ("ew_bh", "等权池 买入持有（band 0.20）",
         lambda _af=True: FixedWeights(base, BAND_EXPERT, "ew_bh"), "ref"),
    ]
    if "BTCUSDT" in pool:
        out.append(("btc_rebal", "纯 BTC 逐bar再平衡（= d_fair_comparison 的「纯 BTC 满仓」）",
                    lambda _af=True: FixedWeights({"BTCUSDT": 1.0}, 0.0, "btc_rebal"), "ref"))
        out.append(("btc_bh", "纯 BTC 买入持有（band 0.20）",
                    lambda _af=True: FixedWeights({"BTCUSDT": 1.0}, BAND_EXPERT, "btc_bh"), "ref"))
    if "BTCUSDT" in pool and "ETHUSDT" in pool:
        out.append(("btceth_bh", "BTC+ETH 各半 买入持有（band 0.20）",
                    lambda _af=True: FixedWeights({"BTCUSDT": 0.5, "ETHUSDT": 0.5},
                                                  BAND_EXPERT, "btceth_bh"), "ref"))
    for e in default_experts(pool):
        out.append((f"exp:{e.name}", f"专家 {e.name}",
                    (lambda af=True, _e=e: OneExpert(_e, pool, BAND_EXPERT)),
                    "expert"))
    for eta in (0.01, 0.20):
        out.append((f"hedge_eta{eta:.2f}_b0.20",
                    f"学习体 Hedge η={eta:.2f} band=0.20",
                    (lambda af=True, _e=eta:
                     HedgeAgent(pool, funding if af else None, _e, BAND_EXPERT)),
                    "hedge"))
    return out


# ==========================================================================
# 计量：OLS + HAC
# ==========================================================================
def ols_with_hac(y: np.ndarray, x: np.ndarray, lags: int = HAC_LAGS) -> dict:
    """y = a + b·x + e 的 OLS，截距与斜率的普通 & Newey–West 稳健标准误。"""
    n = len(y)
    X = np.column_stack([np.ones(n), x])
    XtX = X.T @ X
    try:
        XtX_inv = np.linalg.inv(XtX)
    except np.linalg.LinAlgError:            # 零方差（例如全现金候选）
        return {"n": n, "a": float("nan"), "b": float("nan"),
                "se_a": float("nan"), "se_b": float("nan"),
                "se_a_ols": float("nan"), "se_b_ols": float("nan"),
                "r2": float("nan")}
    if not np.all(np.isfinite(XtX_inv)):
        return {"n": n, "a": float("nan"), "b": float("nan"),
                "se_a": float("nan"), "se_b": float("nan"),
                "se_a_ols": float("nan"), "se_b_ols": float("nan"),
                "r2": float("nan")}
    b = XtX_inv @ (X.T @ y)
    u = y - X @ b
    ssr = float(u @ u)
    sst = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - ssr / sst if sst > 0 else float("nan")
    dof = max(n - 2, 1)
    V_ols = (ssr / dof) * XtX_inv
    w = X * u[:, None]
    S = np.zeros((2, 2))
    for l in range(lags + 1):
        if l == 0:
            G = w.T @ w
        else:
            G = w[l:].T @ w[:-l] + w[:-l].T @ w[l:]
        S += (1.0 - l / (lags + 1.0)) * G
    V_hac = XtX_inv @ S @ XtX_inv
    return {
        "n": n, "a": float(b[0]), "b": float(b[1]),
        "se_a_ols": float(np.sqrt(max(V_ols[0, 0], 0.0))),
        "se_b_ols": float(np.sqrt(max(V_ols[1, 1], 0.0))),
        "se_a": float(np.sqrt(max(V_hac[0, 0], 0.0))),
        "se_b": float(np.sqrt(max(V_hac[1, 1], 0.0))),
        "r2": float(r2),
    }


def ann_alpha(a_bar: float) -> float:
    if not np.isfinite(a_bar):
        return float("nan")
    if a_bar <= -0.999999:
        return -1.0
    return float(np.expm1(BARS_PER_YEAR * np.log1p(a_bar)))


def path_metrics(equity: np.ndarray, base: float = INITIAL_CASH) -> dict:
    e = np.asarray(equity, dtype=float)
    if len(e) < 2:
        return {"final": float(e[-1]) if len(e) else float("nan"), "cagr": float("nan"),
                "ann_vol": float("nan"), "sharpe": float("nan"),
                "vol_norm": float("nan"), "max_dd": float("nan"), "n_bars": int(len(e))}
    r = np.diff(e) / e[:-1]
    r = r[np.isfinite(r)]
    sd = float(r.std(ddof=1)) if len(r) > 1 else 0.0
    ann_vol = sd * np.sqrt(BARS_PER_YEAR)
    sharpe = float(r.mean() / sd * np.sqrt(BARS_PER_YEAR)) if sd > 0 else 0.0
    years = len(e) / BARS_PER_YEAR
    fin = float(e[-1])
    cagr = float(np.expm1(np.log(fin / base) / years)) if (years > 0 and fin > 0) else -1.0
    path = np.concatenate([[base], e])
    peak = np.maximum.accumulate(path)
    dd = (path - peak) / np.where(peak > 0, peak, 1.0)
    return {"final": fin, "cagr": cagr, "ann_vol": ann_vol, "sharpe": sharpe,
            "vol_norm": float(cagr / ann_vol) if ann_vol > 0 else float("nan"),
            "max_dd": float(dd.min()), "n_bars": len(e)}


def compound_returns(rets: np.ndarray, beta: float = 1.0,
                     base: float = INITIAL_CASH) -> float:
    """把收益序列按 β 缩放后复利成一个终值（β 匹配被动持有的终值）。"""
    r = 1.0 + beta * np.asarray(rets, dtype=float)
    if np.any(r <= 0):
        idx = int(np.argmax(r <= 0))
        r = r[:idx + 1].copy()
        r[-1] = 0.0
        return base * float(np.prod(np.clip(r, 0.0, None)))
    return base * float(np.prod(r))


# ==========================================================================
# 引擎装配
# ==========================================================================
def common_index(core: list[str]):
    """所有 core 标的的交集索引 —— 让每个子池都用**同一个窗口**。

    不加这一步，`btc` 池是 79,440 根（2017-08-17 起），`full` 池是 77,503 根
    （2017-11-06 起，BNB 上线后才有交集）。窗口不同会让「去掉 BNB 后结论翻转」
    这条判断混进一段额外行情，所以默认把子池也裁到三币交集上；`be_raw` 池
    故意不裁，用来把「窗口」这个扰动单独量出来。
    """
    idx = None
    for s in core:
        f = store.load_bars(s, "1h")
        idx = f.index if idx is None else idx.intersection(f.index)
    return idx


def load_frames(pool: list[str], index=None) -> dict:
    fr = {}
    for s in pool:
        f = store.load_bars(s, "1h")[
            ["open", "high", "low", "close", "volume", "quote_volume"]]
        fr[s] = f.loc[index] if index is not None else f
    return fr


def run_once(frames: dict, cfg: dict, agent, margin_cfg, funding):
    """一次仿真。净/毛两个账本在同一次遍历里产出（见 SimExchange 模块说明）。

    **funding 一律传入**（除非变体显式要求关掉）—— 这是 2026-09-15 委托人指出的
    实质缺口：仓库里 8/9 个调用点没传，于是引擎从不结算资金费。
    """
    net_costs = CostModel.from_config(cfg, enabled=True)   # 每次全新 → 计数独立
    gross_costs = CostModel(enabled=False)
    res = SimExchange(
        frames, gross_costs, net_costs,
        SimConfig(initial_cash=INITIAL_CASH, warmup=WARMUP,
                  max_gross=MAX_GROSS, max_exposure_per_symbol=MAX_GROSS,
                  margin=margin_cfg, allow_short=True, instrument="perp"),
        funding=funding,
    ).run(agent)
    return res, net_costs


# ==========================================================================
# 归因表
# ==========================================================================
def build_table(raw: dict, pool: list[str], bm_cand: str = "ew_rebal") -> list[dict]:
    """raw[(candidate, variant)] -> 记录。归因表的**基准与候选同变体**（同口径）。"""
    rows = []
    cands = sorted({c for c, _ in raw})
    for cand in cands:
        for v in sorted({vv for c, vv in raw if c == cand}):
            rec = raw.get((cand, v))
            bm = raw.get((bm_cand, v))
            bm_variant_used = v
            if bm is None and v != HEADLINE:
                # 边际变体（k1m005_head 等）可能没给基准也跑同一档 ——
                # 退回基准变体并标注（否则这一档的行会整体消失，见 2026-09-15 的教训）
                bm = raw.get((bm_cand, HEADLINE))
                bm_variant_used = HEADLINE
            if rec is None or bm is None or len(rec["rets"]) == 0:
                continue
            n = min(len(rec["rets"]), len(bm["rets"]))
            if n < 100:
                continue
            y, x = rec["rets"][:n], bm["rets"][:n]
            if not (np.all(np.isfinite(y)) and np.all(np.isfinite(x))):
                continue
            # ---- 波动口径指标：**归零路径上的 vol/sharpe 是假信号** ----
            # 曲线被破产闸门截断时末期收益恰为 −100%，会把 std 抬到几百 %，
            # 于是「已经把本金亏光」的配置反而被读成「风险极高」。
            # 这与 WORK_LOG §16.3 修过的那类假信号同源，所以这里直接置空并从排名里排除。
            wiped = rec["final_net"] <= 0.01 * INITIAL_CASH
            sd = float(np.std(y, ddof=1)) if len(y) > 1 else 0.0
            ann_vol = sd * np.sqrt(BARS_PER_YEAR)
            sharpe = (float(np.mean(y) / sd * np.sqrt(BARS_PER_YEAR))
                      if sd > 0 else 0.0)
            cagr = rec.get("m", {}).get("cagr", float("nan"))
            max_dd = rec.get("m", {}).get("max_dd", float("nan"))
            if wiped:
                cagr = -1.0
                ann_vol = None
                sharpe = None
            vol_norm = (cagr / ann_vol
                        if (not wiped and ann_vol and ann_vol > 0) else None)
            fit = ols_with_hac(y, x)                     # 净路径 vs 同变体净基准
            c_ref = raw.get((cand, "C_nofund"))          # 无摩擦口径的参照
            if c_ref is not None and len(c_ref["rets"]) >= n:
                fg = ols_with_hac(c_ref["rets"][:n], raw[("ew_rebal", "C_nofund")]["rets"][:n])
                beta_g = fg["b"]
                g_base = float(np.prod(1.0 + c_ref["grets"][:n]))
                e_c_gross = c_ref["final_gross"]
            else:
                fg = fit
                beta_g = fit["b"]
                g_base = float(np.prod(1.0 + rec["grets"][:n]))
                e_c_gross = rec["final_gross"]
            bm_g = raw.get(("ew_rebal", "C_nofund"))
            bm_grets = bm_g["grets"][:n] if bm_g is not None else rec["grets"][:n]
            # β 匹配被动持有：用**无摩擦**基准收益 × β_g
            e_beta_gross = compound_returns(bm_grets, beta_g)
            # β 匹配被动持有（净口径，供「有没有超越同等暴露的被动持有」一问）
            e_beta_net = compound_returns(x, fit["b"])
            # 四块分解（只在基准变体 A_head 上给出，避免同一策略在每行重复同一个数）
            A = raw.get((cand, "A_head"))
            B = raw.get((cand, "B_fund"))
            C = raw.get((cand, "C_nofund"))
            ok = (v == HEADLINE and all(r is not None and r["final_net"] > 0
                                        for r in (A, B, C)) and e_beta_gross > 0)
            if ok:
                g_total = float(np.log(A["final_net"]) - np.log(e_beta_gross))
                g_exp = float(np.log(C["final_gross"]) - np.log(e_beta_gross))
                g_trade = float(np.log(C["final_net"]) - np.log(C["final_gross"]))
                g_fund = float(np.log(B["final_net"]) - np.log(C["final_net"]))
                g_liq = float(np.log(A["final_net"]) - np.log(B["final_net"]))
                resid = g_total - (g_exp + g_trade + g_fund + g_liq)
            else:
                g_total = g_exp = g_trade = g_fund = g_liq = resid = float("nan")
            rows.append({
                "pool": "+".join(pool), "candidate": cand, "variant": v,
                "benchmark_variant": bm_variant_used,
                "margin": ENGINE_VARIANTS.get(v, ("?",))[0],
                "engine_funding": ENGINE_VARIANTS.get(v, (None, None))[1],
                "derived": bool(rec.get("derived_from")),
                "n_bars_aligned": n,
                "bankrupt": bool(rec.get("bankrupt")),
                "wiped": bool(wiped),
                "death_ts": rec.get("death_ts"),
                "n_funding_events": rec.get("n_funding_events"),
                "final_net": rec["final_net"],
                "final_gross": rec.get("final_gross", float("nan")),
                "funding_paid": rec.get("funding_paid", 0.0),
                "cost_paid": rec.get("cost_paid", 0.0),
                "cagr": cagr, "ann_vol": ann_vol,
                "vol_norm": vol_norm, "sharpe": sharpe,
                "max_dd": max_dd,
                "beta": fit["b"], "beta_se_hac": fit["se_b"],
                "alpha_bar_bp": fit["a"] * 1e4, "alpha_ann": ann_alpha(fit["a"]),
                "t_alpha_hac": (fit["a"] / fit["se_a"] if fit["se_a"] and fit["se_a"] > 0
                                else float("nan")),
                "t_alpha_ols": (fit["a"] / fit["se_a_ols"]
                                if fit["se_a_ols"] and fit["se_a_ols"] > 0 else float("nan")),
                "r2": fit["r2"],
                "beta_g": beta_g, "alpha_ann_g": ann_alpha(fg["a"]),
                "r2_g": fg["r2"],
                "beta_matched_passive_gross": e_beta_gross,
                "beta_matched_passive_net": e_beta_net,
                "gap_vs_beta_matched_log": g_total,
                "gap_vs_beta_matched_ratio": (rec["final_net"] / e_beta_gross)
                if e_beta_gross > 0 else float("nan"),
                "beat_beta_matched_pretrade": bool(
                    e_beta_gross > 0 and rec["final_net"] > e_beta_gross),
                "beat_beta_matched": bool(ok and g_total > 0),
                "G_exp": g_exp, "G_trade": g_trade, "G_fund": g_fund, "G_liq": g_liq,
                "decomp_residual": resid,
                "mean_target_gross": rec.get("mean_gross", float("nan")),
                "trade_bar_share": rec.get("trade_share", float("nan")),
                "n_trades": rec.get("n_trades"), "n_liquidated": rec.get("n_liq"),
                "cost_pct_of_initial": rec.get("cost_paid", 0.0) / INITIAL_CASH * 100.0,
                "funding_pct_of_initial": rec.get("funding_paid", 0.0) / INITIAL_CASH * 100.0,
                "impact_capped": rec.get("impact_capped"),
            })
    return rows


# ==========================================================================
# 计划
# ==========================================================================
POOLS = {
    "be": ["BTCUSDT", "ETHUSDT"],
    "btc": ["BTCUSDT"],
    "full": ["BTCUSDT", "ETHUSDT", "BNBUSDT"],
    "be_raw": ["BTCUSDT", "ETHUSDT"],
}
NO_UNIFIED_INDEX = {"be_raw"}          # 故意不裁窗口，量化「窗口」这个扰动

# full 池只跑最代表性的几个（时间预算；完整覆盖放在 be 池 = 去掉 BNB 的那个池）
FULL_POOL_KEEP = {"ew_rebal", "ew_bh", "btc_bh", "exp:long_all",
                  "exp:only_BTCUSDT", "hedge_eta0.20_b0.20"}
# btc 池也收窄到代表性规则（同样收窄，理由见 docstring 的时间预算）
BTC_POOL_KEEP = {"ew_rebal", "btc_bh", "exp:long_all", "exp:inv_vol",
                 "exp:mom_720", "exp:short_all", "exp:short_mom_1",
                 "exp:only_BTCUSDT", "hedge_eta0.20_b0.20"}
# 学习体每池只跑 η=0.20 band=0.20 一档（单项 55~63 秒，η 网格装不下）
HEDGE_ETAS = (0.20,)
HEDGE_VARIANTS = {
    "be": ["A_head", "B_fund", "C_nofund", "k1m005_head", "k05_head",
           "k05_nofund", "k1_noengfund", "A_agentNoFund"],
    "btc": ["A_head", "B_fund", "C_nofund"],
    "full": ["A_head", "B_fund", "C_nofund", "k05_head"],
    "be_raw": ["A_head", "k1_noengfund"],
}
# 额外变体的代表性子集
K05_HEAD_SET = {"ew_rebal", "ew_bh", "btc_bh", "exp:only_BTCUSDT", "exp:long_all"}
K1M005_SET = {"exp:short_all", "exp:short_mom_1"}
# 用来验证「A 无强平时 B ≡ A」这条推导（强制真跑，把差值报出来）
VERIFY_DERIVED = [("be", "btc_rebal"), ("full", "btc_rebal")]


def build_plan(pool_name: str, funding, quick: bool) -> list[tuple[str, str]]:
    pool = POOLS[pool_name]
    cands = candidate_factories(pool, funding)
    keys = [c[0] for c in cands]
    kinds = {c[0]: c[3] for c in cands}
    keys = [k for k in keys if not k.startswith("hedge_eta0.01")]
    if pool_name == "full":
        keys = [k for k in keys if k in FULL_POOL_KEEP]
    elif pool_name == "btc":
        keys = [k for k in keys if k in BTC_POOL_KEEP]
    elif pool_name == "be_raw":
        keys = [k for k in keys if k in FULL_POOL_KEEP]
    plan: list[tuple[str, str]] = []
    for key in keys:
        for v in CORE_VARIANTS:
            plan.append((key, v))
    for key in keys:
        if kinds[key] == "hedge":
            for v in HEDGE_VARIANTS.get(pool_name, ["A_head"]):
                plan.append((key, v))
        elif pool_name in ("be", "btc", "full"):
            if key in K05_HEAD_SET:
                plan.append((key, "k05_head"))
            if key in K1M005_SET:
                plan.append((key, "k1m005_head"))
    if quick:
        keep = {"ew_rebal", "ew_bh", "btc_bh", "exp:long_all", "exp:short_all",
                "exp:only_BTCUSDT", "exp:short_mom_1", "hedge_eta0.20_b0.20"}
        plan = [(k, v) for k, v in plan if k in keep and v in CORE_VARIANTS]
    seen, out = set(), []
    for p in plan:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


# ==========================================================================
# 打印
# ==========================================================================
def _f(x, fmt: str, empty: str = "—") -> str:
    return f"{x:{fmt}}" if x is not None and np.isfinite(x) else f"{empty:>{len(empty)}}"


def print_table(rows: list[dict], title: str, variant: str = HEADLINE) -> None:
    print(f"\n=== {title}（变体 {variant}；**归零(净≤1%本金)的 vol/sharpe 置空、不参与排名**） ===")
    hdr = (f"{'candidate':<28}{'net':>12}{'CAGR':>9}{'vol':>8}{'volNorm':>9}"
           f"{'sharpe':>8}{'maxDD':>8}{'beta':>7}{'alphaAnn':>10}{'tHAC':>7}"
           f"{'R2':>7}{'|w|':>6}{'liq':>5}{'fund%':>8}{'cost%':>7}")
    print(hdr)
    print("-" * len(hdr))
    sel = [r for r in rows if r["variant"] == variant]
    alive = [r for r in sel if not r["wiped"]]
    for r in sorted(alive, key=lambda x: -x["final_net"]):
        print(f"{r['candidate']:<28}{r['final_net']:>12,.0f}"
              f"{(r['cagr'] or 0)*100:>8.1f}%"
              f"{(_f(r['ann_vol']*100, '7.1f') + '%') if r['ann_vol'] else '      —':>8}"
              f"{_f(r['vol_norm'], '9.3f')}{_f(r['sharpe'], '8.3f')}"
              f"{r['max_dd']*100:>7.1f}%"
              f"{r['beta']:>7.3f}{r['alpha_ann']*100:>9.1f}%{r['t_alpha_hac']:>7.2f}"
              f"{r['r2']:>7.3f}{r['mean_target_gross']:>6.2f}{r['n_liquidated']:>5}"
              f"{r['funding_pct_of_initial']:>8.1f}{r['cost_pct_of_initial']:>7.1f}")
    dead = [r for r in sorted(sel, key=lambda x: -x["final_net"]) if r["wiped"]]
    if dead:
        print(f"  --- 归零（vol/sharpe 不适用，不参与排名）---")
        for r in dead:
            print(f"{r['candidate']:<28}{r['final_net']:>12,.0f}"
                  f"      归零"
                  f"{'':>8}{'':>9}{'':>8}{r['max_dd']*100:>7.1f}%"
                  f"{r['beta']:>7.3f}{r['alpha_ann']*100:>9.1f}%{r['t_alpha_hac']:>7.2f}"
                  f"{r['r2']:>7.3f}{r['mean_target_gross']:>6.2f}{r['n_liquidated']:>5}"
                  f"{r['funding_pct_of_initial']:>8.1f}{r['cost_pct_of_initial']:>7.1f}"
                  f"  死于 {r['death_ts']}")


def print_decomp(rows: list[dict], title: str) -> None:
    print(f"\n=== {title}（对数缺口；>0 表示赢过同 β 无摩擦被动） ===")
    hdr = (f"{'candidate':<28}{'beta_g':>7}{'vs β匹配':>11}"
           f"{'G_exp':>10}{'G_trade':>10}{'G_fund':>10}{'G_liq':>10}{'残差':>10}")
    print(hdr)
    print("-" * len(hdr))
    sel = [r for r in rows if r["variant"] == HEADLINE]
    for r in sorted(sel, key=lambda x: (-x["gap_vs_beta_matched_log"]
                                        if np.isfinite(x["gap_vs_beta_matched_log"])
                                        else 9e9)):
        print(f"{r['candidate']:<28}{r['beta_g']:>7.3f}"
              f"{r['gap_vs_beta_matched_log']*100:>10.2f}%"
              f"{r['G_exp']*100:>9.2f}%{r['G_trade']*100:>9.2f}%"
              f"{r['G_fund']*100:>9.2f}%{r['G_liq']*100:>9.2f}%"
              f"{r['decomp_residual']*100:>9.2e}%")


# ==========================================================================
# 主流程
# ==========================================================================
CKPT_SCHEMA = 3
CKPT_SCHEMA_KEY = "__schema__"
# 记录里**必须**有这些字段；缺任一项则该条由旧版本写出，其结论不可用。
# 为什么必须丢弃而不是沿用：`funding_paid` 缺失时 `rec.get("funding_paid", 0.0)`
# 会把它变成 0.0，在表里与"这个策略真的没付过资金费"**无法区分** —— 那是静默错误。
# （实测 2026-09-15：用旧 ckpt 恢复跑出的表里，`btc_bh` 的 `funding_pct_of_initial`
# 显示 0.0，而同池真实值约 289%；`G_fund` 则退化成 nan。）
CKPT_REQUIRED = ("funding_paid", "n_funding_events", "bankrupt_at")


def drop_stale_ckpt_records(ckpt: dict) -> list[str]:
    """丢弃 schema 不匹配的检查点记录，返回被丢弃的键（供调用方报告）。

    纯函数（就地修改传入的 dict，无 I/O），便于单测。

    为什么必须丢弃：缺 `funding_paid` 的记录经 `rec.get("funding_paid", 0.0)`
    会变成 0.0，在表里与"这个策略真的没付过资金费"**无法区分**。
    """
    stale = [k for k, v in ckpt.items()
             if k != CKPT_SCHEMA_KEY and isinstance(v, dict)
             and any(f not in v for f in CKPT_REQUIRED)]
    for k in stale:
        del ckpt[k]
    return stale


def rows_from_ckpt(pool_name: str, ckpt: dict) -> list[dict]:
    """从断点记录重建某个池的归因表（不带收益序列，可 JSON 落盘）。"""
    pool = POOLS[pool_name]
    raw = {}
    for key, rec in ckpt.items():
        parts = key.split("|")
        if len(parts) != 3 or parts[0] != pool_name or not isinstance(rec, dict):
            continue
        if "rets" not in rec or "grets" not in rec:
            continue
        r = dict(rec)
        r["rets"] = np.asarray(r["rets"], dtype=float)
        r["grets"] = np.asarray(r["grets"], dtype=float)
        raw[(parts[1], parts[2])] = r
    return build_table(raw, pool)


def last_row_summary(ckpt: dict) -> list[dict]:
    """轻量汇总：每项一行、不含收益序列 —— 每跑完一项就落盘一次。"""
    out = []
    for key, rec in ckpt.items():
        parts = key.split("|")
        if len(parts) != 3 or not isinstance(rec, dict):
            continue
        fn = rec.get("final_net", float("nan"))
        wiped = bool(np.isfinite(fn) and fn <= 0.01 * INITIAL_CASH)
        out.append({
            "pool": parts[0], "candidate": parts[1], "variant": parts[2],
            "final_net": fn,
            "final_gross": rec.get("final_gross"),
            "wiped": wiped,
            "ann_vol": None if wiped else rec.get("m", {}).get("ann_vol"),
            "sharpe": None if wiped else rec.get("m", {}).get("sharpe"),
            "cagr": -1.0 if wiped else rec.get("m", {}).get("cagr"),
            "n_liquidated": rec.get("n_liq"),
            "n_trades": rec.get("n_trades"),
            "cost_paid": rec.get("cost_paid"),
            "funding_paid": rec.get("funding_paid", 0.0),
            "bankrupt": rec.get("bankrupt"),
            "death_ts": rec.get("death_ts"),
            "derived_from": rec.get("derived_from"),
        })
    return sorted(out, key=lambda d: (d["pool"], d["candidate"], d["variant"]))


def assemble(pool_names: list[str], ckpt: dict, run_ctx, n_done: int,
             n_planned: int, complete: bool, verbose: bool) -> dict:
    """把断点里的记录整理成归因表 + 结论，并**立刻落盘**（被掐掉也留得下）。"""
    per_pool: dict[str, list[dict]] = {}
    for p in pool_names:
        rows = rows_from_ckpt(p, ckpt)
        per_pool[p] = rows
        if rows:
            run_ctx.log(f"attribution_{p}", rows)

    # ---- 跨池稳健性 ----
    rob = []
    for p in pool_names:
        rows = [r for r in per_pool.get(p, []) if r["variant"] == HEADLINE]
        if not rows:
            continue
        bm = [r for r in rows if r["candidate"] == "ew_rebal"]
        if not bm:
            continue
        bm = bm[0]
        alive = [r for r in rows if not r["wiped"]]
        best = max(alive, key=lambda r: r["final_net"]) if alive else None
        pos = [r for r in rows if np.isfinite(r["t_alpha_hac"])
               and r["t_alpha_hac"] >= 1.96 and r["alpha_ann"] > 0]
        beat = [r for r in alive if r.get("beat_beta_matched_pretrade")]
        n_exp = sum(1 for r in rows if r["candidate"].startswith("exp:"))
        n_exp_beat_bh = sum(1 for r in rows if r["candidate"].startswith("exp:")
                            and not r["wiped"]
                            and r["final_net"] > bm["final_net"]
                            and r["candidate"] not in ("exp:long_all",))
        rob.append({
            "pool": p, "n_candidates": len(rows), "n_experts": n_exp,
            "bm1_final": bm["final_net"], "bm1_funding_pct": bm["funding_pct_of_initial"],
            "best_candidate": best["candidate"] if best else None,
            "best_final": best["final_net"] if best else None,
            "x_vs_bm1": (best["final_net"] / bm["final_net"]) if best else None,
            "n_sig_pos_alpha": len(pos),
            "n_beat_beta_matched": len(beat),
            "n_experts_beating_ew_bh": n_exp_beat_bh,
            "n_wiped": sum(1 for r in rows if r["wiped"]),
        })

    # ---- 净值比 + Spearman ----
    rv = {}
    for p in pool_names:
        rows = [r for r in per_pool.get(p, []) if r["variant"] == HEADLINE]
        if not rows:
            continue
        bm = [r for r in rows if r["candidate"] == "ew_rebal"]
        if not bm:
            continue
        rv[p] = {r["candidate"]: r["final_net"] / bm[0]["final_net"] for r in rows}
    spearman = {}
    for a in rv:
        for b in rv:
            if a >= b:
                continue
            common = sorted(set(rv[a]) & set(rv[b]))
            if len(common) >= 4:
                xa = np.argsort(np.argsort([rv[a][c] for c in common])).astype(float)
                xb = np.argsort(np.argsort([rv[b][c] for c in common])).astype(float)
                xa -= xa.mean()
                xb -= xb.mean()
                den = np.sqrt((xa ** 2).sum() * (xb ** 2).sum())
                spearman[f"{a}vs{b}"] = float(xa @ xb / den) if den > 0 else float("nan")

    # ---- 变体敏感度 ----
    sens = {}
    for p in pool_names:
        rows = per_pool.get(p, [])
        by = {}
        for r in rows:
            by.setdefault(r["candidate"], {})[r["variant"]] = r["final_net"]
        for c, vals in by.items():
            vals = {v: x for v, x in vals.items() if np.isfinite(x)}
            if len(vals) < 2:
                continue
            lo, hi = min(vals.values()), max(vals.values())
            sens[f"{p}|{c}"] = {"values": vals, "min": lo, "max": hi,
                                "ratio": (hi / lo) if lo > 0 else float("inf")}

    # ---- H2 / 关键判据 ----
    h_rows = [r for rows in per_pool.values() for r in rows
              if r["variant"] == HEADLINE]
    sig = [r for r in h_rows if np.isfinite(r["t_alpha_hac"])
           and r["t_alpha_hac"] >= 1.96 and r["alpha_ann"] > 0]
    beat = [r for r in h_rows if r.get("beat_beta_matched_pretrade")]
    ok_dec = [r for r in h_rows if np.isfinite(r["decomp_residual"])
              and abs(r["decomp_residual"]) < 1e-9]
    liq_nz = [r for r in h_rows if np.isfinite(r["G_liq"]) and abs(r["G_liq"]) > 1e-12]
    fund_gt = [r for r in h_rows if np.isfinite(r["G_fund"])
               and np.isfinite(r["G_trade"]) and abs(r["G_fund"]) > abs(r["G_trade"])]
    wiped_n = [r for r in h_rows if r["wiped"]]

    # long_all 与 ew_buyhold 是否逐位相同（委托人指定要写进报告的结论）
    identical = {}
    for p in pool_names:
        a = ckpt.get(f"{p}|exp:long_all|{HEADLINE}")
        b = ckpt.get(f"{p}|ew_bh|{HEADLINE}")
        if a is not None and b is not None:
            identical[p] = float(abs(a["final_net"] - b["final_net"]))
    h2 = None
    if "be" in rv and "full" in rv:
        b_be = next((r["best_candidate"] for r in rob if r["pool"] == "be"), None)
        b_fu = next((r["best_candidate"] for r in rob if r["pool"] == "full"), None)
        h2 = {"be_best": b_be, "full_best": b_fu, "flips": (b_be != b_fu)}

    ratios = [d["ratio"] for d in sens.values() if np.isfinite(d["ratio"])]
    metrics = {
        "status": "complete" if complete else "partial",
        "n_done": n_done, "n_planned": n_planned,
        "n_rows": sum(len(v) for v in per_pool.values()),
        "n_headline_rows": len(h_rows),
        "n_headline_wiped": len(wiped_n),
        "note_wiped_metric_policy": (
            "net ≤ 1% × 初始本金（10,000）即视为归零：vol/sharpe 置 null 且不参与排名"
            "（归零路径末期收益恰为 −100%，会把年化波动抬到几百 %，是假信号）。"),
        "H1_no_significant_positive_alpha": len(sig) == 0,
        "n_sig_positive_alpha_HAC_t>=1.96": len(sig),
        "sig_positive_alpha_examples": [
            {"pool": r["pool"], "candidate": r["candidate"], "alpha_ann": r["alpha_ann"],
             "t_hac": r["t_alpha_hac"], "beta": r["beta"], "r2": r["r2"]} for r in sig][:20],
        "n_beat_beta_matched_passive": len(beat),
        "beat_beta_matched_examples": [
            {"pool": r["pool"], "candidate": r["candidate"],
             "ratio": r["gap_vs_beta_matched_ratio"]} for r in beat][:20],
        "n_4block_additive_ok": len(ok_dec),
        "decomp_residual_max_abs": max(
            (abs(r["decomp_residual"]) for r in h_rows
             if np.isfinite(r["decomp_residual"])), default=None),
        "n_headline_with_nonzero_G_liq": len(liq_nz),
        "G_liq_examples": [{"pool": r["pool"], "candidate": r["candidate"],
                            "G_liq": r["G_liq"], "n_liquidated": r["n_liquidated"]}
                           for r in liq_nz][:15],
        "n_funding_block_larger_than_trade_block": len(fund_gt),
        "robustness_per_pool": rob,
        "net_value_ratios_vs_bm1": rv,
        "robustness_spearman": spearman,
        "variant_sensitivity": sens,
        "variant_sensitivity_max_ratio": max(ratios) if ratios else None,
        "long_all_vs_ew_buyhold_abs_diff": identical,
        "H2_leader_flips_without_BNB": h2,
        "verdict_location": run_ctx.dir,
    }
    run_ctx.record_metrics(metrics)
    run_ctx.log("rows_incremental", last_row_summary(ckpt))
    if verbose:
        print("\n############ 跨池稳健性（变体 A_head）############")
        for r in rob:
            print(f"  {r['pool']:<8} BM1 {r['bm1_final']:>12,.0f} "
                  f"(资金费 {r['bm1_funding_pct']:>6.0f}% 本金) | 最强者 "
                  f"{r['best_candidate']:<26}{r['best_final']:>12,.0f} "
                  f"({r['x_vs_bm1']:>6.2f}x) | 专家数 {r['n_experts']} | "
                  f"跑赢等权买入持有 {r['n_experts_beating_ew_bh']} | "
                  f"t_α≥1.96 且 α>0 {r['n_sig_pos_alpha']} | 归零 {r['n_wiped']}")
        if h2:
            print(f"  H2：be 池最强者={h2['be_best']}；full 池最强者={h2['full_best']} "
                  f"⇒ 「池内最强」{'改变（H2 成立）' if h2['flips'] else '不变（H2 被证伪）'}")
        if identical:
            print(f"  long_all 与 ew_buyhold 的 |Δ终值|：{identical}"
                  f"（0 表示逐位相同 —— 与 pool_ceiling 的 on_else_off_identical 一致）")
        if spearman:
            print(f"  跨池排序 Spearman：{spearman}")
        print("\n############ 变体敏感度（单次运行的数字不是结论）############")
        for k, d in sorted(sens.items(), key=lambda x: -x[1]["ratio"])[:14]:
            print(f"  {k:<46}{d['ratio']:>7.2f}x  [{d['min']:>10,.0f} … {d['max']:>10,.0f}]")
        print("\n############ 结论判据 ############")
        print(f"  A_head 候选：{len(h_rows)}（归零 {len(wiped_n)}）")
        print(f"  H1 HAC t_α≥1.96 且 α_ann>0：{len(sig)}")
        print(f"     赢过同 β 无摩擦被动持有：{len(beat)}")
        print(f"  四块分解可加性 |残差|<1e-9：{len(ok_dec)}/{len(h_rows)}")
        print(f"  强平通道非零的候选：{len(liq_nz)}")
        print(f"  资金费块 > 交易成本块：{len(fund_gt)}/{len(h_rows)}")
    return metrics


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pools", default="be,btc,full,be_raw")
    ap.add_argument("--ckpt", default="/tmp/vol_attribution_ckpt.json")
    ap.add_argument("--run-dir", default=None)
    ap.add_argument("--deadline-min", type=float, default=20.0)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--list-plan", action="store_true")
    args = ap.parse_args()
    pool_names = args.pools.split(",")

    cfg = cfgmod.load("base")
    funding = FundingTable.load(os.path.join(store.project_root(), FUNDING_CSV))

    if args.list_plan:
        tot = 0
        for p in pool_names:
            plan = build_plan(p, funding, args.quick)
            tot += len(plan)
            print(f"[{p}] {len(POOLS[p])} 标的，{len(plan)} 个 run")
            for k, m in plan:
                print(f"    {k:<28} {m}")
        print(f"合计 {tot} 个 run")
        return

    t0 = time.time()
    core = cfg["universe"]["core"]
    cidx = common_index(core)
    print(f"统一窗口（三币交集）：{len(cidx):,} 根 {cidx[0]:%Y-%m-%d}~{cidx[-1]:%Y-%m-%d}")
    print(f"funding 表：{{s: n}} = "
          f"{ {s: len(d) for s, d in funding.rates_by_hour.items()} }")
    run_ctx = Run("vol_attribution", {
        "script": "scripts/vol_attribution.py",
        "pools": {k: POOLS[k] for k in pool_names},
        "engine": ("SimExchange(instrument=perp, allow_short=True, max_gross=1.0, "
                   "warmup=300, funding=FundingTable[data/funding.csv])"),
        "benchmark": "同池等权每bar再平衡（FixedWeights, band=0, 同引擎同变体）",
        "band_semantics": (f"专家/买入持有候选一律 band={BAND_EXPERT}（开环，与真专家一致）；"
                           "再平衡候选 band=0"),
        "engine_variants": {k: {"margin": MARGIN_DOC[v[0]],
                                "engine_settles_funding": v[1],
                                "agent_believes_funding": v[2]}
                            for k, v in ENGINE_VARIANTS.items()},
        "decomposition": ("G_total = G_exp(敞口缺口) + G_trade(交易成本) + G_fund(资金费) "
                          "+ G_liq(强平通道)，对数空间严格可加"),
        "bars_per_year": BARS_PER_YEAR, "hac_lags": HAC_LAGS,
        "initial_cash": INITIAL_CASH,
        "wiped_policy": "net ≤ 1% × 本金 → vol/sharpe 置空、不参与排名",
        "window": (f"{len(cidx)} 根 {cidx[0]:%Y-%m-%d}~{cidx[-1]:%Y-%m-%d}"
                   f"（子池统一裁到三币交集；be_raw 故意不裁，作为窗口扰动对照）"),
    }, HYPOTHESIS)
    if args.run_dir:
        run_ctx.dir = args.run_dir
        os.makedirs(run_ctx.dir, exist_ok=True)
        run_ctx._write("config.json", run_ctx.cfg)
        run_ctx._write("meta.json", run_ctx.meta)
    print(f"run 目录：{run_ctx.dir}")
    run_ctx.note("hypothesis 已在跑之前写入 meta.json（铁律 5）。")

    ckpt = {}
    if os.path.exists(args.ckpt):
        with open(args.ckpt, encoding="utf-8") as f:
            ckpt = json.load(f)
    rename = {"OFF": "C_nofund", "k1": "k1_noengfund", "k05": "k05_nofund"}
    migrated = 0
    for old_key in list(ckpt):
        parts = old_key.split("|")
        if len(parts) == 3 and parts[2] in rename:
            new_key = "|".join(parts[:2] + [rename[parts[2]]])
            if new_key not in ckpt:
                rec = ckpt[old_key]
                rec["derived_from"] = None
                ckpt[new_key] = rec
                migrated += 1
            del ckpt[old_key]

    # ── schema 守卫：丢弃旧版本写出的记录，让它们重跑 ──────────────────────
    # 不这样做的话，缺字段的记录会被防御式默认值变成 0.0，在表里与真实值无法区分
    # （见 CKPT_REQUIRED 处的说明）。**丢弃并重跑**是唯一不会产出静默错值的做法。
    ckpt_ver = ckpt.get(CKPT_SCHEMA_KEY)
    stale = drop_stale_ckpt_records(ckpt)
    n_recs = sum(1 for k in ckpt if k != CKPT_SCHEMA_KEY)
    print(f"断点：{args.ckpt}（已有 {n_recs} 条，迁移 {migrated} 条，"
          f"schema {ckpt_ver} → {CKPT_SCHEMA}）")
    if stale:
        print(f"⚠️ 丢弃 {len(stale)} 条旧版本记录（缺少 {CKPT_REQUIRED} 中至少一项）"
              f"—— 它们**会被重跑**。沿用它们会把『资金费』静默写成 0.0，"
              f"与『真的没收过资金费』无法区分。样例：{stale[:3]}")

    def save_ckpt() -> None:
        ckpt[CKPT_SCHEMA_KEY] = CKPT_SCHEMA
        with open(args.ckpt, "w", encoding="utf-8") as f:
            json.dump(ckpt, f)

    deadline = t0 + args.deadline_min * 60.0
    total_planned = sum(len(build_plan(p, funding, args.quick)) for p in pool_names)
    done = 0
    skipped = []
    assemble(pool_names, ckpt, run_ctx, 0, total_planned, False, False)  # 先落一次盘
    print(f"计划 {total_planned} 个 run（deadline {args.deadline_min:.0f} 分钟）")

    for pool_name in pool_names:
        pool = POOLS[pool_name]
        frames = load_frames(pool, None if pool_name in NO_UNIFIED_INDEX else cidx)
        print(f"\n########## 池 {pool_name} {pool} —— {len(frames[pool[0]]):,} 根 "
              f"{frames[pool[0]].index[0]:%Y-%m-%d}~{frames[pool[0]].index[-1]:%Y-%m-%d}")
        cat_map = {c[0]: c for c in candidate_factories(pool, funding)}
        plan = build_plan(pool_name, funding, args.quick)

        for cand, v in plan:
            ck = f"{pool_name}|{cand}|{v}"
            if ck in ckpt:
                done += 1
                continue
            mkey, eng_f, ag_f = ENGINE_VARIANTS[v]
            if v == "B_fund" and (pool_name, cand) not in VERIFY_DERIVED:
                a = ckpt.get(f"{pool_name}|{cand}|A_head")
                if a is not None and a.get("n_liq", 0) == 0 and not a.get("bankrupt"):
                    import copy
                    rec = copy.deepcopy(a)
                    rec["derived_from"] = "A_head(n_liq==0)"
                    ckpt[ck] = rec
                    save_ckpt()
                    done += 1
                    continue
            if time.time() > deadline:
                skipped.append(ck)
                continue
            ag = cat_map[cand][2](ag_f)
            res, _cm = run_once(frames, cfg, ag, MARGIN_TABLE[mkey],
                                funding if eng_f else None)
            rec = {
                "rets": [float(x) for x in (np.diff(res.net_equity) / res.net_equity[:-1])],
                "grets": [float(x) for x in (np.diff(res.gross_equity) / res.gross_equity[:-1])],
                "final_net": float(res.net_equity[-1]),
                "final_gross": float(res.gross_equity[-1]),
                "bankrupt": bool(res.bankrupt),
                "n_trades": res.n_trades,
                "n_liq": len(res.liquidated_legs),
                "bankrupt_at": res.bankrupt_at,
                "death_ts": (str(res.index[res.bankrupt_at])
                             if res.bankrupt_at is not None
                             and res.bankrupt_at < len(res.index) else None),
                "last_ts": str(res.index[-1]) if len(res.index) else None,
                "n_funding_events": int(res.n_funding_events),
                "cost_paid": float(res.cost_paid.sum()),
                "funding_paid": (float(res.funding_paid.sum())
                                 if len(res.funding_paid) else 0.0),
                "impact_capped": int(res.n_impact_capped),
                "mean_gross": (float(ag.mean_target_gross())
                               if hasattr(ag, "mean_target_gross") else float("nan")),
                "trade_share": (float(ag.trade_bar_share())
                                if hasattr(ag, "trade_bar_share") else float("nan")),
                "m": path_metrics(res.net_equity),
                "derived_from": None,
            }
            ckpt[ck] = rec
            save_ckpt()
            done += 1
            flag = " *归零*" if rec["final_net"] <= 0.01 * INITIAL_CASH else ""
            print(f"  [{done:>3}/{total_planned}] {pool_name:<7}{cand:<26}{v:<15}"
                  f"net {rec['final_net']:>12,.0f}{flag:<7} liq {rec['n_liq']:>3} "
                  f"fund {rec['funding_paid']:>10,.0f} cost {rec['cost_paid']:>9,.0f}"
                  f"  ({time.time() - t0:>5.0f}s)")
            # **边跑边落盘**：轻量行 + 部分 metrics，被掐掉也留得下产物
            run_ctx.log("rows_incremental", last_row_summary(ckpt))
            run_ctx.record_metrics({"status": "partial", "n_done": done,
                                    "n_planned": total_planned,
                                    "note": "运行中；结束时会被完整版覆盖。"})

        # 这个池跑完了（或被掐）→ 立刻重建 + 落盘这个池的完整归因表
        assemble(pool_names, ckpt, run_ctx, done, total_planned, False, False)
        rows = rows_from_ckpt(pool_name, ckpt)
        if rows:
            print_table(rows, f"池 {pool_name}：波动/β 口径 + 归因")
            print_decomp(rows, f"池 {pool_name}：四块归因分解")
            self_row = [r for r in rows
                        if r["candidate"] == "ew_rebal" and r["variant"] == HEADLINE]
            if self_row:
                s = self_row[0]
                print(f"  [自检] BM1 对自身：β={s['beta']:.6f} "
                      f"α_bar={s['alpha_bar_bp']:.3e}bp R²={s['r2']:.6f}（应为 1 / 0 / 1）")
                run_ctx.note(f"自检 [{pool_name}] BM1 对自身 β={s['beta']:.6f} "
                             f"α_bar={s['alpha_bar_bp']:.3e}bp R²={s['r2']:.6f}")

    metrics = assemble(pool_names, ckpt, run_ctx, done, total_planned, True, True)
    if skipped:
        print(f"\n  因 deadline 未起跑的 run：{len(skipped)} 个；{skipped[:10]}"
              f"{' ...' if len(skipped) > 10 else ''}")
        run_ctx.note(f"因 deadline 未起跑 {len(skipped)} 个 run：{skipped}")
    print(f"\n证据目录：{run_ctx.dir}")
    print(f"总耗时 {time.time() - t0:.0f}s"
          f"；完成 {done}/{total_planned}；metrics.status={metrics['status']}")


if __name__ == "__main__":
    main()
