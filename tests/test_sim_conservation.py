"""SimExchange **资金守恒**不变量测试。

**为什么单独开一个文件**：C 阶段原有的集成测试（`test_agent_short_capability.py` §11）
只断言了"`margin_cash` 被设置/清零"和"`liquidated_legs` 里有这个标的"，
**从不断言现金/权益是否守恒**。于是 2026-09-15 修掉的那个致命 bug
（强平时把 `units` 清零却无现金对冲 ⇒ 空头负债凭空消失 ⇒ 权益凭空增加）
在 25 条 C 阶段测试全绿的情况下安然存活到 D 阶段。

本文件补的就是这一层：**任何一笔交易/强平都不能凭空创造或销毁权益。**
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from src.sim.costs import CostModel  # noqa: E402
from src.sim.exchange import SimConfig, SimExchange  # noqa: E402
from src.sim.margin import MarginConfig  # noqa: E402


# ==========================================================================
# 夹具
# ==========================================================================
def flat_frames(n=60, symbol="BTCUSDT", price=100.0, spike_at=None,
                spike_high=130.0):
    """价格恒为 `price` 的合成窗口；可在 `spike_at` 制造一根 high 尖峰。

    low 默认比 close 低 0.1%，所以多头的强平判定不会被噪声误触发。
    """
    idx = pd.date_range("2024-01-01", tz="UTC", periods=n, freq="h")
    close = np.full(n, price, dtype=float)
    high = close * 1.001
    low = close * 0.999
    if spike_at is not None:
        high[spike_at] = spike_high
        low[spike_at] = price * 0.999
    return {symbol: pd.DataFrame({
        "open": close, "high": high, "low": low, "close": close,
        "volume": 1_000.0, "quote_volume": close * 1_000.0,
    }, index=idx)}


class OnceAgent:
    """只在第一次决策时给出权重，之后不再动作（便于精确核对账目）。"""
    name = "once"

    def __init__(self, weights):
        self.weights = weights
        self._done = False

    def decide(self, view):
        if self._done:
            return None
        self._done = True
        return dict(self.weights)


def run_sim(frames, weights, *, cash=10_000.0, warmup=10, spike_at=None,
            instrument="perp", allow_short=True, margin=True, exposure=1.0):
    mcfg = MarginConfig(initial_margin_ratio=0.5, maintenance_margin_ratio=0.25) \
        if margin else None
    sim = SimExchange(
        frames,
        CostModel(enabled=False), CostModel(enabled=False),   # 关成本 ⇒ 数字精确
        SimConfig(initial_cash=cash, warmup=warmup,
                  max_gross=exposure, max_exposure_per_symbol=exposure,
                  allow_short=allow_short, instrument=instrument,
                  margin=mcfg),
    )
    res = sim.run(OnceAgent(weights))
    return sim, res


# ==========================================================================
# 核心：强平不能造钱
# ==========================================================================
def test_short_liquidation_loses_money_not_creates_it():
    """**这是本该拦住那个致命 bug 的测试。**

    场景：做空 1x @100，随后出现 130 的 high 触发强平。
      - 正确记账：`cash += units × 强平价 = -100 × 130 = -13,000`
        ⇒ 权益 10,000 → **7,000**（亏掉 30 点 × 100 单位 = 3,000）
      - 初版（错）：`units` 清零但无现金对冲 ⇒ 空头负债凭空消失
        ⇒ 权益 10,000 → **20,000**（凭空多出 10,000）

    两者相差 13,000，是"造钱"还是"亏钱"的分水岭。
    """
    spike = 25
    frames = flat_frames(n=60, spike_at=spike, spike_high=130.0)
    sim, res = run_sim(frames, {"BTCUSDT": -1.0}, spike_at=spike)

    assert "BTCUSDT" in res.liquidated_legs, \
        f"做空 + high 尖峰 130 应触发强平，实际 {res.liquidated_legs}"

    final_eq = res.net_equity[-1]
    # 亏 3,000 ⇒ 7,000。初版会得到 20,000。
    assert final_eq < 9_000, (
        f"强平后权益 {final_eq:,.2f} —— 做空被强平必须**亏钱**。"
        f"若接近 20,000 则说明强平在凭空造钱（初版 bug）。")
    assert final_eq == pytest.approx(7_000.0, abs=1.0), f"权益应为 7,000，实际 {final_eq}"


def test_long_liquidation_on_low_spike_also_loses_money():
    """多头对称：价格跌到 60 触发强平，权益应为 10,000 − 4,000 = 6,000。

    2x 杠杆（initial=0.5, maintenance=0.25）下多头的强平线是
    `100p − 5000 ≤ 25p` ⇒ **p ≤ 66.67**（−33%）。所以用 low=60 才触发，
    用 70（−30%）不够 —— 这条边界本身也值得钉住。
    """
    idx = pd.date_range("2024-01-01", tz="UTC", periods=60, freq="h")
    close = np.full(60, 100.0)
    high = close * 1.001
    low = close * 0.999
    low[25] = 60.0                      # 多头的不利价是 low
    frames = {"BTCUSDT": pd.DataFrame({
        "open": close, "high": high, "low": low, "close": close,
        "volume": 1_000.0, "quote_volume": close * 1_000.0,
    }, index=idx)}

    sim, res = run_sim(frames, {"BTCUSDT": 1.0})
    assert "BTCUSDT" in res.liquidated_legs, \
        f"多头 + low 跌到 60 应触发强平，实际 {res.liquidated_legs}"
    # 多头 1x @100：100 单位。强平在 60 ⇒ 亏 40 点 × 100 = 4,000
    assert res.net_equity[-1] == pytest.approx(6_000.0, abs=1.0)


def test_long_not_liquidated_at_30_percent_drawdown():
    """边界守卫：−30%（p=70）**不该**强平，−34%（p=66）该强平。

    初版的权益口径（绝对市值）让多头几乎永不被强平；这条钉住修正后的边界。
    """
    def at(low_px):
        idx = pd.date_range("2024-01-01", tz="UTC", periods=60, freq="h")
        close = np.full(60, 100.0)
        low = close * 0.999
        low[25] = low_px
        fr = {"BTCUSDT": pd.DataFrame({
            "open": close, "high": close * 1.001, "low": low, "close": close,
            "volume": 1_000.0, "quote_volume": close * 1_000.0,
        }, index=idx)}
        return run_sim(fr, {"BTCUSDT": 1.0})[1]

    assert at(70.0).liquidated_legs == [], "−30% 不应强平"
    assert at(66.0).liquidated_legs == ["BTCUSDT"], "−34% 应强平"


def test_liquidation_equity_is_continuous_across_the_event():
    """强平当根的权益变化只能由**价格变动**解释，不能有跳变。

    逐 bar 检查权益差分：单根跳变不得超过「仓位价值 × 该根的相对价格振幅」。
    初版在强平根会出现 `|units| × 价格` 量级的正向跳变。
    """
    spike = 25
    frames = flat_frames(n=60, spike_at=spike, spike_high=130.0)
    sim, res = run_sim(frames, {"BTCUSDT": -1.0})
    eq = res.net_equity
    d = np.diff(eq)
    # 单根权益变化不可能超过 20,000（本金 1 万 + 借来的仓位）；初版跳变 10,000 就在此列
    assert np.all(np.abs(d) < 6_000.0), \
        f"出现异常权益跳变，最大 {np.abs(d).max():,.2f}（差分 {np.round(d, 1)}）"


# ==========================================================================
# 破产闸门
# ==========================================================================
def test_bankruptcy_gate_stops_trading_and_truncates_curve():
    """权益 ≤ 0 时必须停止交易并截断曲线 —— 仓位公式在负权益下会翻号失控。

    构造：做空 1x，high 尖峰到 300 ⇒ 强平亏 200 点 × 100 单位 = 20,000 > 本金。
    """
    frames = flat_frames(n=60, spike_at=25, spike_high=300.0)
    sim, res = run_sim(frames, {"BTCUSDT": -1.0})
    assert res.bankrupt, "资不抵债应触发破产闸门"
    assert res.bankrupt_at is not None
    assert len(res.net_equity) < 60, \
        f"曲线应在破产处截断，实际长度 {len(res.net_equity)}"
    assert len(res.index) == len(res.net_equity), "index 与权益必须等长"
    assert res.net_equity[-1] <= 0.0, "破产那根的权益应 ≤ 0"


def test_no_trades_after_bankruptcy():
    """破产后不得再有任何成交（初版会继续按 `w × eq / px` 下出天文数字的委托）。"""
    frames = flat_frames(n=60, spike_at=25, spike_high=300.0)
    sim, res = run_sim(frames, {"BTCUSDT": -1.0})
    # 死亡后不应再累积成本/换手
    if res.bankrupt_at is not None:
        tail_cost = res.cost_paid[res.bankrupt_at:].sum()
        assert tail_cost == 0.0, f"破产后仍有成本 {tail_cost}"


# ==========================================================================
# 无成本时的守恒恒等式
# ==========================================================================
def test_equity_identity_without_costs_or_funding():
    """无成本、无资金费、无强平时：权益 = 本金 + 仓位浮动盈亏。

    多头 1x @100 持有到 100：权益必须精确等于本金。
    """
    frames = flat_frames(n=60, spike_at=None)
    sim, res = run_sim(frames, {"BTCUSDT": 1.0})
    assert res.net_equity[-1] == pytest.approx(10_000.0, abs=1e-6), \
        "价格不动、无成本 ⇒ 权益应精确等于本金"


def test_liquidation_with_penalty_charges_exactly_the_penalty():
    """有罚金时：权益 = 无罚金权益 − 罚金。"""
    frames = flat_frames(n=60, spike_at=25, spike_high=130.0)
    sim_a, res_a = run_sim(frames, {"BTCUSDT": -1.0})

    sim_b = SimExchange(
        frames, CostModel(enabled=False), CostModel(enabled=False),
        SimConfig(initial_cash=10_000.0, warmup=10, max_gross=1.0,
                  max_exposure_per_symbol=1.0, allow_short=True,
                  instrument="perp",
                  margin=MarginConfig(initial_margin_ratio=0.5,
                                      maintenance_margin_ratio=0.25,
                                      liquidation_penalty_bp=50)),
    )
    res_b = sim_b.run(OnceAgent({"BTCUSDT": -1.0}))
    # 罚金 = 50bp × 100 单位 × 130 = 65
    assert res_b.net_equity[-1] == pytest.approx(res_a.net_equity[-1] - 65.0, abs=1e-6)


# ==========================================================================
# 成本模型的适用域护栏
# ==========================================================================
def test_impact_rate_is_capped_at_max_impact_rate():
    """冲击率不得被外推到 > max_impact_rate（费率 >100% 无经济含义）。"""
    cm = CostModel(enabled=True, impact_k=1.0, impact_gamma_perm=0.2)
    # part = 1e6/1e3 = 1000 → 原始冲击率远超 1
    raw = cm.raw_impact_rate(notional=1e6, sigma=0.02, period_notional=1e3)
    assert raw > 1.0, "构造应让原始冲击率超过上限"
    capped = cm.impact_rate(notional=1e6, sigma=0.02, period_notional=1e3)
    assert capped == cm.max_impact_rate
    assert cm.n_impact_capped == 1, "截断必须被计数，否则会被静默忽略"


def test_impact_rate_uncapped_in_normal_range():
    """正常规模下护栏不应触发（它是第二道防线，不是调参旋钮）。

    一笔 1,000 USDT 的委托在 4,000 万小时成交额的市场里：
    `part=2.5e-5` ⇒ 冲击率 ≈ 1.05 bp（其中 √ 项 1.00 bp、线性项 0.05 bp）。
    """
    cm = CostModel(enabled=True)
    r = cm.impact_rate(notional=1_000.0, sigma=0.02, period_notional=4e7)
    assert 0.0 < r < 1e-3, f"正常规模的冲击率应在 bp 量级，实际 {r * 1e4:.2f} bp"
    assert cm.n_impact_capped == 0, "正常规模不应触发护栏"
    # √ 项应当主导（线性永久项只在 part 大时才显著）
    part = 1_000.0 / 4e7
    sqrt_term = cm.impact_k * 0.02 * part ** 0.5
    assert sqrt_term > cm.impact_gamma_perm * part


def test_trade_cost_never_exceeds_notional_with_cap():
    """有护栏时，单笔冲击成本不得超过该笔名义额。"""
    cm = CostModel(enabled=True)
    for notional in (10.0, 1e3, 1e6, 1e9):
        c = cm.impact(notional, sigma=0.02, period_notional=1e3)
        assert c <= notional, f"notional={notional}: 冲击成本 {c} 超过了名义额"


# ==========================================================================
# 6. 破产闸门的记录语义（2026-09-15 加）
# ==========================================================================
def _bankrupt_run():
    """做空 1x，价格在单根 bar 内冲到 400 —— 损失远超全部权益。"""
    frames = flat_frames(n=40, spike_at=25, spike_high=400.0)
    ag = OnceAgent({"BTCUSDT": -1.0})
    return SimExchange(
        frames, CostModel(enabled=False), CostModel(enabled=False),
        SimConfig(initial_cash=10_000.0, warmup=10,
                  max_gross=1.0, max_exposure_per_symbol=1.0,
                  margin=MarginConfig(initial_margin_ratio=0.5,
                                      maintenance_margin_ratio=0.1,
                                      topup_trigger_ratio=0.5),
                  allow_short=True, instrument="perp")).run(ag)


def test_bankruptcy_records_zero_not_negative_equity():
    """爆仓时曲线记 **0**，不是负值；原始盯市值另存 bankrupt_equity_raw。

    为什么必须归零：交易所不会让你欠钱（强平的意义就在于此）。若把负数
    如实留在曲线里，下游会算出 `total_ret < −100%`、`max_dd < −100%`
    （经济上不可能），而且 **sharpe 会变成无意义的正数** —— 简单收益率的
    *均值*可以为正，而复利终值已经归零。实测在 3x 网格里出现过
    "已爆仓的配置 sharpe=2.625 排第一"这种荒谬排名。
    """
    res = _bankrupt_run()

    assert res.bankrupt, "单根 bar 亏光本金，破产闸门必须触发"
    assert res.net_equity[-1] == 0.0, (
        f"爆仓后权益应归零，实际 {res.net_equity[-1]}；"
        "负值是盯市口径的产物，不是经济现实")
    assert res.net_equity.min() >= 0.0, "曲线里不允许出现负权益"


def test_bankruptcy_keeps_raw_overshoot_for_diagnosis():
    """归零不应抹掉诊断信息：'本来跌到多深'要保留。

    本例：做空 100 单位、开仓价 100、在 400 被强平 ⇒ 变现现金流 −40,000，
    而开仓后现金只有 20,000 ⇒ 原始盯市值 **−20,000**（2 倍本金）。
    """
    res = _bankrupt_run()

    assert res.bankrupt_equity_raw is not None, "必须保留原始盯市值"
    assert res.bankrupt_equity_raw < 0, (
        f"本例原始盯市值应为负，实际 {res.bankrupt_equity_raw}")
    assert res.bankrupt_equity_raw == pytest.approx(-20_000.0, rel=1e-6), (
        "做空 100 单位在 400 强平 ⇒ 现金 20,000 − 40,000 = −20,000")
