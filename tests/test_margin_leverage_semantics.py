"""保证金**参数语义**测试 —— `initial_margin_ratio` 必须由实际杠杆推出。

**背景**（详见 `src/sim/margin.py` 模块 docstring 的"第二次修复"）

2026-09-15 修掉了 `margin.py` 的三处**记账**缺陷（强平造钱、权益口径、不利价），
但**参数**仍然是错的：C/D 阶段所有脚本硬编码 `initial_margin_ratio=0.5`，
等于无条件假设每个仓位都是 2 倍杠杆。由

    k·|u|·p₀ + (p − p₀)·|u| ≤ m·|u|·p   ⇒   p/p₀ ≤ (1−k)/(1−m)

得 `|u|` 两边约掉 ⇒ **强平阈值与杠杆完全无关**，k=0.5/m=0.1 时恒为 −44.4%。
1x 满额买入没有借款，这个停损在经济上不存在。

实测后果（本文件第 4 节用真实数据的前 9,300 根 bar 复现）：
`only_BTCUSDT`（1x、只交易一次、band=0.20 ⇒ 开环永不重入）在
**bar 9104 = 2018-11-23 01:00**（BTC `low = 4,239.67`，入场价 = bar 301 的
open = 7,676.81，`low/p₀ = 0.5523 ≤ 0.5556`）被强平，此后曲线恒定 7.9 年：
终值 5,469 vs 关闭保证金的 101,251。

修法：`MarginConfig.for_leverage(L)` ⇒ `k = 1/L`，阈值变成教科书式
`p/p₀ ≤ (L−1)/(L(1−m))`：L=1 ⇒ 0（永不强平）、L=2 ⇒ 0.5556、L=3 ⇒ 0.7407。

本文件的测试按"越靠后越强"排列：解析式 → 数值（MarginBook）→ 集成（SimExchange）
→ 真实数据回归 → 已知残留缺陷（第 5 节，故意钉住还没修好的那一块）。
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
from src.sim.margin import MarginBook, MarginConfig  # noqa: E402

M_MAINT = 0.1        # 与 C/D 阶段脚本一致的维持保证金率


# ==========================================================================
# 夹具
# ==========================================================================
def flat_frames(n=60, symbol="BTCUSDT", entry=100.0, spike_at=25,
                spike_low_ratio=0.5523):
    """价格恒为 `entry`；在 `spike_at` 造一根 low 下探到 `entry × ratio`。

    默认 ratio = 0.5523 是 **2018-11-23 那根**的真实比率
    （`low 4,239.67 / 入场价 7,676.81`），刚好落在旧阈值 0.5556 之内 ——
    于是同一组数据能同时验证"旧参数会强平"与"新参数不会"。
    """
    idx = pd.date_range("2024-01-01", tz="UTC", periods=n, freq="h")
    close = np.full(n, entry, dtype=float)
    high = close * 1.001
    low = close * 0.999
    low[spike_at] = entry * spike_low_ratio
    return {symbol: pd.DataFrame({
        "open": close, "high": high, "low": low, "close": close,
        "volume": 1_000.0, "quote_volume": close * 1_000.0,
    }, index=idx)}


class OnceAgent:
    """只在第一次决策时给出权重，之后不再动作（与 `only_BTCUSDT` 的行为一致）。"""

    name = "once"

    def __init__(self, weights):
        self.weights = weights
        self._done = False

    def decide(self, view):
        if self._done:
            return None
        self._done = True
        return dict(self.weights)


class ResizeAgent:
    """先给 1x、再给 2x（**同方向、不穿过零**）—— 用来复现"锚点过期"。

    引擎只在「空→有仓 / 有仓→空 / 多空翻转」时同步保证金账本，所以这种
    "只看张"的调仓不会刷新 `margin_cash`/`entry_price`。
    """

    name = "resize"

    def __init__(self, symbol, first=1.0, second=2.0):
        self.symbol = symbol
        self.first, self.second = first, second
        self.stage = 0

    def decide(self, view):
        self.stage += 1
        if self.stage == 1:
            return {self.symbol: self.first}
        if self.stage == 2:
            return {self.symbol: self.second}
        return None


def run_sim(frames, weights, *, margin_cfg, cash=10_000.0, warmup=10,
            exposure=1.0, costs=False, agent=None):
    """跑一次仿真。costs=False 时毛净账本都关成本 ⇒ 数字可精确核对。

    `agent` 给定时用它替代 `OnceAgent(weights)`（例如 `ResizeAgent`）。
    """
    cm = CostModel(enabled=True) if costs else CostModel(enabled=False)
    sim = SimExchange(
        frames, CostModel(enabled=False), cm,
        SimConfig(initial_cash=cash, warmup=warmup, max_gross=exposure,
                  max_exposure_per_symbol=exposure, allow_short=True,
                  instrument="perp", margin=margin_cfg),
    )
    return sim.run(agent if agent is not None else OnceAgent(weights))


def _first_liquidating_ratio(cfg: MarginConfig, units=100.0, price=100.0) -> float:
    """二分找 `MarginBook` 实际触发的 p/p₀（不依赖解析式）。"""
    book = MarginBook(cfg, ["BTC"])
    book.open_leg("BTC", units=units, price=price, cash_pool=[1e12])
    lo, hi = 0.0, 10.0
    for _ in range(200):
        mid = (lo + hi) / 2.0
        if book.is_liquidatable("BTC", units, adverse_price=mid * price):
            lo = mid
        else:
            hi = mid
    return lo


# ==========================================================================
# 1. 参数推导：k = 1/L
# ==========================================================================
def test_for_leverage_sets_initial_ratio_to_inverse_of_leverage():
    for L in (1.0, 2.0, 3.0, 5.0, 0.5):
        cfg = MarginConfig.for_leverage(L)
        assert cfg.initial_margin_ratio == pytest.approx(1.0 / L), \
            f"L={L} 的初始保证金应为 1/L"
        assert cfg.maintenance_margin_ratio == pytest.approx(M_MAINT)


def test_for_leverage_rejects_default_maintenance_above_ratio():
    """L > 1/m 时 k < m ⇒ 仓位在开仓瞬间就满足强平条件，必须直接报错。

    m=0.1 ⇒ L 上限 10。这不是"更严格的风控"，是无解的参数组合。
    """
    MarginConfig.for_leverage(10.0)                      # 边界：k == m，合法
    with pytest.raises(ValueError, match="开仓瞬间"):
        MarginConfig.for_leverage(20.0)


def test_for_leverage_rejects_nonpositive_or_nan_leverage():
    for bad in (0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="leverage"):
            MarginConfig.for_leverage(bad)


def test_default_config_still_means_two_x():
    """向后兼容：默认值 0.5 的语义没有变（= for_leverage(2)）。

    这是"显式传参语义不变"的守卫 —— 既有 22 条 `test_margin_book.py` 与
    `test_sim_conservation.py` 都依赖它。
    """
    d = MarginConfig()
    assert d.initial_margin_ratio == 0.5
    assert d.long_liquidation_ratio() == pytest.approx(
        MarginConfig.for_leverage(2.0, maintenance_margin_ratio=0.25)
        .long_liquidation_ratio())
    # 默认 (k=0.5, m=0.25) ⇒ 多头 −33.33%、空头 +20%（历史单测钉的就是这两条）
    assert d.long_liquidation_ratio() == pytest.approx(2.0 / 3.0)
    assert d.short_liquidation_ratio() == pytest.approx(1.2)


# ==========================================================================
# 2. 解析式：(L−1)/(L(1−m))，L=1 ⇒ 0（永不强平）
# ==========================================================================
def test_long_liquidation_ratio_matches_textbook_formula():
    for L in (1.0, 1.5, 2.0, 3.0, 5.0):
        cfg = MarginConfig.for_leverage(L, maintenance_margin_ratio=M_MAINT)
        expected = (L - 1.0) / (L * (1.0 - M_MAINT))
        assert cfg.long_liquidation_ratio() == pytest.approx(expected), \
            f"L={L}: 阈值应为 (L−1)/(L(1−m))"


def test_one_x_threshold_is_zero_means_never():
    """**核心**：L=1 ⇒ 阈值 0，任何正价格都不触发强平。"""
    cfg = MarginConfig.for_leverage(1.0, maintenance_margin_ratio=M_MAINT)
    assert cfg.long_liquidation_ratio() == 0.0


def test_three_x_threshold_is_about_0741():
    cfg = MarginConfig.for_leverage(3.0, maintenance_margin_ratio=M_MAINT)
    assert cfg.long_liquidation_ratio() == pytest.approx(0.7407, abs=1e-4)
    assert cfg.short_liquidation_ratio() == pytest.approx(
        (1.0 + 1.0 / 3.0) / 1.1, abs=1e-9)


def test_old_hardcoded_parameter_was_exactly_two_x_regardless_of_leverage():
    """把旧参数钉在案：k=0.5 的阈值恒为 0.5556，**与 L 无关**。

    这正是缺陷的解析证据：C/D 阶段的 1x 与 3x 用的是同一条 −44.4% 强平线。
    """
    for L in (1.0, 2.0, 3.0):
        old = MarginConfig(initial_margin_ratio=0.5,
                           maintenance_margin_ratio=M_MAINT)
        assert old.long_liquidation_ratio() == pytest.approx(0.5556, abs=1e-4), \
            f"L={L}: 旧参数（写死 0.5）的阈值不应随杠杆变化"


# ==========================================================================
# 3. 数值核对（直接调 MarginBook，不依赖解析式）
# ==========================================================================
def test_margin_book_numeric_threshold_equals_analytic():
    for L in (1.0, 2.0, 3.0):
        cfg = MarginConfig.for_leverage(L, maintenance_margin_ratio=M_MAINT)
        numeric = _first_liquidating_ratio(cfg)
        analytic = cfg.long_liquidation_ratio()
        assert numeric == pytest.approx(analytic, abs=1e-6), \
            f"L={L}: 数值 {numeric:.6f} vs 解析 {analytic:.6f}"


def test_one_x_is_not_liquidatable_at_any_price():
    """1x 下把价格压到开仓价的 1e-6 倍也不该强平。"""
    cfg = MarginConfig.for_leverage(1.0, maintenance_margin_ratio=M_MAINT)
    book = MarginBook(cfg, ["BTC"])
    book.open_leg("BTC", units=100.0, price=100.0, cash_pool=[1e12])
    for r in (0.5, 0.1, 0.01, 1e-6):
        assert not book.is_liquidatable("BTC", 100.0, adverse_price=r * 100.0), \
            f"1x 满额多头在 p/p₀={r} 被强平 —— 这就是那个不存在的停损"


def test_three_x_long_is_liquidated_just_below_the_threshold():
    cfg = MarginConfig.for_leverage(3.0, maintenance_margin_ratio=M_MAINT)
    book = MarginBook(cfg, ["BTC"])
    book.open_leg("BTC", units=300.0, price=100.0, cash_pool=[1e12])
    assert not book.is_liquidatable("BTC", 300.0, adverse_price=75.0)   # 0.750 > 0.7407
    assert book.is_liquidatable("BTC", 300.0, adverse_price=73.0)       # 0.730 < 0.7407


# ==========================================================================
# 4. 集成：SimExchange 上的可观测行为（同一条规则、同一组数据）
# ==========================================================================
def test_one_x_full_long_survives_the_historical_drawdown_geometry():
    """1x 满额多头 + 2018-11-23 那根的真实跌幅比率（0.5523）⇒ **不该强平**。

    旧参数（k=0.5）在这组数据上必然强平（0.5523 ≤ 0.5556）—— 下一条测试钉住它。
    """
    frames = flat_frames(spike_low_ratio=0.5523)
    cfg = MarginConfig.for_leverage(1.0, maintenance_margin_ratio=M_MAINT)
    res = run_sim(frames, {"BTCUSDT": 1.0}, margin_cfg=cfg)
    assert res.liquidated_legs == [], \
        f"1x 满额持仓在 −44.8% 回撤处被强平：{res.liquidated_legs}"


def test_legacy_parameter_liquidates_the_same_position():
    """对照组：同一组数据、同一策略，只用旧参数 ⇒ 强平。证明上一条测试有效力。"""
    frames = flat_frames(spike_low_ratio=0.5523)
    cfg = MarginConfig(initial_margin_ratio=0.5,
                       maintenance_margin_ratio=M_MAINT)
    res = run_sim(frames, {"BTCUSDT": 1.0}, margin_cfg=cfg)
    assert res.liquidated_legs == ["BTCUSDT"], \
        "旧参数（k=0.5）在 −44.8% 回撤处本应强平 —— 对照失效则本条测试无意义"


def test_margin_on_at_one_x_is_identical_to_margin_off():
    """**最强的不变量**：1x 下 margin ON（k=1）与 margin OFF 的净权益曲线**逐点相同**。

    既然 1x 没有借款，保证金通道就不该改变任何数字。旧参数下两条曲线在
    2018-11-23 分歧（本文件第 4 节标题所述的事件），终值 5,469 vs 101,251。
    """
    frames = flat_frames(spike_low_ratio=0.5523)
    on = run_sim(frames, {"BTCUSDT": 1.0},
                 margin_cfg=MarginConfig.for_leverage(1.0, maintenance_margin_ratio=M_MAINT))
    off = SimExchange(
        frames, CostModel(enabled=False), CostModel(enabled=False),
        SimConfig(initial_cash=10_000.0, warmup=10, max_gross=1.0,
                  max_exposure_per_symbol=1.0, allow_short=True,
                  instrument="perp", margin=None),
    ).run(OnceAgent({"BTCUSDT": 1.0}))
    assert np.array_equal(on.net_equity, off.net_equity), \
        "k=1 时保证金通道不得改变任何权益数字"


def test_three_x_liquidation_lands_on_the_derived_threshold():
    """L=3 下同一根 low：0.750 不触发、0.730 触发（集成层面复核对齐阈值）。"""
    def liquidated_at(ratio):
        res = run_sim(
            flat_frames(spike_low_ratio=ratio), {"BTCUSDT": 3.0},
            margin_cfg=MarginConfig.for_leverage(3.0, maintenance_margin_ratio=M_MAINT),
            exposure=3.0)
        return res.liquidated_legs != []

    assert not liquidated_at(0.750), "0.750 > 阈值 0.7407 不该强平"
    assert liquidated_at(0.730), "0.730 < 阈值 0.7407 应该强平"


# ==========================================================================
# 5. 已知残留缺陷（故意钉住，防止被遗忘）
# ==========================================================================
def test_known_residual_partial_deployment_still_has_a_phantom_stop():
    """**还没修好的那一块**：`k = 1/max_gross` 只在"确实用满杠杆"时正确。

    引擎按 `w × 权益 / 价` 下仓，而 `k` 取自 `max_gross`（配置上界），
    两者不是一回事。于是 `max_gross=3` 而策略只持 1x（现金恰好归零、**没有借款**）时，
    `k=1/3` 仍给出 `p/p₀ ≤ 0.7407` 的强平线（−25.9%）—— 同一个错误的小号版本。

    账本级真值：满额 1x 持仓 `equity = E·(p/p₀)`，维持保证金 `0.1·E·(p/p₀)`，
    条件 `r ≤ 0.1r` 恒假 ⇒ **永不强平**。所以这是残留缺陷，不是新语义。

    修它需要"按实际名义额/权益推 k"（per-leg 杠杆），那要动
    `src/sim/exchange.py` 的保证金同步路径（目前只在开仓/平仓/翻向时同步，
    加仓扣减不触发），**不在本次改动范围内**。本测试的作用是把这条限制钉住，
    并给出量级：L=3 的配置下，未用满杠杆的仓位仍有约 −25.9% 的隐形停损。
    """
    cfg = MarginConfig.for_leverage(3.0, maintenance_margin_ratio=M_MAINT)
    # 只持 1x（weights=1.0 而 max_gross=3 ⇒ 名义额 = 权益，现金归零，无借款）
    res = run_sim(flat_frames(spike_low_ratio=0.73), {"BTCUSDT": 1.0},
                  margin_cfg=cfg, exposure=3.0)
    assert res.liquidated_legs == ["BTCUSDT"], (
        "若这里不再强平，说明残留缺陷已被修掉 —— 请更新本测试与 margin.py 的说明")
    assert cfg.long_liquidation_ratio() == pytest.approx(0.7407, abs=1e-4)


def test_liquidation_description_is_human_readable():
    """`liquidation_description()` 要能进 run 的 config（参数自解释）。"""
    assert "永不强平" in MarginConfig.for_leverage(1.0).liquidation_description()
    d3 = MarginConfig.for_leverage(3.0, maintenance_margin_ratio=M_MAINT)\
        .liquidation_description()
    assert "0.7407" in d3 and "25.9%" in d3


# ==========================================================================
# 6. 第三条缺陷：锚点过期（纯诊断，不改变行为）
# ==========================================================================
def test_stale_anchor_counter_flags_resize_without_reanchor():
    """仓位被调大而不穿过零 ⇒ 锚点过期 ⇒ 计数器必须看得见。

    为什么会有这条：`exchange.py` 只在「空→有仓 / 有仓→空 / 多空翻转」时调用
    `open_leg`（同步 `margin_cash` / `entry_price`）。仓位被**调大调小却不穿零**
    时，账本仍锚在旧仓位，权益式 `margin_cash + (p−p₀)·|u_现|` 不等于 `|u_现|·p`。
    """
    cfg = MarginConfig.for_leverage(1.0, maintenance_margin_ratio=M_MAINT)
    book = MarginBook(cfg, ["BTC"])
    book.open_leg("BTC", units=100.0, price=100.0, cash_pool=[1e12])
    assert book.n_stale_anchor_bars == 0
    assert book.legs["BTC"].units_at_anchor == 100.0

    # 规模涨到 10 倍（引擎不会为此调 open_leg）
    book.is_liquidatable("BTC", 1000.0, adverse_price=100.0)
    assert book.n_stale_anchor_bars == 1
    assert book.stale_anchor_symbols == {"BTC"}

    # 同一规模再来一根 bar 不会重复计数（它就是过期状态，没变）
    book.is_liquidatable("BTC", 1000.0, adverse_price=100.0)
    assert book.n_stale_anchor_bars == 2, "逐 bar 计数，不是逐次变更"

    # 平仓/重开仓后锚点归零 ⇒ 不再算过期
    book.close_leg("BTC", 1000.0, price=100.0, cash_pool=[1e12])
    assert book.legs["BTC"].units_at_anchor == 0.0
    book.is_liquidatable("BTC", 0.0, adverse_price=100.0)
    assert book.n_stale_anchor_bars == 2


def test_stale_anchor_liquidates_a_one_x_long_at_its_entry_price():
    """**锚点过期能把 1x 多头"强平"在建仓价上** —— 连价格都没动。

    100 单位 @100 建仓（k=1 ⇒ margin_cash=10,000）；规模涨到 1,000 单位
    （价格不变）后：权益 = 10,000 + 0·1,000 = 10,000，
    维持保证金 = 0.1 × 1,000 × 100 = 10,000 ⇒ **判定强平**。
    这是账本缺陷，不是经济含义 —— 1x 满额持仓没有借款。
    """
    cfg = MarginConfig.for_leverage(1.0, maintenance_margin_ratio=M_MAINT)
    book = MarginBook(cfg, ["BTC"])
    book.open_leg("BTC", units=100.0, price=100.0, cash_pool=[1e12])
    assert book.is_liquidatable("BTC", 1000.0, adverse_price=100.0), \
        "锚点过期的 1x 多头会被判在建仓价强平 —— 这正是要钉住的缺陷"


def test_stale_anchor_reproduces_the_minus_44_percent_line_end_to_end():
    """端到端：把仓位从 1x 调到 2x（不穿零）+ k=1 ⇒ 又出现一条 −44.4% 的隐形停损。

    与旧参数那条线**数值相同纯属代数巧合**，机制完全不同：
    旧的原因是 `k=0.5`；这里 `k=1` 是对的，错的是锚点没刷新。

    仓位先 1x（units=100）后 2x（units=200，价仍 100），`margin_cash` 仍是
    第一次的 10,000 ⇒ 权益 = 10,000 + (p−100)·200，在 p=55 时为 1,000，
    维持保证金 = 0.1×200×55 = 1,100 ⇒ 强平。**关闭保证金时不会强平。**
    """
    frames = flat_frames(n=60, entry=100.0, spike_at=40, spike_low_ratio=0.55)

    on = run_sim(frames, None, agent=ResizeAgent("BTCUSDT", 1.0, 2.0),
                 margin_cfg=MarginConfig.for_leverage(1.0, maintenance_margin_ratio=M_MAINT),
                 exposure=3.0)
    assert on.liquidated_legs == ["BTCUSDT"], (
        "锚点过期 + 规模翻倍 + k=1 本应复现那条隐形停损；若不再强平，"
        "说明 exchange.py 的保证金同步已被修好 —— 请更新本测试")

    off = SimExchange(
        frames, CostModel(enabled=False), CostModel(enabled=False),
        SimConfig(initial_cash=10_000.0, warmup=10, max_gross=3.0,
                  max_exposure_per_symbol=3.0, allow_short=True,
                  instrument="perp", margin=None),
    ).run(ResizeAgent("BTCUSDT", 1.0, 2.0))
    assert off.liquidated_legs == [], "关闭保证金时不该有任何强平"


def test_liquidated_stale_records_only_liquidations_while_anchor_is_stale():
    """把"空头腿的正常强平"与"锚点过期造成的假强平"分开数。

    这个计数器让 `pool_ceiling` 能回答："某候选的 N 次强平里，几次发生在
    锚点过期状态"。它**不改变**任何强平行为，只是记录。
    """
    cfg = MarginConfig.for_leverage(1.0, maintenance_margin_ratio=M_MAINT)

    # (a) 锚点新鲜（规模与建仓时一致）时强平 ⇒ 不算"过期态强平"
    fresh = MarginBook(cfg, ["BTC"])
    fresh.open_leg("BTC", units=100.0, price=100.0, cash_pool=[1e12])
    fresh.liquidate("BTC", units=100.0, exit_price=50.0, cash_pool=[1e12])
    assert fresh.liquidated_stale == []

    # (b) 规模被放大过但锚点没刷新 ⇒ 记录为过期态强平
    stale = MarginBook(cfg, ["BTC"])
    stale.open_leg("BTC", units=100.0, price=100.0, cash_pool=[1e12])
    stale.is_liquidatable("BTC", 1000.0, adverse_price=100.0)
    stale.liquidate("BTC", units=1000.0, exit_price=100.0, cash_pool=[1e12])
    assert stale.liquidated_stale == ["BTC"]
