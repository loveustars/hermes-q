"""C 阶段：MarginBook 单元测试 —— 接口与不变量。

**2026-09-15 修复说明**：本文件的多条断言原本**编码了错误语义**，随
`src/sim/margin.py` 的三处修复一起重写。保留此说明是为了后来的读者能看懂
"为什么测试被改了"——改测试在这里不是掩盖问题，而是问题本身就在测试里：

  - `test_liquidate_releases_margin_cash_to_cash_pool` 原本断言 `cash == 15_000`，
    即**把"强平时凭空多出 0.5×名义额"写成了正确行为**。
  - `test_is_liquidatable_uses_high_price_threshold` 原本带着这样的注释：
    "做空 + 价跌也强平（开仓时 margin_cash 不足以覆盖）—— 修正断言"。
    作者**观察到空头在盈利时也被强平**，却去迁就行为而不是修代码。
  - `test_needs_topup_threshold` 用了**负价格**（`price=-5.0`）——只有权益公式
    错成"绝对市值"时才可能出现的输入。
  - `test_topup_*` 断言补保不动现金，而补保会在账面凭空增加保证金。

修复后的口径（见 `src/sim/margin.py` 模块 docstring）：
  - 浮动盈亏按**建仓价**算：`(price − entry_price) × units`，多空自动对称
  - 强平 = **在强平价上强制平仓**：`cash += units × 强平价 − 罚金`，不退保证金
  - 压力测试用**不利价**：多头 low、空头 high
  - 补保**真实从现金池扣款**，以可用现金为上限
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest  # noqa: E402

from src.sim.margin import MarginBook, MarginConfig  # noqa: E402


# ==========================================================================
# 配置校验
# ==========================================================================
def test_margin_config_rejects_inverted_ratios():
    with pytest.raises(ValueError, match="必须 ≤"):
        MarginConfig(initial_margin_ratio=0.5, maintenance_margin_ratio=0.6)


def test_margin_config_rejects_bad_topup_ratio():
    with pytest.raises(ValueError, match="topup_trigger_ratio"):
        MarginConfig(initial_margin_ratio=0.5, maintenance_margin_ratio=0.25,
                     topup_trigger_ratio=1.5)
    with pytest.raises(ValueError, match="topup_trigger_ratio"):
        MarginConfig(initial_margin_ratio=0.5, maintenance_margin_ratio=0.25,
                     topup_trigger_ratio=0.0)


def test_margin_config_rejects_negative_penalty():
    with pytest.raises(ValueError, match="liquidation_penalty_bp"):
        MarginConfig(liquidation_penalty_bp=-1.0)


# ==========================================================================
# 建仓 / 平仓
# ==========================================================================
def test_open_leg_records_initial_margin_in_ledger():
    """建仓：记下 margin_cash 与 build 价，但不扣 cash_pool。

    不扣的理由：SimExchange 主循环建仓时已按名义额全额扣现金，
    这里再扣会双重扣钱。
    """
    book = MarginBook(MarginConfig(initial_margin_ratio=0.5), ["BTC"])
    cash = [10_000.0]
    book.open_leg("BTC", units=100.0, price=100.0, cash_pool=cash)
    assert book.legs["BTC"].margin_cash == 5_000
    assert book.legs["BTC"].initial_margin == 5_000
    assert book.legs["BTC"].entry_price == 100.0
    assert cash[0] == 10_000, f"open_leg 不应扣 cash_pool，实际 {cash[0]}"


def test_open_leg_short_also_records_initial_margin():
    """做空建仓：与做多对称，记同样的 initial_required 与建仓价。"""
    book = MarginBook(MarginConfig(initial_margin_ratio=0.5), ["BTC"])
    cash = [10_000.0]
    book.open_leg("BTC", units=-100.0, price=100.0, cash_pool=cash)
    assert book.legs["BTC"].margin_cash == 5_000
    assert book.legs["BTC"].entry_price == 100.0
    assert cash[0] == 10_000


def test_open_zero_units_no_op():
    book = MarginBook(MarginConfig(), ["BTC"])
    cash = [10_000.0]
    book.open_leg("BTC", units=0.0, price=100.0, cash_pool=cash)
    assert cash[0] == 10_000
    assert book.legs["BTC"].margin_cash == 0.0


def test_close_leg_clears_ledger_no_cash_op():
    """close_leg 只清零账本，不操作 cash（主循环的 cash -= d×px 已处理）。"""
    book = MarginBook(MarginConfig(), ["BTC"])
    cash = [10_000.0]
    book.open_leg("BTC", units=100.0, price=100.0, cash_pool=cash)
    released = book.close_leg("BTC", units=100.0, price=110.0, cash_pool=cash)
    assert released == 0.0
    assert book.legs["BTC"].margin_cash == 0.0
    assert book.legs["BTC"].entry_price == 0.0
    assert cash[0] == 10_000


# ==========================================================================
# 浮动盈亏口径（修复点 2）
# ==========================================================================
def test_equity_uses_entry_relative_pnl_not_absolute_value():
    """权益按**相对建仓价**的浮动盈亏算，不是仓位绝对市值。

    做空 100 单位 @100，价跌到 90 应**盈利** +1000，权益 6000。
    初版写成 `margin_cash + units × price = 5000 − 9000 = −4000`，
    那是把空头的负市值当负债，于是空头在开仓瞬间权益就恒为负。
    """
    book = MarginBook(MarginConfig(), ["BTC"])
    cash = [10_000.0]
    book.open_leg("BTC", units=-100.0, price=100.0, cash_pool=cash)
    assert book.pnl("BTC", units=-100.0, price=90.0) == pytest.approx(1_000.0)
    assert book.equity("BTC", units=-100.0, price=90.0) == pytest.approx(6_000.0)


def test_equity_long_profits_when_price_rises():
    book = MarginBook(MarginConfig(), ["BTC"])
    book.open_leg("BTC", units=100.0, price=100.0, cash_pool=[10_000.0])
    assert book.pnl("BTC", units=100.0, price=110.0) == pytest.approx(1_000.0)
    assert book.equity("BTC", units=100.0, price=110.0) == pytest.approx(6_000.0)


def test_equity_is_symmetric_at_entry():
    """多空在建仓价的权益必须相等且为正 —— 初版空头在此为 −0.5N。"""
    book = MarginBook(MarginConfig(initial_margin_ratio=0.5), ["A", "B"])
    book.open_leg("A", units=100.0, price=100.0, cash_pool=[1e9])
    book.open_leg("B", units=-100.0, price=100.0, cash_pool=[1e9])
    assert book.equity("A", units=100.0, price=100.0) == pytest.approx(5_000.0)
    assert book.equity("B", units=-100.0, price=100.0) == pytest.approx(5_000.0)


# ==========================================================================
# 强平（修复点 1）
# ==========================================================================
def test_liquidate_realizes_position_and_does_not_refund_margin():
    """强平 = 按强平价强制平仓：`cash += units × 强平价`，**不退保证金**。

    初版是 `cash += margin_cash`（= 0.5×名义额），而那笔保证金从未被单独
    借记过 —— 每强平一次白得半个名义额，做空时收益恒为正，是 D 阶段
    "权益冲到 2.1 亿"的直接原因。
    """
    book = MarginBook(MarginConfig(initial_margin_ratio=0.5,
                                   maintenance_margin_ratio=0.25), ["BTC"])
    cash = [10_000.0]
    book.open_leg("BTC", units=100.0, price=100.0, cash_pool=cash)
    # 多头被强平在 low=70：变现 100 × 70 = 7000
    realized = book.liquidate("BTC", units=100.0, exit_price=70.0, cash_pool=cash)
    assert realized == pytest.approx(7_000.0)
    assert cash[0] == pytest.approx(17_000.0)   # 不是 15_000（初版）
    assert book.legs["BTC"].margin_cash == 0.0
    assert book.legs["BTC"].entry_price == 0.0
    assert "BTC" in book.liquidated


def test_liquidate_short_realizes_negative_units():
    """空头被强平：units<0，变现为负（要花钱买回），不再是"白得"。"""
    book = MarginBook(MarginConfig(initial_margin_ratio=0.5,
                                   maintenance_margin_ratio=0.25), ["BTC"])
    cash = [10_000.0]
    book.open_leg("BTC", units=-100.0, price=100.0, cash_pool=cash)
    realized = book.liquidate("BTC", units=-100.0, exit_price=120.0, cash_pool=cash)
    assert realized == pytest.approx(-12_000.0)
    assert cash[0] == pytest.approx(-2_000.0)


def test_liquidate_with_penalty_deducts_from_realization():
    book = MarginBook(MarginConfig(initial_margin_ratio=0.5,
                                   maintenance_margin_ratio=0.25,
                                   liquidation_penalty_bp=100), ["BTC"])
    cash = [10_000.0]
    book.open_leg("BTC", units=100.0, price=100.0, cash_pool=cash)
    # 变现 100×100=10000，罚金 100bp × 100 × 100 = 100 → net 9900
    realized = book.liquidate("BTC", units=100.0, exit_price=100.0, cash_pool=cash)
    assert realized == pytest.approx(9_900.0, abs=1e-3)
    assert cash[0] == pytest.approx(19_900.0, abs=1e-3)


def test_liquidation_is_cash_neutral_when_no_penalty():
    """**核心不变量**：无罚金时，强平不改变权益。

    强平前权益 = cash + units×价；强平后 = (cash + units×价) + 0。
    两边相等 ⇒ 强平只是"把仓位按当前价变现"，既不造钱也不烧钱。
    """
    for units, exit_px in ((100.0, 70.0), (-100.0, 120.0), (100.0, 130.0)):
        book = MarginBook(MarginConfig(initial_margin_ratio=0.5,
                                       maintenance_margin_ratio=0.25), ["BTC"])
        cash = [10_000.0]
        book.open_leg("BTC", units=units, price=100.0, cash_pool=cash)
        eq_before = cash[0] + units * exit_px
        book.liquidate("BTC", units=units, exit_price=exit_px, cash_pool=cash)
        eq_after = cash[0] + 0.0
        assert eq_after == pytest.approx(eq_before), \
            f"units={units} exit={exit_px}: 强平改变了权益 {eq_before} → {eq_after}"


def test_liquidate_marks_symbol_but_allows_reopen():
    """强平后 symbol 仍可重新开仓（discard liquidated 标记）。"""
    book = MarginBook(MarginConfig(initial_margin_ratio=0.5,
                                   maintenance_margin_ratio=0.25), ["BTC"])
    cash = [10_000.0]
    book.open_leg("BTC", units=100.0, price=100.0, cash_pool=cash)
    book.liquidate("BTC", units=100.0, exit_price=70.0, cash_pool=cash)
    assert "BTC" in book.liquidated
    book.open_leg("BTC", units=50.0, price=80.0, cash_pool=cash)
    assert "BTC" not in book.liquidated, "重新开仓应解除 liquidated 标记"
    assert book.legs["BTC"].margin_cash == 0.5 * 50 * 80  # 2000
    assert book.legs["BTC"].entry_price == 80.0


# ==========================================================================
# 强平触发线（修复点 2 + 3）
# ==========================================================================
def test_is_liquidatable_long_uses_low_short_uses_high():
    """多头的不利价是 low、空头是 high —— 用 adverse_price 选。"""
    assert MarginBook.adverse_price(100.0, low=70.0, high=130.0) == 70.0
    assert MarginBook.adverse_price(-100.0, low=70.0, high=130.0) == 130.0


def test_is_liquidatable_thresholds_are_symmetric_and_sane():
    """2x 杠杆（initial=0.5, maintenance=0.25）下的强平线：

      多头：价跌到 66.67（−33%）触发
      空头：价涨到 120（+20%）触发
    且**建仓价处两个方向都不触发** —— 初版空头在建仓瞬间就判强平。
    """
    cfg = MarginConfig(initial_margin_ratio=0.5, maintenance_margin_ratio=0.25)

    long_book = MarginBook(cfg, ["BTC"])
    long_book.open_leg("BTC", units=100.0, price=100.0, cash_pool=[1e9])
    assert not long_book.is_liquidatable("BTC", 100.0, adverse_price=100.0), \
        "建仓价处不应强平"
    assert not long_book.is_liquidatable("BTC", 100.0, adverse_price=67.0)
    assert long_book.is_liquidatable("BTC", 100.0, adverse_price=66.0)

    short_book = MarginBook(cfg, ["BTC"])
    short_book.open_leg("BTC", units=-100.0, price=100.0, cash_pool=[1e9])
    assert not short_book.is_liquidatable("BTC", -100.0, adverse_price=100.0), \
        "建仓价处不应强平（初版这里恒为 True）"
    assert not short_book.is_liquidatable("BTC", -100.0, adverse_price=119.0)
    assert short_book.is_liquidatable("BTC", -100.0, adverse_price=121.0)


def test_is_liquidatable_false_when_short_profits():
    """空头盈利（价跌）绝不能判强平 —— 初版把这条注释掉改成了"应强平"。"""
    book = MarginBook(MarginConfig(initial_margin_ratio=0.5,
                                   maintenance_margin_ratio=0.25), ["BTC"])
    book.open_leg("BTC", units=-100.0, price=100.0, cash_pool=[1e9])
    assert not book.is_liquidatable("BTC", -100.0, adverse_price=80.0)
    assert not book.is_liquidatable("BTC", -100.0, adverse_price=50.0)


# ==========================================================================
# 补保（修复点：不再凭空造钱）
# ==========================================================================
def test_needs_topup_threshold_with_real_prices():
    """权益跌破 topup_required → 需补保。用真实价格，不用负价。"""
    book = MarginBook(MarginConfig(initial_margin_ratio=0.1,
                                   maintenance_margin_ratio=0.05,
                                   topup_trigger_ratio=0.5), ["BTC"])
    book.open_leg("BTC", units=100.0, price=100.0, cash_pool=[1e9])
    # initial_margin = 0.1×100×100 = 1000，topup_required = 500
    # 价 96：pnl = (96−100)×100 = −400 → equity = 600 > 500 → 不触发
    assert not book.needs_topup("BTC", units=100.0, price=96.0)
    # 价 94：pnl = −600 → equity = 400 < 500 → 触发
    assert book.needs_topup("BTC", units=100.0, price=94.0)


def test_topup_debits_cash_pool():
    """补保**真实从现金池扣款** —— 初版只加 margin_cash 不动现金。"""
    book = MarginBook(MarginConfig(initial_margin_ratio=0.1,
                                   maintenance_margin_ratio=0.05,
                                   topup_trigger_ratio=0.5), ["BTC"])
    cash = [10_000.0]
    book.open_leg("BTC", units=100.0, price=100.0, cash_pool=cash)
    # 价 90：pnl = −1000 → equity = 1000 − 1000 = 0，需补到 initial_margin=1000
    taken = book.topup_to("BTC", units=100.0, price=90.0, cash_pool=cash)
    assert taken == pytest.approx(1_000.0)
    assert cash[0] == pytest.approx(9_000.0), "补保必须扣现金"
    assert book.legs["BTC"].margin_cash == pytest.approx(2_000.0)  # 1000 初始 + 1000 补


def test_topup_capped_by_available_cash():
    """现金不够时补不满 —— 不能凭空补出来。"""
    book = MarginBook(MarginConfig(initial_margin_ratio=0.1,
                                   maintenance_margin_ratio=0.05,
                                   topup_trigger_ratio=0.5), ["BTC"])
    cash = [300.0]
    book.open_leg("BTC", units=100.0, price=100.0, cash_pool=cash)
    taken = book.topup_to("BTC", units=100.0, price=90.0, cash_pool=cash)
    assert taken == pytest.approx(300.0)
    assert cash[0] == pytest.approx(0.0)
    assert book.legs["BTC"].margin_cash == pytest.approx(1_300.0)


def test_topup_no_op_if_above_threshold():
    book = MarginBook(MarginConfig(initial_margin_ratio=0.1,
                                   maintenance_margin_ratio=0.05), ["BTC"])
    cash = [10_000.0]
    book.open_leg("BTC", units=100.0, price=100.0, cash_pool=cash)
    taken = book.topup_to("BTC", units=100.0, price=100.0, cash_pool=cash)
    assert taken == 0
    assert cash[0] == 10_000
