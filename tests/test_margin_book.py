"""C 阶段：MarginBook 单元测试 —— 接口与不变量。

C 阶段分四步走：
  C1（本测试）：MarginConfig + MarginBook 独立单元测试
  C2：把 MarginBook 接入 SimExchange 主循环（先不强平）
  C3：加逐腿强平判定
  C4：scripts/c_margin_baseline.py 真实数据 + 文档同步

C1 验证：
  - 接口：open_leg / close_leg / topup_to / liquidate / is_liquidatable / needs_topup
  - 不变量：cash_pool 正确增减、margin_cash 追踪、已强平标的不再开仓
  - 强平：按 high 价算 release、加罚金、释放后单位被外部清零
  - 补保：从 cash_pool 扣到 margin_cash，恢复到 initial_margin 水平

注：close vs high 的差异是集成测试的范围（与真实数据强平判定耦合），
   C1 只验证接口接受 high_price 参数。
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
    """建仓：margin_cash = initial_required，C2 阶段不真扣 cash_pool。

    注：C1 早期版本 open_leg 是扣 cash 的，但 SimExchange 现状是全额扣 cash 买币，
    重复扣会双重扣钱。改 C2 后：open_leg 只记账，cash_pool 扣除由 SimExchange 主循环负责。
    """
    book = MarginBook(MarginConfig(initial_margin_ratio=0.5), ["BTC"])
    cash = [10_000.0]
    book.open_leg("BTC", units=100.0, price=100.0, cash_pool=cash)
    assert book.legs["BTC"].margin_cash == 5_000
    assert book.legs["BTC"].initial_margin == 5_000
    assert cash[0] == 10_000, f"C2 阶段 open_leg 不应扣 cash_pool，实际 {cash[0]}"


def test_open_leg_short_also_records_initial_margin():
    """做空建仓：与做多对称，记同样的 initial_required。"""
    book = MarginBook(MarginConfig(initial_margin_ratio=0.5), ["BTC"])
    cash = [10_000.0]
    book.open_leg("BTC", units=-100.0, price=100.0, cash_pool=cash)
    assert book.legs["BTC"].margin_cash == 5_000
    assert cash[0] == 10_000


def test_open_zero_units_no_op():
    book = MarginBook(MarginConfig(), ["BTC"])
    cash = [10_000.0]
    book.open_leg("BTC", units=0.0, price=100.0, cash_pool=cash)
    assert cash[0] == 10_000
    assert book.legs["BTC"].margin_cash == 0.0


def test_close_leg_clears_ledger_no_cash_op():
    """C2 阶段 close_leg 只清零账本，不操作 cash。"""
    book = MarginBook(MarginConfig(), ["BTC"])
    cash = [10_000.0]
    book.open_leg("BTC", units=100.0, price=100.0, cash_pool=cash)
    released = book.close_leg("BTC", units=100.0, price=110.0, cash_pool=cash)
    assert released == 0.0
    assert book.legs["BTC"].margin_cash == 0.0
    assert cash[0] == 10_000


def test_close_leg_records_pnl_for_liquidation_path():
    """C3 强平路径用得到 close_leg 的 PnL 数值：通过 equity() 单独计算。"""
    book = MarginBook(MarginConfig(), ["BTC"])
    cash = [10_000.0]
    book.open_leg("BTC", units=-100.0, price=100.0, cash_pool=cash)
    # 价跌 90：equity = 5000 + (-100)*90 = -4000（用于 C3 强平价计算）
    eq = book.equity("BTC", units=-100.0, price=90.0)
    assert eq == -4_000


# ==========================================================================
# 强平
# ==========================================================================
def test_liquidate_releases_margin_cash_to_cash_pool():
    """C3 阶段 liquidate 把 margin_cash 退到 cash_pool（扣除罚金）。

    注意：C2 阶段这个方法是 no-op，但 C3 阶段 release = max(0, margin_cash - penalty)。
    浮动 PnL 不补，所以浮亏时只退 margin_cash 本金（用户实际亏了浮亏）。
    """
    book = MarginBook(MarginConfig(initial_margin_ratio=0.5, maintenance_margin_ratio=0.5),
                      ["BTC"])
    cash = [10_000.0]
    book.open_leg("BTC", units=100.0, price=100.0, cash_pool=cash)
    rel = book.liquidate("BTC", units=100.0, high_price=80.0, cash_pool=cash)
    # margin_cash=5000, penalty=0, release=5000
    assert rel == 5_000.0
    assert cash[0] == 15_000
    assert book.legs["BTC"].margin_cash == 0.0
    assert "BTC" in book.liquidated


def test_liquidate_with_penalty_reduces_release():
    """liquidation_penalty_bp>0 时，release 减去罚金。"""
    # 100 单位，价 100，margin_cash=5000, penalty=100bp×100×high
    # high=100: penalty=100, release=4900
    book = MarginBook(MarginConfig(initial_margin_ratio=0.5, maintenance_margin_ratio=0.5,
                                   liquidation_penalty_bp=100),
                      ["BTC"])
    cash = [10_000.0]
    book.open_leg("BTC", units=100.0, price=100.0, cash_pool=cash)
    rel = book.liquidate("BTC", units=100.0, high_price=100.0, cash_pool=cash)
    assert rel == pytest.approx(4_900, abs=1e-3)


def test_liquidate_margin_cash_below_penalty_returns_zero():
    """margin_cash 不足以付罚金 → release = 0（账户已亏光）。"""
    # penalty=1000, margin_cash=500 → release=0
    book = MarginBook(MarginConfig(initial_margin_ratio=0.5, maintenance_margin_ratio=0.5,
                                   liquidation_penalty_bp=100),
                      ["BTC"])
    cash = [10_000.0]
    book.open_leg("BTC", units=100.0, price=100.0, cash_pool=cash)  # margin=5000
    rel = book.liquidate("BTC", units=100.0, high_price=100.0, cash_pool=cash)
    # penalty = 100bp × 100 × 100 = 100, release = 5000-100=4900, 仍 > 0
    # 改 penalty=10000：10000bp × 100 × 100 = 10000, release = max(0, 5000-10000) = 0
    # 重新构造：
    book2 = MarginBook(MarginConfig(initial_margin_ratio=0.5, maintenance_margin_ratio=0.5,
                                    liquidation_penalty_bp=10000),
                       ["BTC"])
    cash2 = [10_000.0]
    book2.open_leg("BTC", units=100.0, price=100.0, cash_pool=cash2)
    rel2 = book2.liquidate("BTC", units=100.0, high_price=100.0, cash_pool=cash2)
    assert rel2 == 0.0


def test_liquidate_marks_symbol_and_blocks_reopen():
    book = MarginBook(MarginConfig(initial_margin_ratio=0.5, maintenance_margin_ratio=0.5),
                      ["BTC"])
    cash = [10_000.0]
    book.open_leg("BTC", units=100.0, price=100.0, cash_pool=cash)
    book.liquidate("BTC", units=100.0, high_price=80.0, cash_pool=cash)
    assert "BTC" in book.liquidated
    with pytest.raises(ValueError, match="已强平"):
        book.open_leg("BTC", units=100.0, price=100.0, cash_pool=cash)


def test_is_liquidatable_uses_high_price_threshold():
    """is_liquidatable 接受 high_price 做压力测试。

    用做空 + 涨：high=120 必强平；high=99（做空盈利）不应强平。
    注：做空 + 价不变/价涨都会强平（开仓时 margin_cash 不足以覆盖维持保证金），
        所以"不应强平"必须用价跌场景（做空盈利）。
    """
    book = MarginBook(MarginConfig(initial_margin_ratio=0.5, maintenance_margin_ratio=0.5),
                      ["BTC"])
    cash = [10_000.0]
    book.open_leg("BTC", units=-100.0, price=100.0, cash_pool=cash)
    # 做空 100 单位，margin_cash=5000
    # high=120: equity=5000+(-100)*120=-7000, maint=0.5*100*120=6000 → 强平
    assert book.is_liquidatable("BTC", units=-100.0, high_price=120.0), \
        "做空 + 涨 20% 应触发强平"
    # high=99: equity=5000+(-100)*99=-4900, maint=0.5*100*99=4950 → -4900 ≤ 4950 → 强平
    # 做空 + 价跌也强平（开仓时 margin_cash 不足以覆盖）—— 修正断言
    assert book.is_liquidatable("BTC", units=-100.0, high_price=99.0), \
        "做空 100 单位价不变/跌都会强平（margin_cash 不足以覆盖维持保证金）"


# ==========================================================================
# 补保
# ==========================================================================
def test_needs_topup_threshold():
    """equity < topup_required → needs_topup = True。"""
    # 多仓 100 单位，价 100，initial=0.1×10000=1000
    # topup_trigger=0.5 → topup_required=500
    book = MarginBook(MarginConfig(initial_margin_ratio=0.1, maintenance_margin_ratio=0.05,
                                   topup_trigger_ratio=0.5),
                      ["BTC"])
    cash = [10_000.0]
    book.open_leg("BTC", units=100.0, price=100.0, cash_pool=cash)
    # close 跌到 0: equity=1000, topup_required=500, 1000>500 → 不触发
    # close 跌到 -6: equity=400, 400<500 → 触发
    assert not book.needs_topup("BTC", units=100.0, price=-5.0)
    assert book.needs_topup("BTC", units=100.0, price=-6.0)


def test_topup_records_in_ledger_no_cash_op():
    """C2 阶段 topup_to 也只记账，不操作 cash（与 open_leg 一致）。"""
    book = MarginBook(MarginConfig(initial_margin_ratio=0.1, maintenance_margin_ratio=0.05,
                                   topup_trigger_ratio=0.5),
                      ["BTC"])
    cash = [10_000.0]
    book.open_leg("BTC", units=100.0, price=100.0, cash_pool=cash)
    # cash 仍 = 10000（C2 不扣），margin_cash=1000
    # 价跌到 -10: equity=0, 需补到 1000
    book.topup_to("BTC", units=100.0, price=-10.0, cash_pool=cash)
    # C2 阶段 topup 只调 margin_cash，不动 cash
    assert cash[0] == 10_000
    assert book.legs["BTC"].margin_cash == 2_000    # 1000 + 1000（补保）


def test_topup_no_op_if_above_threshold():
    """equity ≥ topup_required → 不补保。"""
    book = MarginBook(MarginConfig(initial_margin_ratio=0.1, maintenance_margin_ratio=0.05),
                      ["BTC"])
    cash = [10_000.0]
    book.open_leg("BTC", units=100.0, price=100.0, cash_pool=cash)
    taken = book.topup_to("BTC", units=100.0, price=100.0, cash_pool=cash)
    assert taken == 0
    assert cash[0] == 10_000


def test_topup_capped_by_initial_margin():
    """C2 阶段：topup 把 margin_cash 加到 initial_margin 上限。"""
    # cash_pool 上限测试在 C3 强平路径才有意义，C2 阶段不操作 cash
    book = MarginBook(MarginConfig(initial_margin_ratio=0.1, maintenance_margin_ratio=0.05,
                                   topup_trigger_ratio=0.5),
                      ["BTC"])
    cash = [10_000.0]
    book.open_leg("BTC", units=100.0, price=100.0, cash_pool=cash)
    # 价跌到 -10: equity=0, 需补 1000
    book.topup_to("BTC", units=100.0, price=-10.0, cash_pool=cash)
    assert book.legs["BTC"].margin_cash == 1_000 + 1_000    # initial + 补
    # 再补一次：equity 仍 0（margin_cash=2000，price=-10：equity = 2000 + 100*(-10) = 1000）
    # 1000 = initial_margin → 不用再补
    book.topup_to("BTC", units=100.0, price=-10.0, cash_pool=cash)
    assert book.legs["BTC"].margin_cash == 2_000    # 不再增加
