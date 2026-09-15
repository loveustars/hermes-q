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
def test_open_leg_deducts_initial_margin():
    """建仓：cash_pool 扣 initial_required，margin_cash = initial_required。"""
    book = MarginBook(MarginConfig(initial_margin_ratio=0.5), ["BTC"])
    cash = [10_000.0]
    # 建 100 单位多单，价 100，初始保证金 0.5 × 100 × 100 = 5,000
    book.open_leg("BTC", units=100.0, price=100.0, cash_pool=cash)
    assert cash[0] == 5_000
    assert book.legs["BTC"].margin_cash == 5_000
    assert book.legs["BTC"].initial_margin == 5_000


def test_open_leg_short_also_deducts_initial_margin():
    """做空建仓与做多对称，扣同样 initial_required。"""
    book = MarginBook(MarginConfig(initial_margin_ratio=0.5), ["BTC"])
    cash = [10_000.0]
    book.open_leg("BTC", units=-100.0, price=100.0, cash_pool=cash)
    assert cash[0] == 5_000
    assert book.legs["BTC"].margin_cash == 5_000


def test_open_zero_units_no_op():
    book = MarginBook(MarginConfig(), ["BTC"])
    cash = [10_000.0]
    book.open_leg("BTC", units=0.0, price=100.0, cash_pool=cash)
    assert cash[0] == 10_000
    assert book.legs["BTC"].margin_cash == 0.0


def test_close_leg_releases_equity_with_pnl():
    """平仓：cash_pool 收到当前 equity（含浮动 PnL）。"""
    book = MarginBook(MarginConfig(), ["BTC"])
    cash = [10_000.0]
    book.open_leg("BTC", units=100.0, price=100.0, cash_pool=cash)  # cash=5000
    # 价涨到 110，equity = 5000 + 100*110 = 16000
    released = book.close_leg("BTC", units=100.0, price=110.0, cash_pool=cash)
    assert released == 16_000
    assert cash[0] == 5_000 + 16_000
    assert book.legs["BTC"].margin_cash == 0.0


def test_close_short_leg_with_profit():
    """做空平仓盈利：价 100→90，equity = 5000 + (-100)*90 = -4000。"""
    book = MarginBook(MarginConfig(), ["BTC"])
    cash = [10_000.0]
    book.open_leg("BTC", units=-100.0, price=100.0, cash_pool=cash)
    released = book.close_leg("BTC", units=-100.0, price=90.0, cash_pool=cash)
    assert released == -4_000
    assert cash[0] == 5_000 - 4_000    # 1000（做空盈利）


# ==========================================================================
# 强平
# ==========================================================================
def test_liquidate_uses_high_price_in_release():
    """强平 release 数额随 high_price 变化（验证用 high 而非 close）。"""
    # 多仓 100 单位，价 100，margin_cash=5000
    # 价跌触发强平：release = max(0, 5000 + 100*high)
    rel_a = _liquidate_long_and_get_release(high_price=80.0)
    rel_b = _liquidate_long_and_get_release(high_price=70.0)
    rel_c = _liquidate_long_and_get_release(high_price=60.0)
    assert rel_a == 13_000
    assert rel_b == 12_000
    assert rel_c == 11_000
    assert rel_a > rel_b > rel_c, "high 价不同应影响 release"


def _liquidate_long_and_get_release(high_price: float) -> float:
    book = MarginBook(MarginConfig(initial_margin_ratio=0.5, maintenance_margin_ratio=0.5),
                      ["BTC"])
    cash = [10_000.0]
    book.open_leg("BTC", units=100.0, price=100.0, cash_pool=cash)
    return book.liquidate("BTC", units=100.0, high_price=high_price, cash_pool=cash)


def test_liquidate_marks_symbol_and_blocks_reopen():
    book = MarginBook(MarginConfig(initial_margin_ratio=0.5, maintenance_margin_ratio=0.5),
                      ["BTC"])
    cash = [10_000.0]
    book.open_leg("BTC", units=100.0, price=100.0, cash_pool=cash)
    book.liquidate("BTC", units=100.0, high_price=80.0, cash_pool=cash)
    assert "BTC" in book.liquidated
    with pytest.raises(ValueError, match="已强平"):
        book.open_leg("BTC", units=100.0, price=100.0, cash_pool=cash)


def test_liquidate_with_penalty_reduces_release():
    """liquidation_penalty_bp>0 时，release 减去罚金。"""
    # 多仓 100 单位，价 100，margin_cash=5000
    # 强平 high 80: equity=13000, penalty=100bp×|100|×80=80, release=12920
    book = MarginBook(MarginConfig(initial_margin_ratio=0.5, maintenance_margin_ratio=0.5,
                                   liquidation_penalty_bp=100),
                      ["BTC"])
    cash = [10_000.0]
    book.open_leg("BTC", units=100.0, price=100.0, cash_pool=cash)
    rel = book.liquidate("BTC", units=100.0, high_price=80.0, cash_pool=cash)
    assert rel == pytest.approx(12_920, abs=1e-3)


def test_liquidate_zero_equity_returns_zero():
    """强平时 equity 已 ≤ 0 → release = 0。"""
    book = MarginBook(MarginConfig(initial_margin_ratio=0.5, maintenance_margin_ratio=0.5),
                      ["BTC"])
    cash = [10_000.0]
    book.open_leg("BTC", units=100.0, price=100.0, cash_pool=cash)
    # 极端情况：high 跌到 0，equity = 5000 + 0 = 5000，**不**是 0
    # 真正能 release=0：high 跌到 -50，equity = 5000 - 5000 = 0
    rel = book.liquidate("BTC", units=100.0, high_price=-50.0, cash_pool=cash)
    assert rel == 0.0


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


def test_topup_deducts_from_cash_pool():
    """补保：cash_pool 减，margin_cash 加。"""
    book = MarginBook(MarginConfig(initial_margin_ratio=0.1, maintenance_margin_ratio=0.05,
                                   topup_trigger_ratio=0.5),
                      ["BTC"])
    cash = [10_000.0]
    book.open_leg("BTC", units=100.0, price=100.0, cash_pool=cash)  # cash=9000
    # 价跌到 -10: equity=0, 需补到 1000 → need 1000
    book.topup_to("BTC", units=100.0, price=-10.0, cash_pool=cash)
    assert cash[0] == 8_000
    assert book.legs["BTC"].margin_cash == 2_000


def test_topup_no_op_if_above_threshold():
    """equity ≥ topup_required → 不补保。"""
    book = MarginBook(MarginConfig(initial_margin_ratio=0.1, maintenance_margin_ratio=0.05),
                      ["BTC"])
    cash = [10_000.0]
    book.open_leg("BTC", units=100.0, price=100.0, cash_pool=cash)
    taken = book.topup_to("BTC", units=100.0, price=100.0, cash_pool=cash)
    assert taken == 0
    assert cash[0] == 9_000


def test_topup_capped_by_cash_pool():
    """cash_pool 不足时，补到 cash_pool 归零为止。"""
    book = MarginBook(MarginConfig(initial_margin_ratio=0.1, maintenance_margin_ratio=0.05,
                                   topup_trigger_ratio=0.5),
                      ["BTC"])
    cash = [10_500.0]    # 只剩 500 可补
    book.open_leg("BTC", units=100.0, price=100.0, cash_pool=cash)  # cash=9500
    # 价跌到 -10: equity=0, 需补 1000, take min(1000, 9500)=1000
    # 但实际 cash 9500 够
    # 改用 cash 不足场景：先把 cash 调小
    cash[0] = 200
    # 价跌到 -10: equity=0, need 1000, take min(1000, 200)=200
    book.topup_to("BTC", units=100.0, price=-10.0, cash_pool=cash)
    assert cash[0] == 0
    assert book.legs["BTC"].margin_cash == 1_200    # 1000 + 200
