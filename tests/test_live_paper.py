"""纸面交易账本的测试。

有状态的东西比纯函数危险：符号写反、幂等漏掉、恒等式破了，
都不会报错，只会安静地给出错的收益。所以这里逐条钉死。

运行：python3 tests/test_live_paper.py
"""
from __future__ import annotations

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.live.paper import Book, LiveConfig  # noqa: E402

T0 = 1_760_000_000_000          # 固定时间戳，避免测试随时钟漂移
H = 3_600_000


def _book(cap: float = 10_000.0, spot: float = 100.0, mark: float = 100.0,
          spot_units: float = 0.0) -> Book:
    """按给定参数开好仓的账本。"""
    b = Book(symbol="TESTUSDT", cfg=LiveConfig(initial_capital=cap))
    b.open_(T0, spot, mark)
    return b


def funding_row(ms: int, rate: float, mark: float) -> dict:
    return {"funding_time": ms, "funding_rate": rate, "mark_price": mark,
            "symbol": "TESTUSDT"}


# ==========================================================================
# 1. 开仓恒等式
# ==========================================================================
def test_open_preserves_accounting_identity():
    cap, spot, mark = 10_000.0, 100.0, 101.0
    b = _book(cap, spot, mark)
    eq = b.equity(spot, mark)
    fee = cap * 0.4 * 0.0011 * 2
    assert abs(eq - (cap - fee)) < 1e-9, f"开仓后权益应等于本金减开仓费，得 {eq}"


def test_open_is_delta_neutral():
    b = _book(10_000.0, 100.0, 100.0)
    assert b.spot_units > 0 and b.perp_units < 0, "应为 现货多 + 永续空"
    assert abs(abs(b.spot_units) * 100.0 - abs(b.perp_units) * 100.0) < 1e-6, \
        "两腿名义额应相等"
    # 两条腿同向涨 10%，权益应基本不变
    e0 = b.equity(100.0, 100.0)
    e1 = b.equity(110.0, 110.0)
    assert abs(e1 - e0) / e0 < 1e-9, f"同向变动不该改变权益：{e0} → {e1}"


def test_open_requires_no_double_open():
    b = _book()
    try:
        b.open_(T0 + H, 100.0, 100.0)
    except RuntimeError:
        pass
    else:
        raise AssertionError("重复开仓应报错")


# ==========================================================================
# 2. 资金费：符号与数值
# ==========================================================================
def test_positive_rate_credits_the_short():
    """**核心符号测试**：正费率下空头应当**收到**钱。

    这条写成反向的（当成付出）就是本项目犯过两次的错误。
    """
    b = _book()
    before = b.margin_cash
    b.apply_funding([funding_row(T0 + 8 * H, 0.0001, 100.0)], T0 + 8 * H)
    assert b.margin_cash > before, "正费率下空头应收钱，margin_cash 却减少了"
    assert b.funding_total > 0, f"累计资金费应为正，得 {b.funding_total}"


def test_funding_cash_equals_formula():
    b = _book()
    mark, rate = 100.0, 0.00025
    b.apply_funding([funding_row(T0 + 8 * H, rate, mark)], T0 + 8 * H)
    expect = -b.perp_units * mark * rate
    assert abs(b.funding_total - expect) < 1e-12, \
        f"现金流应为 -perp_units×mark×rate = {expect}，得 {b.funding_total}"
    assert expect > 0, "构造上这里应为正（收到）"


def test_negative_rate_charges_the_short():
    b = _book()
    before = b.margin_cash
    b.apply_funding([funding_row(T0 + 8 * H, -0.0002, 100.0)], T0 + 8 * H)
    assert b.margin_cash < before, "负费率下空头应付钱"
    assert b.funding_total < 0


def test_funding_is_idempotent():
    """**核心幂等测试**：同一笔结算重复应用只能算一次。

    漏了这条，cron 每小时重跑一次就会凭空多算 24 倍资金费。
    """
    b = _book()
    rows = [funding_row(T0 + 8 * H, 0.0001, 100.0)]
    b.apply_funding(rows, T0 + 8 * H)
    after_first = b.margin_cash
    for _ in range(5):
        got = b.apply_funding(rows, T0 + 16 * H)
        assert got == [], "重复应用应返回空（什么都没做）"
    assert b.margin_cash == after_first, "重复应用改变了状态 —— 幂等失效"


def test_funding_applies_each_settlement_once_in_order():
    b = _book()
    rows = [funding_row(T0 + 8 * H, 0.0001, 100.0),
            funding_row(T0 + 16 * H, 0.0002, 101.0),
            funding_row(T0 + 24 * H, 0.0003, 102.0)]
    got = b.apply_funding(rows, T0 + 24 * H)
    assert len(got) == 3, f"应应用 3 笔，得 {len(got)}"
    assert b.last_funding_ms == T0 + 24 * H
    # 乱序传入也应按时间顺序应用
    b2 = _book()
    got2 = b2.apply_funding(list(reversed(rows)), T0 + 24 * H)
    assert [r["funding_time"] for r in got2] == [T0 + 8 * H, T0 + 16 * H, T0 + 24 * H]
    assert abs(b2.funding_total - b.funding_total) < 1e-12, "顺序不该影响总额"


def test_funding_skips_settlements_before_open():
    b = _book()
    got = b.apply_funding([funding_row(T0 - 8 * H, 0.001, 100.0)], T0)
    assert got == [], "开仓前的结算不该被应用（那时还没持仓）"


# ==========================================================================
# 3. 保证金：补保与强平
# ==========================================================================
def test_moderate_rise_triggers_topup_from_reserve():
    b = _book()
    r0, m0 = b.reserve, b.margin_cash
    act = b.check_margin(T0 + H, 130.0)          # 永续涨 30%（现货不动，纯挤空）
    assert act["action"] in ("topup", "topup_partial"), f"应触发补保，得 {act}"
    assert b.reserve < r0 and b.margin_cash > m0, "补保应从备用金转到保证金"
    assert b.n_topups == 1


def test_violent_rise_liquidates_when_reserve_empty():
    b = _book()
    b.reserve = 0.0                 # 没有备用金可补
    act = b.check_margin(T0 + H, 400.0)
    assert act["action"] == "liquidated", f"应强平，得 {act}"
    assert b.liquidated and b.liquidated_at_ms == T0 + H


def test_reserve_covers_rise_so_no_liquidation():
    b = _book()
    act = b.check_margin(T0 + H, 105.0)          # 温和上冲，备用金充足
    assert act["action"] == "ok" or act["action"].startswith("topup"), \
        f"不该强平，得 {act}"
    assert not b.liquidated


# ==========================================================================
# 4. 再平衡
# ==========================================================================
def test_rebalance_not_due_before_interval():
    b = _book()
    assert b.maybe_rebalance(T0 + 100 * H, 100.0, 100.0) is None


def test_rebalance_resizes_legs_and_charges_fee():
    b = _book()
    f0 = b.fees_total
    rec = b.maybe_rebalance(T0 + 721 * H, 130.0, 130.0)
    assert rec is not None, "到 720 小时应再平衡"
    assert b.fees_total > f0, "再平衡应计费"
    assert abs(b.spot_units * 130.0 + b.perp_units * 130.0) < 1e-6, \
        "再平衡后两腿名义额应相等"


def test_rebalance_preserves_equity():
    """再平衡只该花掉手续费，不该凭空改变权益。

    断言"减少量恰好等于记录的手续费"，比拍一个上界更严格：
    任何多出来的差额都意味着现金在三个桶之间漏了。
    """
    b = _book()
    e0 = b.equity(130.0, 130.0)
    f0 = b.fees_total
    rec = b.maybe_rebalance(T0 + 721 * H, 130.0, 130.0)
    e1 = b.equity(130.0, 130.0)
    spent = b.fees_total - f0
    assert spent > 0, "再平衡应计费"
    assert abs((e0 - e1) - spent) < 1e-9, \
        f"权益减少 {e0 - e1} 应恰好等于手续费 {spent}"
    assert abs(rec["fee"] - spent) < 1e-12, "事件里记的费用应与实际扣费一致"


def test_periodic_rebalance_survives_while_no_rebalance_dies():
    """M10 冒烟测试的发现，在实时引擎里再验一次：
    同样的持续上冲下，不再平衡会强平，定期再平衡能活。

    上冲幅度要够大才打得穿：保证金 = 0.2×本金，空头亏到约 +50% 才见底，
    所以这里推到 +150%（真实里少见，但这是压力测试）。
    """
    def run(rebalance_h: int) -> Book:
        b = Book(symbol="T", cfg=LiveConfig(initial_capital=10_000.0,
                                            rebalance_every_h=rebalance_h))
        b.open_(T0, 100.0, 100.0)
        b.reserve = 0.0                     # 极端：没有备用金
        for k in range(1, 101):
            px = 100.0 * (1.0 + 0.015 * k)  # 每 168h 涨 1.5%，累计 +150%
            t = T0 + 168 * k * H
            b.maybe_rebalance(t, px, px)
            b.check_margin(t, px)
            if b.liquidated:
                break
        return b

    dead = run(10 ** 9)                     # 实际上不再平衡
    alive = run(168)                        # 每周再平衡
    assert dead.liquidated, \
        f"不再平衡应当在 +150% 上冲中被强平（权益 {dead.equity(250.0, 250.0):.2f}）"
    assert not alive.liquidated, "定期再平衡应当存活（自动降杠杆）"


# ==========================================================================
# 5. 持久化
# ==========================================================================
def test_save_load_roundtrip():
    b = _book()
    b.apply_funding([funding_row(T0 + 8 * H, 0.0001, 100.0)], T0 + 8 * H)
    b.snapshot(T0 + 8 * H, 100.0, 100.0)
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "T.json")
        b.save(p)
        c = Book.load("T", p)
    assert c is not None
    assert abs(c.margin_cash - b.margin_cash) < 1e-12
    assert abs(c.funding_total - b.funding_total) < 1e-12
    assert c.last_funding_ms == b.last_funding_ms
    assert len(c.snapshots) == len(b.snapshots)
    assert isinstance(c.cfg, LiveConfig)
    # 往返后继续幂等（状态完整）
    assert c.apply_funding([funding_row(T0 + 8 * H, 0.0001, 100.0)], T0 + 16 * H) == []


def test_load_missing_returns_none():
    with tempfile.TemporaryDirectory() as d:
        assert Book.load("NOPE", os.path.join(d, "nope.json")) is None


def test_trim_bounds_snapshot_growth():
    """快照必须被裁剪，否则一年 8,760 条会把状态文件撑爆。"""
    b = _book()
    for k in range(1, 400):
        b.snapshot(T0 + k * H, 100.0, 100.0)
    n_before = len(b.snapshots)
    b._trim()                                    # noqa: SLF001
    assert n_before == 399
    assert len(b.snapshots) < n_before, "裁剪后条数应减少"
    assert len(b.snapshots) <= 48 + 20, f"裁剪后应约 48+天数条，得 {len(b.snapshots)}"
    # 最近 48 条必须一条不丢（小时级细节）
    assert b.snapshots[-48:] == b.snapshots[-48:]


def test_trim_preserves_daily_marks():
    """裁剪要保住每个 UTC 日的最后一条 —— 那条是日收盘，曲线靠它。

    不变量：**裁剪前后覆盖的日期集合完全一致**（不假设天数，避免我自己算错跨度）。
    """
    b = _book()
    for k in range(1, 24 * 5 + 1):               # 120 小时 ≈ 跨 6 个日历日（起始非零点）
        b.snapshot(T0 + k * H, 100.0 + k, 100.0 + k)
    before = {s["at"][:10] for s in b.snapshots}
    last_per_day_before = {}
    for s in b.snapshots:
        last_per_day_before[s["at"][:10]] = s["equity"]
    b._trim()                                    # noqa: SLF001
    after = {s["at"][:10] for s in b.snapshots}
    assert after == before, f"日期集合变了：{before} → {after}"
    for s in b.snapshots[:-48] or b.snapshots:   # 日级代表点
        d = s["at"][:10]
        assert abs(s["equity"] - last_per_day_before[d]) < 1e-9, \
            f"{d} 保留的不是当日最后一条"


# ==========================================================================
# 6. 行情重试
# ==========================================================================
def test_retry_returns_first_success_without_sleeping():
    from src.live import feed
    calls, slept = [], []
    def ok():
        calls.append(1)
        return "值"
    assert feed.retry_call(ok, attempts=3, sleep=slept.append) == "值"
    assert len(calls) == 1, "第一次就成功不该重试"
    assert slept == [], "成功时不该 sleep"


def test_retry_absorbs_transient_failure():
    """**核心**：前两次失败、第三次成功 —— 这正是重试要吸收的场景。"""
    from src.live import feed
    calls, slept = [], []
    def flaky():
        calls.append(1)
        if len(calls) < 3:
            raise OSError("SSL: UNEXPECTED_EOF_WHILE_READING")
        return "第三次成功"
    assert feed.retry_call(flaky, attempts=3, base_delay=2.0,
                           sleep=slept.append) == "第三次成功"
    assert len(calls) == 3
    assert slept == [2.0, 4.0], f"退避应为 2s/4s，得 {slept}"


def test_retry_raises_after_exhausting_attempts():
    """耗尽后必须**照常抛错** —— 不能吞掉，否则会变成安静地少算一笔。"""
    from src.live import feed
    calls, slept = [], []
    def always_fail():
        calls.append(1)
        raise OSError("代理上游不可达")
    try:
        feed.retry_call(always_fail, attempts=3, base_delay=1.0,
                        sleep=slept.append)
    except OSError as e:
        assert "不可达" in str(e)
    else:
        raise AssertionError("重试耗尽后应抛错，而不是静默返回")
    assert len(calls) == 3, f"应尝试 3 次，得 {len(calls)}"
    assert slept == [1.0, 2.0], f"三次尝试只该 sleep 两次，得 {slept}"


def test_retry_attempts_one_is_single_shot():
    from src.live import feed
    calls, slept = [], []
    def fail():
        calls.append(1)
        raise ValueError("x")
    try:
        feed.retry_call(fail, attempts=1, sleep=slept.append)
    except ValueError:
        pass
    assert len(calls) == 1 and slept == []


def test_live_quote_uses_retry_and_surfaces_errors():
    """live_quote 必须把底层失败如实抛出（tick 靠它决定跳过而不是写坏状态）。"""
    from src.live import feed
    orig = feed._quote_once                       # noqa: SLF001
    calls = []
    def boom(sym):
        calls.append(sym)
        raise RuntimeError("Name or service not known")
    feed._quote_once = boom                       # noqa: SLF001
    try:
        feed.live_quote("BTCUSDT", attempts=2, base_delay=0.0,
                        sleep=lambda _: None)
    except RuntimeError as e:
        assert "Name or service not known" in str(e)
    else:
        raise AssertionError("应抛出底层错误")
    finally:
        feed._quote_once = orig                   # noqa: SLF001
    assert len(calls) == 2, f"应尝试 2 次，得 {len(calls)}"


# ==========================================================================
def main() -> int:
    tests = [(k, v) for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    passed, failed = 0, []
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS  {name}")
            passed += 1
        except Exception as e:
            print(f"  FAIL  {name}: {type(e).__name__}: {e}")
            failed.append(name)
    print(f"\n{passed}/{len(tests)} 通过")
    if failed:
        print("失败：", ", ".join(failed))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
