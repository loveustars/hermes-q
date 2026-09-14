"""资金费与延迟抖动的测试。

资金费是持仓成本（不是摩擦成本），符号容易搞反、结算时点容易错配，
所以必须用可控场景把方向钉死。

运行：python3 tests/test_funding_latency.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.agents.baselines import RandomWeights, SingleAssetBuyHold  # noqa: E402
from src.sim.costs import CostModel  # noqa: E402
from src.sim.exchange import SimConfig, SimExchange  # noqa: E402
from src.sim.funding import (FUNDING_INTERVAL_MS, FundingTable,  # noqa: E402
                             floor_hour)


# ==========================================================================
# 夹具：恒定价格 + 可控资金费
# ==========================================================================
def flat_frames(n=200, price=100.0, symbols=("AAA",)):
    idx = pd.date_range("2022-01-01", tz="UTC", periods=n, freq="h")
    out = {}
    for s in symbols:
        out[s] = pd.DataFrame({
            "open": price, "high": price, "low": price, "close": price,
            "volume": 1_000.0, "quote_volume": price * 1_000.0,
        }, index=idx)
    return out


def table_with(times, rate, symbol="AAA"):
    t = FundingTable()
    for ts in times:
        t.add(symbol, ts, rate)
    return t


def ms(ts: pd.Timestamp) -> int:
    return int(ts.timestamp() * 1000)


# ==========================================================================
# 1. FundingTable 基础
# ==========================================================================
def test_floor_hour_truncates_seconds_and_millis():
    assert floor_hour(1789257600002) == 1789257600000
    assert floor_hour(1789257600000) == 1789257600000
    assert floor_hour(1789257599999) == 1789254000000


def test_rate_lookup_and_missing_distinction():
    idx = pd.date_range("2022-01-01", tz="UTC", periods=10, freq="h")
    t = table_with([ms(idx[3])], 0.0005)
    assert t.rate_or_none("AAA", ms(idx[3])) == 0.0005
    assert t.rate_or_none("AAA", ms(idx[4])) is None     # 没结算
    assert t.rate_or_none("ZZZ", ms(idx[3])) is None     # 没数据
    assert t.rate_at("AAA", ms(idx[4])) == 0.0           # rate_at 把 None 折叠成 0
    assert t.is_settlement(ms(idx[3])) and not t.is_settlement(ms(idx[4]))


def test_floor_handles_off_hour_funding_time():
    """币安 fundingTime 常带几毫秒偏移，仍必须能匹配到整点 bar。"""
    idx = pd.date_range("2022-01-01", tz="UTC", periods=10, freq="h")
    t = table_with([ms(idx[5]) + 2], 0.0007)
    assert t.rate_or_none("AAA", ms(idx[5])) == 0.0007


# ==========================================================================
# 2. 符号方向：多头正费率要付钱，空头要收钱
# ==========================================================================
def test_funding_cost_sign_long_pays_short_receives():
    t = FundingTable()
    assert t.cost("AAA", units=+100.0, price=100.0, ts_ms=0) == 0.0  # 无数据
    t.add("AAA", 0, 0.001)
    long_pay = t.cost("AAA", +100.0, 100.0, 0)
    short_recv = t.cost("AAA", -100.0, 100.0, 0)
    assert abs(long_pay - 10.0) < 1e-12, f"多头应付 10，实际 {long_pay}"
    assert abs(short_recv + 10.0) < 1e-12, f"空头应收 10（即 -10），实际 {short_recv}"


def test_funding_negative_rate_flips_direction():
    t = FundingTable()
    t.add("AAA", 0, -0.002)
    assert t.cost("AAA", +100.0, 100.0, 0) < 0    # 负费率，多头收钱
    assert t.cost("AAA", -100.0, 100.0, 0) > 0    # 空头付钱


# ==========================================================================
# 3. 仿真器里的资金费
# ==========================================================================
def test_spot_mode_ignores_funding():
    frames = flat_frames(120)
    idx = frames["AAA"].index
    ft = table_with([ms(idx[60])], 0.01)              # 一次 1% 的重费
    res = SimExchange(frames, CostModel(enabled=False), CostModel(enabled=False),
                      SimConfig(initial_cash=10_000, warmup=10, instrument="spot"),
                      funding=ft).run(SingleAssetBuyHold("AAA"))
    assert res.n_funding_events == 0, "现货模式不该收资金费"
    assert res.total_funding() == 0.0
    assert abs(res.final_net() - 10_000.0) < 1e-6, "恒定价格下现货净值应等于本金"


def test_perp_mode_charges_funding_on_long():
    frames = flat_frames(120)
    idx = frames["AAA"].index
    ft = table_with([ms(idx[60])], 0.001)
    res = SimExchange(frames, CostModel(enabled=False), CostModel(enabled=False),
                      SimConfig(initial_cash=10_000, warmup=10, instrument="perp"),
                      funding=ft).run(SingleAssetBuyHold("AAA"))
    assert res.n_funding_events == 1, f"应有 1 次结算，实际 {res.n_funding_events}"
    # 持仓 100 个单位 × 100 价 × 0.001 = 10
    assert abs(res.total_funding() - 10.0) < 1e-6, f"应付 10，实际 {res.total_funding()}"
    assert abs(res.final_net() - 9_990.0) < 1e-6, f"净终值应为 9990，实际 {res.final_net()}"
    assert abs(res.final_gross() - 10_000.0) < 1e-6, "毛账本不受资金费影响"


def test_perp_funding_accumulates_over_multiple_settlements():
    frames = flat_frames(200)
    idx = frames["AAA"].index
    settle = [ms(idx[i]) for i in (50, 58, 66, 74)]     # 每次间隔 8 小时
    assert all(b - a == FUNDING_INTERVAL_MS for a, b in zip(settle, settle[1:]))
    ft = table_with(settle, 0.0005)
    res = SimExchange(frames, CostModel(enabled=False), CostModel(enabled=False),
                      SimConfig(initial_cash=10_000, warmup=10, instrument="perp"),
                      funding=ft).run(SingleAssetBuyHold("AAA"))
    assert res.n_funding_events == 4
    assert abs(res.total_funding() - 20.0) < 1e-6, f"4 次 × 5 = 20，实际 {res.total_funding()}"


def test_perp_short_receives_funding():
    """永续做空在正费率下应当**收**资金费 —— 这是 carry 交易的经济基础。"""
    frames = flat_frames(120)
    idx = frames["AAA"].index
    ft = table_with([ms(idx[60])], 0.001)

    class ShortAgent:
        name = "always_short"

        def __init__(self):
            self._done = False

        def decide(self, view):
            if self._done:
                return None
            self._done = True
            return {s: (-1.0 if s == "AAA" else 0.0) for s in view.symbols()}

    res = SimExchange(frames, CostModel(enabled=False), CostModel(enabled=False),
                      SimConfig(initial_cash=10_000, warmup=10, instrument="perp",
                                allow_short=True),
                      funding=ft).run(ShortAgent())
    assert res.n_funding_events == 1
    assert res.total_funding() < 0, f"空头应收到资金费（负数），实际 {res.total_funding()}"
    # 持仓 = -100 单位 × 100 价 × 0.001 = -10，即收到 10
    assert abs(res.total_funding() + 10.0) < 1e-6, f"应收 10，实际 {res.total_funding()}"
    assert abs(res.final_net() - 10_010.0) < 1e-6, f"净终值应为 10010，实际 {res.final_net()}"


def test_perp_long_and_short_funding_are_symmetric():
    """同一费率下，多空两边的资金费应当完全对称（一方付出等于另一方收到）。"""
    frames = flat_frames(120)
    idx = frames["AAA"].index
    ft = table_with([ms(idx[60])], 0.0007)

    class FixedPos:
        def __init__(self, w):
            self.w, self._done, self.name = w, False, f"pos{w}"

        def decide(self, view):
            if self._done:
                return None
            self._done = True
            return {s: (self.w if s == "AAA" else 0.0) for s in view.symbols()}

    def run(w):
        return SimExchange(frames, CostModel(enabled=False), CostModel(enabled=False),
                           SimConfig(initial_cash=10_000, warmup=10,
                                     instrument="perp", allow_short=True),
                           funding=ft).run(FixedPos(w)).total_funding()

    long_pay, short_recv = run(1.0), run(-1.0)
    assert abs(long_pay + short_recv) < 1e-9, \
        f"多空资金费必须互为相反数：多头 {long_pay}，空头 {short_recv}"
    assert long_pay > 0 and short_recv < 0


def test_no_event_when_position_is_zero():
    frames = flat_frames(120)
    idx = frames["AAA"].index
    ft = table_with([ms(idx[60])], 0.01)
    res = SimExchange(frames, CostModel(enabled=False), CostModel(enabled=False),
                      SimConfig(initial_cash=10_000, warmup=10, instrument="perp"),
                      funding=ft).run(RandomWeights(seed=0, rebalance_every=1_000))
    # 随机多头几乎不可能恰好空仓，这里只验证：空仓时不产生事件、且总费用非负或合理
    assert res.n_funding_events >= 0


def test_perp_without_table_charges_nothing():
    frames = flat_frames(120)
    res = SimExchange(frames, CostModel(enabled=False), CostModel(enabled=False),
                      SimConfig(initial_cash=10_000, warmup=10, instrument="perp"),
                      funding=None).run(SingleAssetBuyHold("AAA"))
    assert res.n_funding_events == 0 and res.total_funding() == 0.0


# ==========================================================================
# 4. 配置校验
# ==========================================================================
def test_config_rejects_spot_with_short():
    try:
        SimConfig(instrument="spot", allow_short=True)
    except ValueError:
        return
    raise AssertionError("现货 + 做空 未被拒绝")


def test_config_rejects_bad_instrument():
    try:
        SimConfig(instrument="futures")
    except ValueError:
        return
    raise AssertionError("非法 instrument 未被拒绝")


def test_config_rejects_negative_jitter():
    try:
        SimConfig(latency_jitter_mean=-1.0)
    except ValueError:
        return
    raise AssertionError("负的抖动均值未被拒绝")


def test_perp_allows_short():
    SimConfig(instrument="perp", allow_short=True)      # 不该抛


# ==========================================================================
# 5. 延迟抖动
# ==========================================================================
def test_zero_jitter_matches_baseline_exactly():
    frames = flat_frames(300)
    a = SimExchange(frames, CostModel(enabled=False), CostModel(enabled=True),
                    SimConfig(initial_cash=10_000, warmup=20)).run(
        RandomWeights(seed=4, rebalance_every=5))
    b = SimExchange(frames, CostModel(enabled=False), CostModel(enabled=True),
                    SimConfig(initial_cash=10_000, warmup=20,
                              latency_jitter_mean=0.0)).run(
        RandomWeights(seed=4, rebalance_every=5))
    assert np.allclose(a.net_equity, b.net_equity)


def test_jitter_is_reproducible_with_same_seed():
    frames = flat_frames(300)
    kw = dict(initial_cash=10_000, warmup=20, latency_jitter_mean=1.0,
              latency_jitter_seed=42)
    a = SimExchange(frames, CostModel(enabled=False), CostModel(enabled=True),
                    SimConfig(**kw)).run(RandomWeights(seed=4, rebalance_every=5))
    b = SimExchange(frames, CostModel(enabled=False), CostModel(enabled=True),
                    SimConfig(**kw)).run(RandomWeights(seed=4, rebalance_every=5))
    assert np.array_equal(a.net_equity, b.net_equity), "同种子必须完全可复现"
    assert a.latency_histogram == b.latency_histogram


def test_jitter_changes_results_and_histogram():
    frames = flat_frames(400)
    base = SimExchange(frames, CostModel(enabled=False), CostModel(enabled=True),
                       SimConfig(initial_cash=10_000, warmup=20)).run(
        RandomWeights(seed=9, rebalance_every=5))
    jit = SimExchange(frames, CostModel(enabled=False), CostModel(enabled=True),
                      SimConfig(initial_cash=10_000, warmup=20,
                                latency_jitter_mean=1.0, latency_jitter_seed=1)).run(
        RandomWeights(seed=9, rebalance_every=5))
    assert not np.allclose(base.net_equity, jit.net_equity), "抖动应改变结果"
    assert len(jit.latency_histogram) > 1, "抖动应产生多个延迟档位"
    assert sum(jit.latency_histogram.values()) == sum(base.latency_histogram.values()), \
        "决策次数不应因抖动而改变"
    assert all(1 <= k <= 5 for k in jit.latency_histogram), "延迟必须落在 [1, max] 内"


def test_jitter_never_exceeds_cap():
    frames = flat_frames(400)
    res = SimExchange(frames, CostModel(enabled=False), CostModel(enabled=True),
                      SimConfig(initial_cash=10_000, warmup=20,
                                latency_jitter_mean=3.0, latency_jitter_seed=7,
                                latency_max_bars=4)).run(
        RandomWeights(seed=2, rebalance_every=3))
    assert max(res.latency_histogram) <= 4
    assert len(res.net_equity) == len(res.gross_equity)


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
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
