"""数据质量与缺口机制的测试。

缺口是会污染成交时点假设的那一项（实测三标的各 27 处、最长 34 小时、
位置完全一致＝交易所停机），所以必须有测试兜住。

运行：python3 tests/test_data_quality.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.agents.baselines import RandomWeights, SingleAssetBuyHold  # noqa: E402
from src.data.quality import (gap_flags, gap_report,  # noqa: E402
                              trading_time_weights, union_gap_flags)
from src.env.market_view import LookaheadError, MarketView, Precomputed  # noqa: E402
from src.sim.costs import CostModel  # noqa: E402
from src.sim.exchange import SimConfig, SimExchange  # noqa: E402


def gapped_index(n_pre=50, n_post=50, gap_hours=5, start="2020-01-01"):
    """构造一个中间有缺口的 1 小时时间轴。"""
    pre = pd.date_range(start, periods=n_pre, freq="h", tz="UTC")
    post = pd.date_range(pre[-1] + pd.Timedelta(hours=1 + gap_hours),
                         periods=n_post, freq="h", tz="UTC")
    return pre.append(post)


def make_frames(index, symbols=("AAA", "BBB"), seed=0):
    rng = np.random.default_rng(seed)
    out = {}
    for s in symbols:
        close = 100 * np.exp(np.cumsum(rng.normal(0.0, 0.01, len(index))))
        open_ = np.concatenate([[close[0]], close[:-1]])
        out[s] = pd.DataFrame({
            "open": open_, "high": np.maximum(open_, close) * 1.001,
            "low": np.minimum(open_, close) * 0.999, "close": close,
            "volume": 100.0, "quote_volume": close * 100.0,
        }, index=index)
    return out


# ==========================================================================
# 1. 缺口识别
# ==========================================================================
def test_gap_flags_marks_exactly_the_post_gap_bar():
    idx = gapped_index(50, 50, 5)
    f = gap_flags(idx)
    assert len(f) == len(idx) == 100
    assert f[0] == False, "第一根没有前一根可比，不应标记"
    assert f.sum() == 1, f"应恰好标记 1 根，实际 {f.sum()}"
    # 缺口后第一根的位置是 index 50
    assert f[50] == True, f"缺口后的第一根（第 50 位）应被标记，实际标记在 {np.where(f)[0]}"
    assert not f[49] and not f[51]


def test_gap_flags_on_continuous_index_is_all_false():
    idx = pd.date_range("2021-01-01", periods=200, freq="h", tz="UTC")
    assert gap_flags(idx).sum() == 0


def test_gap_report_counts_missing_bars():
    idx = gapped_index(50, 50, 5)
    r = gap_report(idx)
    assert r["n_gaps"] == 1
    assert r["missing_bars"] == 5, f"应缺 5 根，实际 {r['missing_bars']}"
    assert r["bars"] == 100


def test_union_gap_flags_across_symbols():
    idx = gapped_index(50, 50, 3)
    frames = make_frames(idx)
    u = union_gap_flags(frames)
    assert u.sum() == 1 and u[50]


def test_trading_time_weights_expands_at_gap():
    idx = gapped_index(50, 50, 5)
    w = trading_time_weights(idx)
    assert len(w) == len(idx)
    assert w[0] == 1.0
    assert w[50] == 6.0, f"跨 5 根缺口的 bar 应代表 6 个单位时间，实际 {w[50]}"
    normal = np.delete(w, 0)
    assert np.allclose(np.delete(normal, 49), 1.0)


# ==========================================================================
# 2. MarketView 的缺口查询不能破坏因果封印
# ==========================================================================
def test_market_view_is_gap_and_causality():
    idx = gapped_index(50, 50, 5)
    frames = make_frames(idx)
    pre = Precomputed(frames, {20}, {20})
    v = MarketView(frames, 60, pre)
    assert v.is_gap(60) == False
    assert v.is_gap(50) == True, "回看已发生的缺口应当允许"
    try:
        v.is_gap(61)
    except LookaheadError:
        pass
    else:
        raise AssertionError("is_gap 越权访问未被拒绝")


def test_market_view_without_precompute_reports_no_gap():
    idx = gapped_index(50, 50, 5)
    frames = make_frames(idx)
    v = MarketView(frames, 60, None)
    assert v.is_gap() == False, "没有预计算信息时不应臆造缺口"


# ==========================================================================
# 3. 仿真器的缺口策略
# ==========================================================================
def test_gap_policy_validation():
    try:
        SimConfig(gap_policy="whatever")
    except ValueError:
        return
    raise AssertionError("非法 gap_policy 未被拒绝")


def test_skip_policy_actually_skips():
    idx = gapped_index(80, 200, 6)
    frames = make_frames(idx, seed=3)
    sim = dict(initial_cash=10_000.0, warmup=20, latency_bars=1)
    ag_e = RandomWeights(seed=1, rebalance_every=1)
    ag_s = RandomWeights(seed=1, rebalance_every=1)
    cm = CostModel(enabled=True)
    r_e = SimExchange(frames, CostModel(enabled=False), cm,
                      SimConfig(gap_policy="execute", **sim)).run(ag_e)
    r_s = SimExchange(frames, CostModel(enabled=False), cm,
                      SimConfig(gap_policy="skip", **sim)).run(ag_s)
    assert r_e.n_gap_bars >= 1, "应识别出缺口 bar"
    assert r_e.n_gap_trades >= 1, "execute 口径下应有跨缺口成交"
    assert r_s.n_gap_skipped >= 1, "skip 口径下应有被跳过的成交"
    assert r_s.n_trades < r_e.n_trades, "skip 应比 execute 少成交"
    assert len(r_s.net_equity) == len(r_e.net_equity), "两种口径的 bar 数必须一致"
    assert abs(r_s.final_net() - r_e.final_net()) > 0, "两种口径的结果应当不同"


def test_gap_free_data_has_identical_results_under_both_policies():
    """连续无缺口时，两种口径必须给出完全相同的结果。"""
    idx = pd.date_range("2022-01-01", periods=300, freq="h", tz="UTC")
    frames = make_frames(idx, seed=5)
    sim = dict(initial_cash=10_000.0, warmup=20, latency_bars=1)
    cm = CostModel(enabled=True)
    outs = []
    for pol in ("execute", "skip"):
        res = SimExchange(frames, CostModel(enabled=False), cm,
                          SimConfig(gap_policy=pol, **sim)).run(
            RandomWeights(seed=2, rebalance_every=3))
        outs.append(res)
    assert outs[0].n_gap_bars == 0 and outs[1].n_gap_skipped == 0
    assert np.allclose(outs[0].net_equity, outs[1].net_equity), \
        "无缺口时两种口径不该有任何差异"


def test_buyhold_never_trades_at_gap():
    idx = gapped_index(80, 300, 6)
    frames = make_frames(idx, seed=7)
    res = SimExchange(frames, CostModel(enabled=False), CostModel(enabled=True),
                      SimConfig(initial_cash=10_000.0, warmup=20)).run(
        SingleAssetBuyHold("AAA"))
    # 买入持有只在第一根决策，那根不在缺口后（warmup=20，缺口在第 80 根后）
    assert res.n_gap_trades == 0


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
