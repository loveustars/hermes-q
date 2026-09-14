"""M3 因果封印与撮合正确性测试 —— 未来函数注入测试。

设计原则：**假设策略会作弊**。这里主动构造"想偷看未来"的策略，
系统必须拒绝，而不是靠写策略时小心。

运行方式：
    python3 tests/test_causality.py          # 直接运行
    pytest tests/test_causality.py           # 若装了 pytest
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.agents.baselines import Agent, SingleAssetBuyHold  # noqa: E402
from src.env.market_view import LookaheadError, MarketView  # noqa: E402
from src.eval import metrics  # noqa: E402
from src.sim.costs import CostModel  # noqa: E402
from src.sim.exchange import SimConfig, SimExchange  # noqa: E402


# --------------------------------------------------------------------------
# 测试夹具：构造一段确定性价格序列
# --------------------------------------------------------------------------
def make_frames(n: int = 400, seed: int = 7) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-01", periods=n, freq="h", tz="UTC")
    out = {}
    for sym, drift in [("AAA", 0.0002), ("BBB", -0.0001)]:
        r = rng.normal(drift, 0.01, n)
        close = 100 * np.exp(np.cumsum(r))
        open_ = np.concatenate([[close[0]], close[:-1]])
        out[sym] = pd.DataFrame({
            "open": open_,
            "high": np.maximum(open_, close) * 1.002,
            "low": np.minimum(open_, close) * 0.998,
            "close": close,
            "volume": np.full(n, 100.0),
            "quote_volume": close * 100.0,
        }, index=idx)
    return out


# --------------------------------------------------------------------------
# 1. MarketView 越权访问必须抛异常
# --------------------------------------------------------------------------
def test_view_rejects_future_index():
    frames = make_frames(50)
    v = MarketView(frames, 10)
    assert v.close("AAA") == frames["AAA"]["close"].iloc[10]
    assert v.close("AAA", 10) == frames["AAA"]["close"].iloc[10]

    for bad in (11, 25, 49):
        try:
            v.close("AAA", bad)
        except LookaheadError:
            pass
        else:
            raise AssertionError(f"访问第 {bad} 根 bar 未被拒绝（当前 t=10）")


def test_view_window_never_includes_future():
    frames = make_frames(50)
    t = 20
    v = MarketView(frames, t)
    w = v.window("AAA", "close", 10)
    assert len(w) == 10
    expected = frames["AAA"]["close"].iloc[11:21].to_numpy()
    assert np.allclose(w, expected), "window 必须恰好是 t-9..t，不能多一根"


def test_view_returns_length_matches_window():
    frames = make_frames(50)
    v = MarketView(frames, 20)
    # 语义：returns(sym, n) = n 个收益率（需要 n+1 个价格）
    assert len(v.returns("AAA", 10)) == 10
    assert len(v.returns("AAA", 1)) == 1
    # 且不能包含未来
    px = v.window("AAA", "close", 11)          # t-10..t
    assert np.allclose(v.returns("AAA", 10), np.diff(np.log(px)))


# --------------------------------------------------------------------------
# 2. 想偷看未来的策略，必须在仿真里直接崩掉
# --------------------------------------------------------------------------
class CheatingAgent(Agent):
    """故意偷看下一根 bar 的收盘价，用来验证系统会拒绝。"""

    name = "cheater"

    def decide(self, view: MarketView):
        peek = view.close("AAA", view.step + 1)      # 越权
        return {"AAA": 1.0 if peek > view.close("AAA") else 0.0}


def test_cheating_agent_is_rejected():
    frames = make_frames(200)
    sim = SimExchange(frames, CostModel(enabled=False), CostModel(enabled=False),
                      SimConfig(initial_cash=10_000, warmup=50))
    try:
        sim.run(CheatingAgent())
    except LookaheadError:
        return
    raise AssertionError("偷看未来的策略居然跑完了 —— 因果封印失效")


# --------------------------------------------------------------------------
# 3. 成交价必须是下一根 bar 的开盘价，不是当根收盘价
# --------------------------------------------------------------------------
class RecordExecAgent(Agent):
    """只买一次，记录决策时看到的收盘价，之后维持。"""

    name = "record_exec"

    def __init__(self):
        self.seen: list[tuple[int, float]] = []
        self._done = False

    def decide(self, view: MarketView):
        if self._done:
            return None
        self._done = True
        self.seen.append((view.step, view.close("AAA")))
        return {"AAA": 1.0, "BBB": 0.0}


def test_execution_price_is_next_bar_open():
    frames = make_frames(200)
    ag = RecordExecAgent()
    CostModel()
    sim = SimExchange(frames, CostModel(enabled=False), CostModel(enabled=False),
                      SimConfig(initial_cash=10_000, warmup=50, latency_bars=1))
    res = sim.run(ag)

    t_decision, close_at_decision = ag.seen[0]
    ex = t_decision + 1
    open_at_exec = float(frames["AAA"]["open"].iloc[ex])

    # 权益 = 现金 + 持仓，建仓后第一根 bar 的权益应等于本金 × (close_ex / open_ex)
    ratio = res.gross_equity[0] / 10_000
    expected = float(frames["AAA"]["close"].iloc[ex]) / open_at_exec
    assert abs(ratio - expected) < 1e-6, (
        f"成交价不是下一根开盘价：实际比例 {ratio:.8f}，"
        f"按 next-bar-open 应为 {expected:.8f}，"
        f"按当根收盘应为 1.0")


# --------------------------------------------------------------------------
# 4. 成本模型的边界行为
# --------------------------------------------------------------------------
def test_zero_cost_model_makes_gross_equal_net():
    frames = make_frames(400)
    zero = CostModel(enabled=False)
    sim = SimExchange(frames, zero, CostModel(enabled=False),
                      SimConfig(initial_cash=10_000, warmup=100))
    from src.agents.baselines import RandomWeights
    res = sim.run(RandomWeights(seed=1, rebalance_every=12))
    assert np.allclose(res.net_equity, res.gross_equity), "零成本模型下毛净必须完全一致"
    assert res.cost_paid.sum() == 0.0


def test_positive_costs_make_net_worse():
    frames = make_frames(400)
    sim = SimExchange(frames, CostModel(enabled=False), CostModel(enabled=True),
                      SimConfig(initial_cash=10_000, warmup=100))
    from src.agents.baselines import RandomWeights
    res = sim.run(RandomWeights(seed=1, rebalance_every=6))
    assert res.net_equity[-1] < res.gross_equity[-1], "有成本时净终值必须低于毛终值"
    assert res.cost_paid.sum() > 0


def test_buyhold_has_almost_no_turnover():
    frames = make_frames(400)
    sim = SimExchange(frames, CostModel(enabled=False), CostModel(enabled=True),
                      SimConfig(initial_cash=10_000, warmup=100))
    res = sim.run(SingleAssetBuyHold("AAA"))
    assert res.n_trades <= 2, f"买入持有不该频繁交易，实际 {res.n_trades} 笔"


# --------------------------------------------------------------------------
# 5. 指标口径：收益必须以固定本金为基数
# --------------------------------------------------------------------------
def test_metrics_use_fixed_initial():
    eq = np.array([9_000.0, 10_800.0, 12_000.0])
    s = metrics.summary(eq, None, None, 24 * 365, initial=10_000.0)
    assert abs(s["total_return"] - 0.20) < 1e-6, "应以固定本金 10,000 为基数"
    assert s["final_equity"] == 12_000.0
    s2 = metrics.summary(eq, None, None, 24 * 365)   # 不传 initial 时退化为 e[0]
    assert abs(s2["total_return"] - (12_000 / 9_000 - 1)) < 1e-6


# --------------------------------------------------------------------------
# 6. 数据对齐
# --------------------------------------------------------------------------
def test_misaligned_frames_rejected():
    frames = make_frames(200)
    bad = dict(frames)
    bad["CCC"] = frames["AAA"].iloc[:150]
    try:
        SimExchange(bad, CostModel(enabled=False), CostModel(enabled=False),
                    SimConfig(warmup=10))
    except ValueError:
        return
    raise AssertionError("时间轴不一致的标的居然被接受了")


# --------------------------------------------------------------------------
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
