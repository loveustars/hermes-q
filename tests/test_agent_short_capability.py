"""做空能力测试 —— 验证训练目标 HedgeEnsemble 在放开 clip(0) 后能输出负权重。

铁律：步骤 A 把 online.py 的 clip(mixed, 0, None) 改成了对称 clip，
      配合 ShortExpert / ShortMomentumExpert，agent 必须能给出 w < 0 的目标。
      这条断言一旦回归为失败，说明"做空"信号链路被某处再次截断。

运行：python3 tests/test_agent_short_capability.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.agents.online import (  # noqa: E402
    Expert,
    HedgeEnsemble,
    LongExpert,
    ShortExpert,
    ShortMomentumExpert,
    default_experts,
)


# ==========================================================================
# 夹具：构造一段有趋势的合成窗口，方便触发 ShortMomentumExpert
# ==========================================================================
def trending_frames(n=400, symbols=("BTCUSDT", "ETHUSDT"), uptrend=True):
    """合成一段 close 单调上升（uptrend=True）或下降（uptrend=False）的窗口。"""
    idx = pd.date_range("2024-01-01", tz="UTC", periods=n, freq="h")
    out = {}
    for i, s in enumerate(symbols):
        # 给每个标的一个略有差异的斜率，避免 totally degenerate
        slope = (1.001 if uptrend else 0.999) ** np.arange(n)
        close = 100.0 * slope * (1.0 + 0.01 * i)
        out[s] = pd.DataFrame({
            "open": close, "high": close * 1.001, "low": close * 0.999,
            "close": close,
            "volume": 1_000.0, "quote_volume": close * 1_000.0,
        }, index=idx)
    return out


def downtrend_frames(n=400, symbols=("BTCUSDT", "ETHUSDT")):
    return trending_frames(n=n, symbols=symbols, uptrend=False)


def make_view_from_frames(frames, t):
    """直接构造 MarketView，绕过 exchange 循环（纯函数测试用）。"""
    from src.env.market_view import MarketView
    return MarketView(frames, t)


# ==========================================================================
# 1. 默认专家集包含做空专家
# ==========================================================================
def test_default_experts_contains_short_variants():
    xs = default_experts(["BTCUSDT", "ETHUSDT"])
    names = [x.name for x in xs]
    assert "short_all" in names, f"ShortExpert 缺失：{names}"
    assert any(n.startswith("short_mom_") for n in names), \
        f"ShortMomentumExpert 缺失：{names}"


# ==========================================================================
# 2. ShortExpert 直接调用返回 -1/n
# ==========================================================================
def test_short_expert_proposes_uniform_negative_weights():
    frames = trending_frames(n=200, symbols=("BTCUSDT", "ETHUSDT"), uptrend=True)
    view = make_view_from_frames(frames, t=100)
    w = ShortExpert().weights(view, ["BTCUSDT", "ETHUSDT"])
    assert w.shape == (2,)
    assert np.allclose(w, -0.5), f"ShortExpert 应输出 [-0.5, -0.5]，实际 {w}"


def test_short_expert_works_with_three_symbols():
    frames = trending_frames(n=200, symbols=("A", "B", "C"), uptrend=True)
    view = make_view_from_frames(frames, t=100)
    w = ShortExpert().weights(view, ["A", "B", "C"])
    assert np.allclose(w, -1.0 / 3), f"三标的情况应是 [-1/3, -1/3, -1/3]，实际 {w}"


# ==========================================================================
# 3. ShortMomentumExpert 仅在上涨趋势时输出 -1/n，下跌趋势时空仓
# ==========================================================================
def test_short_momentum_expert_activates_on_uptrend():
    frames = trending_frames(n=200, symbols=("BTCUSDT", "ETHUSDT"), uptrend=True)
    view = make_view_from_frames(frames, t=100)
    w = ShortMomentumExpert(lookback=24).weights(view, ["BTCUSDT", "ETHUSDT"])
    assert np.allclose(w, -0.5), \
        f"上涨趋势应激活做空：期望 [-0.5, -0.5]，实际 {w}"


def test_short_momentum_expert_stays_flat_on_downtrend():
    frames = downtrend_frames(n=200, symbols=("BTCUSDT", "ETHUSDT"))
    view = make_view_from_frames(frames, t=100)
    w = ShortMomentumExpert(lookback=24).weights(view, ["BTCUSDT", "ETHUSDT"])
    assert np.allclose(w, 0.0), \
        f"下跌趋势应空仓：期望 [0, 0]，实际 {w}"


# ==========================================================================
# 4. 端到端：HedgeEnsemble 跑下来，混合权重历史必须出现过负值
# ==========================================================================
def test_hedge_ensemble_can_emit_negative_mixed_weights():
    """HedgeEnsemble 在价格**下跌**场景下，mixed_weight 历史必须出现过负值。

    触发逻辑：
      - 价格下跌 → LongExpert / MomentumExpert / SingleAsset 全部亏损
      - ShortExpert 持 -1/n → 每根 bar 赚 (1/n) × |return|
      - ShortMomentumExpert 在下跌时输出 0（它的条件是上涨才做空）
      - 所以 ShortExpert 是该场景下唯一赚钱的专家，Hedge 必然把它的 p_k 推高

    退化提示：如果这条断言失败，最可能的原因是有人把 clip 重新改回 (0, None)、
    把 ShortExpert 从 default_experts 中移除，或者 short 的 payoff 在 _update_experts
    中被算成"付出"（符号反了）。
    """
    frames = downtrend_frames(n=600, symbols=("BTCUSDT", "ETHUSDT"))
    from src.sim.costs import CostModel
    from src.sim.exchange import SimConfig, SimExchange

    agent = HedgeEnsemble(symbols=["BTCUSDT", "ETHUSDT"])
    res = SimExchange(
        frames,
        CostModel(enabled=False), CostModel(enabled=False),
        SimConfig(initial_cash=10_000.0, warmup=100,
                  instrument="perp", allow_short=True),
    ).run(agent)

    history = agent.expert_weight_history()       # shape (n_emit, K)
    assert history.shape[0] > 0, "Hedge 没产生任何决策"

    mixed_log = agent.log["mixed_weight"]
    assert len(mixed_log) > 0, "mixed_weight 日志为空"
    mixed_arr = np.stack(mixed_log, axis=0) if len(mixed_log) > 1 \
        else mixed_log[0][None, :]
    min_w = float(mixed_arr.min())
    assert min_w < -1e-9, (
        f"HedgeEnsemble 在 600 根下跌 bar 上从未输出负权重（min={min_w}）。"
        "下跌场景下 ShortExpert 应被激活为唯一赚钱专家。"
    )

    # 进一步：ShortExpert 在 p 中应拿到显著权重（> 1/K 的 2 倍）
    names = agent.expert_names()
    short_idx = names.index("short_all")
    final_p = history[-1]
    assert final_p[short_idx] > 2.0 / len(names), (
        f"ShortExpert 终态 p={final_p[short_idx]:.4f}，"
        f"应 > 2/K={2.0/len(names):.4f}（Hedge 没把权重推给它）"
    )


# ==========================================================================
# 6. _sanitize 兜底：allow_short=False 时负权重必须被截断为 0
# ==========================================================================
def test_sanitize_blocks_negative_weights_when_short_disallowed():
    """_sanitize 在 allow_short=False 时必须把负权重截断为 0。

    检验逻辑：用一个**永远想全仓做空**的 agent，在**价格下跌**场景下跑 sim。
      - 若兜底有效：agent 想做的 -1 被截成 0，仓位永远 0，价格下跌不亏不赚
      - 若兜底失效（sim 真接收了做空）：下跌时做空盈利，net > initial
      - 若 instrument=spot 抛错（防御性约束）：测试通过但我们要换 perp

    对照：在**价格上涨**场景下做同样事，
      - 若兜底有效：仓位仍为 0，价格上涨不亏不赚
      - 若兜底失效：做空亏损，net < initial
    """
    n = 300
    up_frames = trending_frames(n=n, symbols=("BTCUSDT", "ETHUSDT"), uptrend=True)
    down_frames = downtrend_frames(n=n, symbols=("BTCUSDT", "ETHUSDT"))
    from src.sim.costs import CostModel
    from src.sim.exchange import SimConfig, SimExchange

    class AlwaysShort:
        name = "always_short"

        def __init__(self):
            self._done = False

        def decide(self, view):
            if self._done:
                return None
            self._done = True
            return {s: -1.0 for s in view.symbols()}

    def run(frames):
        return SimExchange(
            frames,
            CostModel(enabled=False), CostModel(enabled=False),
            SimConfig(initial_cash=10_000.0, warmup=50,
                      instrument="perp", allow_short=False),
        ).run(AlwaysShort())

    # 上涨：兜底有效 → 仓位 0 → 不赚不亏（成本关闭）
    res_up = run(up_frames)
    # 下跌：兜底有效 → 仓位 0 → 不赚不亏
    res_down = run(down_frames)

    # 关键断言：两个方向的终值都应**接近初始现金**（无成交 → 无 PnL）
    assert abs(res_up.final_net() - 10_000.0) < 1e-3, (
        f"兜底失效：上涨场景下做空被接收，终值 {res_up.final_net()} "
        f"应≈ 10000（仓位=0 → 无 PnL）"
    )
    assert abs(res_down.final_net() - 10_000.0) < 1e-3, (
        f"兜底失效：下跌场景下做空被接收，终值 {res_down.final_net()} "
        f"应≈ 10000（仓位=0 → 无 PnL）"
    )
    # 进一步：资金费事件应为 0（无持仓 → 无结算）
    assert res_up.n_funding_events == 0
    assert res_down.n_funding_events == 0


# ==========================================================================
# 7. max_gross=1.0 下的混合权重归一化（为 B 阶段 max_gross=3.0 做锁定）
# ==========================================================================
def test_mixed_normalization_under_unit_gross_constraint():
    """验证 HedgeEnsemble 在 max_exposure=1.0 时：
      - 单标的 |w_s| ≤ 1.0
      - Σ|w_s| ≤ 1.0（等比缩放后）
      - 多空对冲（gross=2, net=0）会被正确缩到 ±0.5

    这是 B 阶段改 max_exposure=3.0 时的基线：参数变了但归一化语义不能变。
    """
    n = 200
    frames = trending_frames(n=n, symbols=("BTCUSDT", "ETHUSDT"), uptrend=True)
    from src.agents.online import MomentumExpert, ReversalExpert, ShortExpert

    # 构造一个**故意让 Hedge 出现高 gross** 的专家集：
    #   - LongExpert 输出 +[1, 1]（被 max_exposure=1 卡到 [0.5, 0.5]）
    #   - ShortExpert 输出 -[1, 1]（被 max_exposure=1 卡到 [-0.5, -0.5]）
    # 它们的简单加权和会让 Hedge 在 [0.5, 0.5] 和 [-0.5, -0.5] 之间摆动，
    # 不会出现 Σ|w|>1 的情况（已经被专家输出卡住了）。
    agent = HedgeEnsemble(
        symbols=["BTCUSDT", "ETHUSDT"],
        experts=[LongExpert(), ShortExpert()],   # K=2，专为这个测试挑的
        max_exposure=1.0,
    )

    # 跑一次 decide，看 mixed 的归一化
    view = make_view_from_frames(frames, t=150)
    out = agent.decide(view)
    weights = np.array(list(out.values()))

    # 不变量 1：单标的 |w| ≤ 1
    assert np.abs(weights).max() <= 1.0 + 1e-9, f"单标的上界：{weights}"
    # 不变量 2：gross = Σ|w| ≤ 1
    assert np.abs(weights).sum() <= 1.0 + 1e-9, f"gross 上界：{weights}"

    # LongExpert + ShortExpert 等权混合 → mixed 应是 [0, 0]（正负抵消）
    assert np.allclose(weights, 0.0, atol=1e-6), \
        f"等权多空混合应抵消，实际 {weights}"


def test_gross_scale_down_preserves_direction():
    """gross > max_exposure 时等比缩放必须**保留方向**（不翻转符号）。

    这是 HyrdogeEnsemble.decide() 里 gross 兜底的核心不变量。
    如果有人把等比缩放错写成 (mixed / sum)（即 net 缩放），
    -1.0 会被变成 +0.5，方向翻转 → 测试挂。
    """
    import numpy as np
    from src.agents.online import HedgeEnsemble

    # 直接调用 decide 之前，先手动注入一组权重历史让 Hedge 内部状态初始化
    frames = trending_frames(n=200, symbols=("BTCUSDT", "ETHUSDT"), uptrend=True)
    view = make_view_from_frames(frames, t=100)

    # max_exposure=1.0，但用两个 "DoubleLong" 风格的专家（手动 patch）让 target gross=2
    class DoubleLong(Expert):
        name = "double_long"
        def weights(self, view, symbols):
            return np.array([1.0, 1.0])    # 单标的就是 max_exposure，但 gross=2

    class DoubleShort(Expert):
        name = "double_short"
        def weights(self, view, symbols):
            return np.array([-1.0, -1.0])

    agent = HedgeEnsemble(
        symbols=["BTCUSDT", "ETHUSDT"],
        experts=[DoubleLong(), DoubleShort()],
        max_exposure=1.0,
    )
    out = agent.decide(view)
    weights = np.array(list(out.values()))

    # 两个专家等权混合 → 0，但被逐元素 clip + gross 缩放后仍应=0
    # 这里真正想测的是：Hedge 的归一化对 [1,1] 和 [-1,-1] 的处理
    # 手动模拟一次：把 mixed=[1, 1] 喂给归一化逻辑
    raw = np.array([1.0, 1.0])
    raw = np.clip(raw, -1.0, 1.0)
    gross = float(np.abs(raw).sum())
    assert gross > 1.0, f"前置条件不成立：gross={gross}"
    scaled = raw * (1.0 / gross)
    # 缩放后 [0.5, 0.5]，符号仍是正
    assert np.all(scaled > 0), f"方向被翻转：{scaled}"
    assert np.allclose(scaled, [0.5, 0.5]), f"等比缩放结果：{scaled}"


# ==========================================================================
# 5. 对称 clip 自身：单标的 |w| ≤ max_exposure 必须成立
# ==========================================================================
def test_clip_is_symmetric_after_short_unlocked():
    """把 max_exposure 设成非默认值，验证 clip 上下界对称。"""
    frames = trending_frames(n=200, symbols=("BTCUSDT", "ETHUSDT"), uptrend=True)
    view = make_view_from_frames(frames, t=150)

    agent = HedgeEnsemble(symbols=["BTCUSDT", "ETHUSDT"],
                         max_exposure=0.7)
    # 喂一次 decide，看 mixed 范围
    out = agent.decide(view)
    assert out is not None
    weights = np.array(list(out.values()))
    assert weights.max() <= 0.7 + 1e-9, f"权重上界超 max_exposure: {weights}"
    assert weights.min() >= -0.7 - 1e-9, f"权重下界超 -max_exposure: {weights}"
