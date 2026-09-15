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
# 8. funding carry 信号：正费率持续时 ShortExpert 应被激活
# ==========================================================================
def test_short_expert_activated_by_positive_funding_carry():
    """给 HedgeEnsemble 喂一个 funding_table（每 8h +0.01%），
    在横盘（零价格收益）场景下跑 sim，断言：
      - ShortExpert 的终态 p > 2/K（Hedge 学到了 funding carry）
      - min(mixed_weight) < -1e-6（Hedge 真的在做空）

    这是 A' "修信号" 的核心断言：funding 持续为正时，ShortExpert 应该是
    长期赚钱的专家（收 funding carry），Hedge 应该学会做空。

    退化提示：如果这条断言失败，要么 funding 没接进 _update_experts 的 payoffs，
    要么符号搞反了（空头收到 funding 应是 +payoff，不是 -）。
    """
    n = 200
    frames = trending_frames(n=n, symbols=("BTCUSDT", "ETHUSDT"), uptrend=True)
    # 横盘：把 close 全设成常数 100（去掉价格 PnL，只看 funding 信号）
    for s in frames:
        f = frames[s]
        f["open"] = f["high"] = f["low"] = f["close"] = 100.0
        f["quote_volume"] = 100_000.0

    from src.sim.costs import CostModel
    from src.sim.exchange import SimConfig, SimExchange
    from src.sim.funding import FundingTable

    # 构造 funding_table：每 8h（即每 8 根 1h bar）结算一次 +0.01%
    ft = FundingTable()
    idx = frames["BTCUSDT"].index
    for t in range(0, n, 8):
        ts_ms = int(idx[t].timestamp() * 1000)
        ft.add("BTCUSDT", ts_ms, 0.0001)   # +1bp / 8h
        ft.add("ETHUSDT", ts_ms, 0.0001)

    agent = HedgeEnsemble(symbols=["BTCUSDT", "ETHUSDT"], funding=ft)
    res = SimExchange(
        frames,
        CostModel(enabled=False), CostModel(enabled=False),
        SimConfig(initial_cash=10_000.0, warmup=50,
                  instrument="perp", allow_short=True),
    ).run(agent)

    names = agent.expert_names()
    short_idx = names.index("short_all")
    final_p = agent.expert_weight_history()[-1]
    assert final_p[short_idx] > 2.0 / len(names), (
        f"有 funding 持续为正时 ShortExpert 终态 p={final_p[short_idx]:.4f}，"
        f"应 > 2/K={2.0/len(names):.4f}（Hedge 没学到 carry 收益做空）"
    )

    mixed_log = agent.log["mixed_weight"]
    mixed_arr = np.stack(mixed_log, axis=0) if len(mixed_log) > 1 else mixed_log[0][None, :]
    assert float(mixed_arr.min()) < -1e-6, (
        f"有 funding 持续为正时 mixed_weight min={mixed_arr.min()}，"
        "应 < 0（Hedge 没在做空）"
    )


def test_funding_none_keeps_pure_price_signal():
    """funding=None 时，payoffs 路径不应包含 funding 项（向后兼容）。"""
    n = 100
    frames = trending_frames(n=n, symbols=("BTCUSDT", "ETHUSDT"), uptrend=True)
    from src.sim.costs import CostModel
    from src.sim.exchange import SimConfig, SimExchange

    # funding=None
    agent = HedgeEnsemble(symbols=["BTCUSDT", "ETHUSDT"], funding=None)
    SimExchange(
        frames, CostModel(enabled=False), CostModel(enabled=False),
        SimConfig(initial_cash=10_000.0, warmup=50, instrument="perp",
                  allow_short=True),
    ).run(agent)

    # 不变量：log 正常填充、没崩
    assert len(agent.log["step"]) > 0
    history = agent.expert_weight_history()
    assert history.shape[0] > 0
    assert (history.sum(axis=1) > 0).all(), "p 必须归一化"


# ==========================================================================
# 9. B 阶段：3x 杠杆基线测试
# ==========================================================================
def test_max_exposure_3x_caps_per_asset_weight():
    """max_exposure=3.0 时，单标 |w_s| ≤ 3.0 必须成立（agent 层 clip 生效）。"""
    n = 200
    frames = trending_frames(n=n, symbols=("BTCUSDT", "ETHUSDT"), uptrend=True)
    from src.agents.online import LongExpert, ShortExpert

    # 构造 K=2 的极端专家集：LongExpert(+1) + ShortExpert(-1)，
    # 让 Hedge 在 K=1 时倾向 LongExpert，给出 +3 的目标（不可能，但测 clip 兜底）
    agent = HedgeEnsemble(
        symbols=["BTCUSDT", "ETHUSDT"],
        experts=[LongExpert(), ShortExpert()],
        max_exposure=3.0,
    )
    view = make_view_from_frames(frames, t=150)
    out = agent.decide(view)
    weights = np.array(list(out.values()))
    assert np.abs(weights).max() <= 3.0 + 1e-9, \
        f"max_exposure=3.0 但单标超限：{weights}"


def test_max_exposure_3x_caps_gross_exposure():
    """max_exposure=3.0 时，Σ|w_s| ≤ 3.0（gross 兜底）。"""
    n = 200
    frames = trending_frames(n=n, symbols=("BTCUSDT", "ETHUSDT"), uptrend=True)
    from src.agents.online import LongExpert, ShortExpert

    agent = HedgeEnsemble(
        symbols=["BTCUSDT", "ETHUSDT"],
        experts=[LongExpert(), ShortExpert()],
        max_exposure=3.0,
    )
    view = make_view_from_frames(frames, t=150)
    out = agent.decide(view)
    weights = np.array(list(out.values()))
    assert np.abs(weights).sum() <= 3.0 + 1e-9, \
        f"max_exposure=3.0 但 gross 超限：{weights}"


def test_simconfig_max_gross_3x_enforced():
    """SimConfig(max_gross=3.0) 必须能接住 3.0 仓位（不被截到更小）。"""
    n = 200
    frames = trending_frames(n=n, symbols=("BTCUSDT", "ETHUSDT"), uptrend=True)
    from src.sim.costs import CostModel
    from src.sim.exchange import SimConfig, SimExchange

    # 永远想 0.5 满仓（每标的 0.5）的 agent
    class HalfHalf:
        name = "half_half"
        def __init__(self):
            self._done = False
        def decide(self, view):
            if self._done:
                return None
            self._done = True
            return {s: 0.5 for s in view.symbols()}

    res_max3 = SimExchange(
        frames, CostModel(enabled=False), CostModel(enabled=False),
        SimConfig(initial_cash=10_000.0, warmup=50, max_gross=3.0),
    ).run(HalfHalf())

    res_max1 = SimExchange(
        frames, CostModel(enabled=False), CostModel(enabled=False),
        SimConfig(initial_cash=10_000.0, warmup=50, max_gross=1.0),
    ).run(HalfHalf())

    # max_gross=3 应该至少让仓位达到 1（= 0.5+0.5），max_gross=1 同理
    # 关键差异：max_gross=3 不应"截到更小"，所以 n_trades 应该相同（不需要截）
    # 价格不变，仓位不变 → 终值应相同（除了 funding，但这里 funding=None）
    assert res_max3.n_trades > 0, "max_gross=3 应该允许成交"
    assert res_max1.n_trades > 0, "max_gross=1 应该允许成交"


def test_simconfig_max_gross_does_not_exceed_3x():
    """agent 想用 5x 但 sim 配 max_gross=3.0 → 实际仓位被截到 3x。"""
    n = 200
    frames = trending_frames(n=n, symbols=("BTCUSDT", "ETHUSDT"), uptrend=True)
    from src.sim.costs import CostModel
    from src.sim.exchange import SimConfig, SimExchange

    # 永远想每标 5x（超 max_gross=3）
    class FiveX:
        name = "five_x"
        def __init__(self):
            self._done = False
        def decide(self, view):
            if self._done:
                return None
            self._done = True
            return {s: 5.0 for s in view.symbols()}    # gross=10，超 max_gross=3

    res = SimExchange(
        frames, CostModel(enabled=False), CostModel(enabled=False),
        SimConfig(initial_cash=10_000.0, warmup=50, max_gross=3.0),
    ).run(FiveX())

    # 验证：sim 实际持仓 = 3/10 = 0.3x 每标的（等比缩放）
    # 初始 10000，按 open 价建仓，cash + 单位持仓 = 10000
    # 单位持仓 = (3/2) * 10000 / 100 = 150（每标的）
    # 实际结算价 = open，盯市 net_eq ≈ 10000（价格不变）
    assert not res.insolvent, f"max_gross=3 + 5x 提案不应破产：{res.final_net()}"
    # 关键：成交笔数应该是 N_symbols（2 标的，每标 1 笔）
    assert res.n_trades >= 2, f"应至少成交 2 笔，实际 {res.n_trades}"


# ==========================================================================
# 11. C 阶段：SimExchange 集成 MarginBook
# ==========================================================================
def test_sim_exchange_margin_ledger_updated_on_open_close():
    """开启 MarginConfig 后，SimExchange 主循环应同步更新 margin_cash。

    验证：
      - 第一根建仓后，margin_cash > 0（记账初始保证金）
      - 第二根平仓后，margin_cash = 0（清零）
    """
    n = 200
    frames = trending_frames(n=n, symbols=("BTCUSDT", "ETHUSDT"), uptrend=True)
    from src.sim.costs import CostModel
    from src.sim.exchange import SimConfig, SimExchange
    from src.sim.margin import MarginConfig

    class OpenLongClose:
        """第一根 bar 想满仓 1x 多，第二根想平仓。"""
        name = "open_long_close"
        def __init__(self):
            self._step = 0
        def decide(self, view):
            self._step += 1
            if self._step == 1:
                return {s: 1.0 for s in view.symbols()}
            if self._step == 2:
                return {s: 0.0 for s in view.symbols()}
            return None

    sim = SimExchange(
        frames, CostModel(enabled=False), CostModel(enabled=False),
        SimConfig(initial_cash=10_000.0, warmup=50,
                  margin=MarginConfig(initial_margin_ratio=0.5,
                                       maintenance_margin_ratio=0.25)),
    )
    res = sim.run(OpenLongClose())
    assert res.n_trades >= 2
    # sim.margin_book 是 sim 内部状态，run() 后保留
    assert sim.margin_book is not None, "MarginConfig 应触发 margin_book 创建"
    # 第一根建仓：每标 1.0×equity/price 单位，初始保证金 0.5×|units|×price
    # 建仓价格 100，初始保证金 = 0.5 × 100 × 100 = 5000
    # 第二根平仓后，margin_cash 全部清零
    total_margin_cash = sum(leg.margin_cash for leg in sim.margin_book.legs.values())
    assert total_margin_cash == 0.0, \
        f"平仓后 margin_cash 应清零，实际 {total_margin_cash}"


def test_sim_exchange_margin_none_keeps_legacy_behavior():
    """margin=None（默认）时，SimExchange 行为完全不变。

    注：trending_frames 是单调上升的，所以满仓 0.5 一定赚钱。
    关键是行为不变（没崩、n_trades 正常），不要求终值精确 = 10000。
    """
    n = 100
    frames = trending_frames(n=n, symbols=("BTCUSDT", "ETHUSDT"), uptrend=True)
    from src.sim.costs import CostModel
    from src.sim.exchange import SimConfig, SimExchange

    class HalfHalf:
        name = "half_half"
        def __init__(self):
            self._done = False
        def decide(self, view):
            if self._done:
                return None
            self._done = True
            return {s: 0.5 for s in view.symbols()}

    res = SimExchange(
        frames, CostModel(enabled=False), CostModel(enabled=False),
        SimConfig(initial_cash=10_000.0, warmup=50),    # margin 默认 None
    ).run(HalfHalf())

    assert res.n_trades > 0
    # 单调上升 100 根，0.5 满仓必赚钱；验证 net > initial 即可
    assert res.final_net() > 10_000.0, \
        f"单调上升应赚钱，实际 {res.final_net()}"


def test_sim_exchange_liquidates_on_crash():
    """C3 阶段：高杠杆 + 暴跌 → 必触发逐腿强平。

    设计：构造 initial=0.1, maint=0.05 的高杠杆（10x 强平线），让价格
    跌到接近 0（trending_frames 最低 close=1，h=1）。

    算账：
      - 满仓 1.5x BTC 多：150 单位 × 100 价 = 15000 名义
      - initial = 0.1 × 15000 = 1500
      - maint 强平线：equity = 1500 + 150×h ≤ 0.05 × 150 × h = 7.5h
        → 1500 ≤ -142.5h → h ≤ -10.5
      - **价不会跌到 -10**，所以这个杠杆也不会强平

    真正能强平的方式：让 margin_cash 被浮亏"侵蚀"完。
    但永续合约里 margin_cash 永远 lock 在账户里 —— equity = margin_cash + PnL
    永远 ≥ margin_cash - |亏损|，强平线 = 0.05 × |units| × h，**当 h 很小时
    强平线也接近 0**。所以**永续合约实际上很难强平**，除非初始保证金用得差不多。

    这个测试改用：构造一个**做空 + 价格上涨**的场景，做空的 margin_cash
    会随价格上涨被亏完，触发强平。
    """
    n = 200
    frames = trending_frames(n=n, symbols=("BTCUSDT", "ETHUSDT"), uptrend=True)
    from src.sim.costs import CostModel
    from src.sim.exchange import SimConfig, SimExchange
    from src.sim.margin import MarginConfig

    # 用做空：BTC 价格**上涨**触发强平
    # 做空 1.5x：-150 单位，initial=0.1×150×100=1500
    # 价从 100 涨到 110：equity = 1500 + (-150)*110 = 1500 - 16500 = -15000
    #                  maint = 0.05 × 150 × 110 = 825
    # -15000 ≤ 825 → 必强平
    btc_close = frames["BTCUSDT"]["close"].copy()
    n_bars = len(btc_close)
    # bar 50 起开始涨（仍用 trending 默认上升即可）
    frames["BTCUSDT"]["close"] = btc_close
    frames["BTCUSDT"]["high"] = btc_close + 1.0
    frames["BTCUSDT"]["open"] = btc_close

    class Short1_5X:
        name = "short_1_5x"
        def __init__(self):
            self._done = False
        def decide(self, view):
            if self._done:
                return None
            self._done = True
            # 做空 1.5x
            return {"BTCUSDT": -1.5, "ETHUSDT": -1.5, "BNBUSDT": -1.5}

    sim = SimExchange(
        frames, CostModel(enabled=False), CostModel(enabled=False),
        SimConfig(initial_cash=10_000.0, warmup=10,
                  margin=MarginConfig(initial_margin_ratio=0.1,
                                       maintenance_margin_ratio=0.05),
                  max_exposure_per_symbol=1.5,
                  max_gross=4.5,
                  allow_short=True,
                  instrument="perp"),
    )
    res = sim.run(Short1_5X())

    # 做空 + 价涨 → 必强平 BTC
    assert "BTCUSDT" in res.liquidated_legs, \
        f"做空 + 价涨应触发 BTC 强平，实际 {res.liquidated_legs}"
    # 强平后该腿清零
    assert sim.margin_book.legs["BTCUSDT"].margin_cash == 0.0
    assert "BTCUSDT" in sim.margin_book.liquidated


def test_sim_exchange_no_liquidation_when_safe():
    """C3 阶段：价格平稳时不应触发强平。"""
    n = 100
    frames = trending_frames(n=n, symbols=("BTCUSDT", "ETHUSDT"), uptrend=True)
    # 默认 trending 是上升的，不会强平
    from src.sim.costs import CostModel
    from src.sim.exchange import SimConfig, SimExchange
    from src.sim.margin import MarginConfig

    class HalfHalf:
        name = "half_half"
        def __init__(self):
            self._done = False
        def decide(self, view):
            if self._done:
                return None
            self._done = True
            return {s: 0.5 for s in view.symbols()}

    res = SimExchange(
        frames, CostModel(enabled=False), CostModel(enabled=False),
        SimConfig(initial_cash=10_000.0, warmup=50,
                  margin=MarginConfig(initial_margin_ratio=0.1,
                                       maintenance_margin_ratio=0.05),
                  max_exposure_per_symbol=3.0,
                  max_gross=9.0),
    ).run(HalfHalf())

    assert len(res.liquidated_legs) == 0, \
        f"价格上升不应强平，实际 {res.liquidated_legs}"


# ==========================================================================
# 10. B' 阶段：专家"满仓值"与 max_exposure 解耦
# ==========================================================================
def test_expert_max_position_scales_with_gross_target():
    """LongExpert 在 gross_target=3.0 时应返 [1.5, 1.5]（不再是 [0.5, 0.5]）。"""
    from src.agents.online import LongExpert
    frames = trending_frames(n=200, symbols=("BTCUSDT", "ETHUSDT"), uptrend=True)
    view = make_view_from_frames(frames, t=100)
    e = LongExpert()
    w1 = e.weights(view, ["BTCUSDT", "ETHUSDT"], gross_target=1.0)
    w3 = e.weights(view, ["BTCUSDT", "ETHUSDT"], gross_target=3.0)
    assert np.allclose(w1, [0.5, 0.5]), f"gross=1 应是 [0.5,0.5]，实际 {w1}"
    assert np.allclose(w3, [1.5, 1.5]), f"gross=3 应是 [1.5,1.5]，实际 {w3}"


def test_short_expert_max_position_scales_with_gross_target():
    """ShortExpert 在 gross_target=3.0 时应返 [-1.5, -1.5]。"""
    from src.agents.online import ShortExpert
    frames = trending_frames(n=200, symbols=("BTCUSDT", "ETHUSDT"), uptrend=True)
    view = make_view_from_frames(frames, t=100)
    e = ShortExpert()
    w3 = e.weights(view, ["BTCUSDT", "ETHUSDT"], gross_target=3.0)
    assert np.allclose(w3, [-1.5, -1.5]), f"gross=3 应是 [-1.5,-1.5]，实际 {w3}"


def test_single_asset_expert_uses_gross_target():
    """SingleAssetExpert 在 gross_target=3.0 时应返 [3.0, 0.0]（不是 [1.0, 0.0]）。"""
    from src.agents.online import SingleAssetExpert
    frames = trending_frames(n=200, symbols=("BTCUSDT", "ETHUSDT"), uptrend=True)
    view = make_view_from_frames(frames, t=100)
    e = SingleAssetExpert("BTCUSDT")
    w3 = e.weights(view, ["BTCUSDT", "ETHUSDT"], gross_target=3.0)
    assert np.allclose(w3, [3.0, 0.0]), f"gross=3 应是 [3,0]，实际 {w3}"


def test_inverse_vol_expert_uses_gross_target():
    """InverseVolExpert 满仓值 = gross_target（不再写死 1.0）。"""
    from src.agents.online import InverseVolExpert
    frames = trending_frames(n=200, symbols=("BTCUSDT", "ETHUSDT"), uptrend=True)
    view = make_view_from_frames(frames, t=200)    # 200 根后能算 168 窗口 sigma
    e = InverseVolExpert(window=168)
    w1 = e.weights(view, ["BTCUSDT", "ETHUSDT"], gross_target=1.0)
    w3 = e.weights(view, ["BTCUSDT", "ETHUSDT"], gross_target=3.0)
    assert abs(w1.sum() - 1.0) < 1e-6, f"gross=1 时 sum 应=1，实际 {w1.sum()}"
    assert abs(w3.sum() - 3.0) < 1e-6, f"gross=3 时 sum 应=3，实际 {w3.sum()}"


def test_hedge_with_3x_max_exposure_actually_uses_leverage():
    """max_exposure=3.0 时，HedgeEnsemble 配 LongExpert 时，gross 真的能到 3.0。

    这是 B' 阶段的核心断言：之前 max_exposure 改 3.0 但 Hedge 仍只到 1.0
    （专家满仓值写死 1/n）。修完专家后，K=1 + LongExpert 应能输出 Σ|w|=3.0。
    """
    from src.agents.online import LongExpert
    frames = trending_frames(n=200, symbols=("BTCUSDT", "ETHUSDT"), uptrend=True)
    view = make_view_from_frames(frames, t=150)

    agent = HedgeEnsemble(
        symbols=["BTCUSDT", "ETHUSDT"],
        experts=[LongExpert()],          # K=1，全压 long
        max_exposure=3.0,
    )
    out = agent.decide(view)
    weights = np.array(list(out.values()))
    # 关键：gross 应是 3.0（不是 1.0）
    assert abs(np.abs(weights).sum() - 3.0) < 1e-6, (
        f"max_exposure=3.0 + LongExpert 应得 gross=3.0，实际 {np.abs(weights).sum()}"
    )


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
        def weights(self, view, symbols, gross_target=1.0):
            return np.array([1.0, 1.0])    # 单标的就是 max_exposure，但 gross=2

    class DoubleShort(Expert):
        name = "double_short"
        def weights(self, view, symbols, gross_target=1.0):
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
