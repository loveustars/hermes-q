"""仿真交易所 —— 单次遍历，双账本（毛收益 / 净收益）。

关键设计：**同一次遍历同时维护两个账本**，两者的交易决策完全相同，
差别只在于是否扣除成本。这样"成本吃掉了多少收益"是严格同口径的对比，
而不是跑两遍拿可能不一致的结果来比。

执行约定（防未来函数）：
  - 策略在第 t 根 bar 收盘后做决策，只能看到 0..t
  - 订单挂到 pending 队列，在第 t+latency 根 bar 的**开盘价**成交
  - latency 支持抖动（泊松），模拟真实下单延迟的不确定性
  - 冲击模型用 t 时刻可见的 σ 与成交额，不用未来值
  - 资金费在每个 8 小时结算点上按持仓名义额收取（仅净账本，单独计量）

资金费只进净账本：毛账本的口径是"假如交易完全免费"，
而资金费不是摩擦成本，是持仓的持有成本，必须单独可见。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..env.market_view import MarketView, Precomputed
from .costs import CostModel
from .funding import FundingTable
from .margin import MarginBook


@dataclass
class SimConfig:
    initial_cash: float = 10_000.0
    warmup: int = 300
    latency_bars: int = 1
    allow_short: bool = False
    max_gross: float = 1.0
    sigma_window: int = 168          # 一周小时线，用于冲击模型
    vol_window: int = 168            # 冲击模型里的成交量窗口
    min_trade_notional: float = 1.0  # 小于此金额不下单（避免无穷小换手）
    # 缺口处理。实测 BTC/ETH 各 27~28 处、最长 34 小时，三标的缺口位置一致（交易所停机）。
    #   "execute"（默认）：照常成交。交易所当时确实闭市，下一个可成交价就是复牌开盘价。
    #   "skip"：跨缺口那一根不交易，只做盯市。更保守，用于敏感性对照。
    gap_policy: str = "execute"
    # 仪器类型：spot 无资金费；perp 计资金费（并允许做空）
    instrument: str = "spot"
    # 延迟抖动：实际成交延迟 = latency_bars + Poisson(jitter_mean)，上限 latency_max_bars
    latency_jitter_mean: float = 0.0
    latency_jitter_seed: int = 0
    latency_max_bars: int = 5
    # per-symbol 杠杆上限（C 阶段）：单标的 |w_s| ≤ max_exposure_per_symbol。
    # 与 max_gross 形成二层防御：max_gross 是组合上限，max_exposure_per_symbol 是单标上限。
    max_exposure_per_symbol: float = 3.0
    # 保证金配置（C 阶段）：None = 不启用保证金追踪（向后兼容）。
    # 启用后，建仓扣 initial_margin，平仓退 margin_cash + 浮动 PnL。
    # 强平由 max_exposure_per_symbol 范围内的逐腿强平判定（SimConfig 仍可有 cfg.margin）。
    margin: "MarginConfig | None" = None
    # 破产闸门（2026-09-15 修）。净权益 ≤ 此值时**立即停止交易**并截断曲线。
    #
    # 为什么必须有：仓位按 `want_n = target_w[s] * eq_n / px[s]` 定，而这条式子
    # 在 eq_n < 0 时会**翻号**——想要 3x 做多的策略变成巨额做空，且幅度随 |eq_n|
    # 增大而失控。实测在 3x + 逐腿强平配置下，权益到 −23,467 后环境仍在交易，
    # 单笔名义额冲到 1.86 亿（本金 1 万），权益从 206 涨到 260 万（12,600 倍）——
    # 这个"反弹"完全来自破产账户被强制做空的意外，与策略无关。
    #
    # 现实里这种账户在权益转负之前就被强平了（强平的意义就在于此），所以
    # "权益 ≤ 0 还继续下单"没有任何对应物。设为 0.0 表示"资不抵债即出局"。
    insolvency_floor: float = 0.0
    stop_when_bankrupt: bool = True

    def __post_init__(self):
        if self.gap_policy not in ("execute", "skip"):
            raise ValueError(f"gap_policy 只能是 execute / skip，收到 {self.gap_policy!r}")
        if self.instrument not in ("spot", "perp"):
            raise ValueError(f"instrument 只能是 spot / perp，收到 {self.instrument!r}")
        if self.instrument == "spot" and self.allow_short:
            raise ValueError("现货模式不允许做空；做空请设 instrument='perp'")
        if self.latency_jitter_mean < 0:
            raise ValueError("latency_jitter_mean 不能为负")
        if self.max_exposure_per_symbol <= 0:
            raise ValueError("max_exposure_per_symbol 必须 > 0")


@dataclass
class SimResult:
    index: pd.DatetimeIndex
    net_equity: np.ndarray
    gross_equity: np.ndarray
    cost_paid: np.ndarray
    turnover_notional: np.ndarray
    n_trades: int
    initial_cash: float = 10_000.0
    insolvent_at: int | None = None
    # 破产闸门触发的位置（见 SimConfig.insolvency_floor）。
    # **非 None 时曲线在这一点被截断**：net_equity / index / cost_paid 等都只到
    # 死亡那一根。于是下游指标（total_ret、sharpe、max_dd）都只覆盖"活着的时段"，
    # 不会用一条虚构的复活曲线去算夏普。
    bankrupt_at: int | None = None
    # 闸门触发时的原始盯市值 —— 可能 < 0（单根 bar 亏掉的不止全部权益）。
    # 曲线里记的是归零后的 0.0；这个字段只为排查保留"本来跌到多深"。
    bankrupt_equity_raw: float | None = None
    n_gap_bars: int = 0
    n_gap_trades: int = 0
    n_gap_skipped: int = 0
    funding_paid: np.ndarray = field(default_factory=lambda: np.zeros(0))
    n_funding_events: int = 0
    latency_histogram: dict = field(default_factory=dict)
    symbols: list[str] = field(default_factory=list)
    # 冲击率被适用域护栏截断的成交笔数（见 CostModel.max_impact_rate）。
    # >0 表示该策略的规模已超出平方根律的适用范围，**结果不可用于横向比较**。
    n_impact_capped: int = 0
    liquidated_legs: list[str] = field(default_factory=list)    # C 阶段：被强平的标的列表（按时间顺序）

    @property
    def net_ret(self) -> np.ndarray:
        return np.diff(self.net_equity) / self.net_equity[:-1]

    @property
    def gross_ret(self) -> np.ndarray:
        return np.diff(self.gross_equity) / self.gross_equity[:-1]

    @property
    def insolvent(self) -> bool:
        return self.insolvent_at is not None

    @property
    def bankrupt(self) -> bool:
        """资不抵债、已被破产闸门中止（曲线已截断）。比 `insolvent` 更强。"""
        return self.bankrupt_at is not None

    def final_net(self) -> float:
        return float(self.net_equity[-1])

    def final_gross(self) -> float:
        return float(self.gross_equity[-1])

    def cost_drag_bp(self) -> float:
        """全周期成本占**初始本金**的比例（基点）—— 不能用 e[0] 做基数。"""
        return float(self.cost_paid.sum() / self.initial_cash * 1e4)

    def total_funding(self) -> float:
        return float(self.funding_paid.sum()) if len(self.funding_paid) else 0.0


class SimExchange:
    def __init__(self, frames: dict[str, pd.DataFrame], gross_costs: CostModel,
                 net_costs: CostModel, cfg: SimConfig,
                 funding: FundingTable | None = None):
        self.frames = frames
        self.symbols = list(frames)
        self.gross_costs = gross_costs     # enabled=False
        self.net_costs = net_costs         # enabled=True
        self.cfg = cfg
        self.funding = funding
        lens = {len(f) for f in frames.values()}
        if len(lens) != 1:
            raise ValueError(f"各标的 bar 数不一致: {lens}，请先对齐")
        self.T = lens.pop()
        idx = frames[self.symbols[0]].index
        for s in self.symbols[1:]:
            if not frames[s].index.equals(idx):
                raise ValueError(f"{s} 的时间轴与其他标的不一致，请先对齐")
        # 滚动统计预计算：等价但快一到两个数量级（详见 Precomputed 的说明）
        self.pre = Precomputed(frames, {cfg.sigma_window}, {cfg.vol_window})
        # C 阶段：保证金账本（per-symbol 逻辑子账户）
        # margin=None 时不启用（向后兼容），C3 强平也不会触发
        self.margin_book = (MarginBook(cfg.margin, list(frames))
                            if cfg.margin is not None else None)

    def _sanitize(self, target: dict, view: MarketView) -> dict[str, float]:
        out = {}
        for s in self.symbols:
            w = float(target.get(s, 0.0))
            if not np.isfinite(w):
                w = 0.0
            if not self.cfg.allow_short:
                w = max(w, 0.0)
            # per-symbol 杠杆上限（C 阶段）：单标的 |w_s| ≤ max_exposure_per_symbol
            w = max(min(w, self.cfg.max_exposure_per_symbol),
                    -self.cfg.max_exposure_per_symbol)
            out[s] = w
        gross = sum(abs(v) for v in out.values())
        if gross > self.cfg.max_gross and gross > 0:
            k = self.cfg.max_gross / gross
            out = {s: v * k for s, v in out.items()}
        return out

    # ------------------------------------------------------------------
    def run(self, agent) -> SimResult:
        cfg = self.cfg
        start = cfg.warmup
        if start >= self.T - cfg.latency_bars - 1:
            raise ValueError("warmup 过长，剩余 bar 不足")

        jitter_rng = np.random.default_rng(cfg.latency_jitter_seed)
        use_funding = (cfg.instrument == "perp" and self.funding is not None
                       and len(self.funding.rates_by_hour) > 0)

        cash_n = cash_g = cfg.initial_cash
        units_n = {s: 0.0 for s in self.symbols}
        units_g = {s: 0.0 for s in self.symbols}

        idx_out, net_eq, gross_eq = [], [], []
        cost_paid, turnover, funding_paid = [], [], []
        n_trades = 0
        n_funding_events = 0
        insolvent_at: int | None = None
        bankrupt_at: int | None = None
        bankrupt_equity_raw: float | None = None   # 闸门触发时的原始盯市值（可能为负）
        n_gap_trades = 0
        n_gap_skipped = 0
        latency_hist: dict[int, int] = {}
        gap_mask = self.pre.gaps if self.pre is not None else None
        n_gap_bars = int(gap_mask.sum()) if gap_mask is not None else 0
        result_liquidated_legs: list[str] = []        # C 阶段：被强平的标的（按时间顺序）

        first_record = start + cfg.latency_bars
        pending: dict[int, tuple[dict[str, float], int]] = {}
        idx = self.frames[self.symbols[0]].index

        for t in range(start, self.T):
            ex = t
            at_gap = bool(gap_mask[ex]) if gap_mask is not None else False
            bar_cost = 0.0
            bar_turnover = 0.0
            bar_funding = 0.0

            # ---------- 1) 执行到期订单（成交价 = 本根 open）----------
            due = pending.pop(ex, None)
            if due is not None:
                target_w, decision_step = due
                if at_gap and cfg.gap_policy == "skip":
                    n_gap_skipped += 1
                else:
                    if at_gap:
                        n_gap_trades += 1
                    # 冲击模型的 σ 与成交额用**决策时刻**可见的值（因果），
                    # 不用执行时刻的，避免延迟抖动带来隐性未来信息。
                    dview = MarketView(self.frames, decision_step, self.pre)
                    px = {s: float(self.frames[s]["open"].iloc[ex]) for s in self.symbols}
                    eq_n = cash_n + sum(units_n[s] * px[s] for s in self.symbols)
                    eq_g = cash_g + sum(units_g[s] * px[s] for s in self.symbols)
                    for s in self.symbols:
                        sigma = dview.sigma(s, cfg.sigma_window)
                        pn = dview.period_notional(s, cfg.vol_window)

                        want_n = target_w[s] * eq_n / px[s]
                        d_n = want_n - units_n[s]
                        notional_n = abs(d_n) * px[s]
                        if notional_n >= cfg.min_trade_notional:
                            c = self.net_costs.trade_cost(s, notional_n, sigma, pn)
                            cash_n -= c
                            cash_n -= d_n * px[s]
                            old_units = units_n[s]
                            units_n[s] = want_n
                            # C 阶段：保证金账本同步
                            # 注意 `[cash_n]` 只是装了 cash_n 副本的列表，改它不影响
                            # 局部变量 cash_n —— 所以凡是要动现金的路径都必须用**返回值**。
                            if self.margin_book is not None:
                                if abs(old_units) < 1e-9 and abs(want_n) >= 1e-9:
                                    self.margin_book.open_leg(
                                        s, want_n, px[s], [cash_n])
                                elif abs(want_n) < 1e-9 and abs(old_units) >= 1e-9:
                                    self.margin_book.close_leg(
                                        s, old_units, px[s], [cash_n])
                                elif old_units * want_n < 0:
                                    # 多空翻转 = 旧腿结束 + 新腿开始。必须重锚，
                                    # 否则浮动盈亏会拿旧方向的建仓价算（符号反了）。
                                    self.margin_book.open_leg(
                                        s, want_n, px[s], [cash_n])
                            bar_cost += c
                            n_trades += 1

                        want_g = target_w[s] * eq_g / px[s]
                        d_g = want_g - units_g[s]
                        notional_g = abs(d_g) * px[s]
                        if notional_g >= cfg.min_trade_notional:
                            cash_g -= d_g * px[s]
                            units_g[s] = want_g
                            bar_turnover += notional_g

            # ---------- 1.5) 逐腿强平判定（C 阶段）----------
            # 用**不利价**做压力测试（多头 low、空头 high），触发时按不利价强制平仓。
            #
            # 必须用 liquidate 的**返回值**加到 cash_n：`[cash_n]` 是浮点副本的列表，
            # 改它改不到 cash_n。初版正是踩了这个坑——`cash_pool[0] += release` 是
            # 空操作，而 `units_n[s] = 0.0` 却真的执行了，于是空头的负债凭空消失、
            # 权益凭空增加 |units| × 价格。这是 D 阶段"权益冲到 2.1 亿"的直接原因。
            if self.margin_book is not None:
                for s in self.symbols:
                    if s in self.margin_book.liquidated:
                        continue
                    if abs(units_n[s]) < 1e-9:
                        continue
                    low_px = float(self.frames[s]["low"].iloc[ex])
                    high_px = float(self.frames[s]["high"].iloc[ex])
                    adverse_px = self.margin_book.adverse_price(
                        units_n[s], low_px, high_px)
                    if self.margin_book.is_liquidatable(
                            s, units_n[s], adverse_px):
                        realized = self.margin_book.liquidate(
                            s, units_n[s], adverse_px, [cash_n])
                        cash_n += realized          # ← 真正的现金动作
                        units_n[s] = 0.0
                        result_liquidated_legs.append(s)

            # ---------- 2) 资金费结算（按收盘价近似结算价）----------
            if use_funding:
                ts_ms = int(idx[ex].timestamp() * 1000)
                for s in self.symbols:
                    rate = self.funding.rate_or_none(s, ts_ms)
                    if rate is None or units_n[s] == 0.0:
                        continue
                    price = float(self.frames[s]["close"].iloc[ex])
                    pay = units_n[s] * price * rate
                    cash_n -= pay
                    bar_funding += pay
                    n_funding_events += 1

            # ---------- 3) 盯市 ----------
            if ex >= first_record:
                pxc = {s: float(self.frames[s]["close"].iloc[ex]) for s in self.symbols}
                idx_out.append(idx[ex])
                net_eq.append(cash_n + sum(units_n[s] * pxc[s] for s in self.symbols))
                gross_eq.append(cash_g + sum(units_g[s] * pxc[s] for s in self.symbols))
                cost_paid.append(bar_cost)
                turnover.append(bar_turnover)
                funding_paid.append(bar_funding)
                if insolvent_at is None and net_eq[-1] < 0.01 * cfg.initial_cash:
                    insolvent_at = len(net_eq) - 1
                # 破产闸门：资不抵债即出局，**必须先于下一根 bar 的建仓**。
                # 放在盯市之后，好让曲线包含"死掉的那一根"（死前权益是真实记录的）。
                if (cfg.stop_when_bankrupt and bankrupt_at is None
                        and net_eq[-1] <= cfg.insolvency_floor):
                    bankrupt_at = len(net_eq) - 1
                    # **强平语义：交易所不会让你欠钱 —— 权益归零，不是负值。**
                    # 单根 bar 内的损失可以超过全部权益（3x + 高相关标的同步下跌时
                    # 常见）。若把负数如实留在曲线里，下游会算出
                    #   total_ret < −100%、max_dd < −100%（经济上不可能），
                    # 而且 sharpe 会变成**无意义的正数**（简单收益率的均值可以为
                    # 正，而复利终值已经归零），进而把已爆仓的配置排到排行榜第一。
                    # 原始盯市值另存 bankrupt_equity_raw，供排查用。
                    bankrupt_equity_raw = float(net_eq[-1])
                    net_eq[-1] = max(0.0, net_eq[-1])
                    gross_eq[-1] = max(0.0, gross_eq[-1])
                    break

            # ---------- 4) 决策（只能在看到 t 及之前的数据后做）----------
            if ex >= self.T - 1:
                continue
            view = MarketView(self.frames, ex, self.pre)
            target = agent.decide(view)
            if target is None:
                continue
            lat = cfg.latency_bars
            if cfg.latency_jitter_mean > 0:
                lat += int(jitter_rng.poisson(cfg.latency_jitter_mean))
            lat = int(max(1, min(lat, cfg.latency_max_bars)))
            latency_hist[lat] = latency_hist.get(lat, 0) + 1
            exec_at = min(ex + lat, self.T - 1)
            pending[exec_at] = (self._sanitize(target, view), ex)

        return SimResult(
            index=pd.DatetimeIndex(idx_out),
            net_equity=np.asarray(net_eq, dtype=float),
            gross_equity=np.asarray(gross_eq, dtype=float),
            cost_paid=np.asarray(cost_paid, dtype=float),
            turnover_notional=np.asarray(turnover, dtype=float),
            n_trades=n_trades,
            initial_cash=cfg.initial_cash,
            insolvent_at=insolvent_at,
            bankrupt_at=bankrupt_at,
            bankrupt_equity_raw=bankrupt_equity_raw,
            n_impact_capped=self.net_costs.n_impact_capped,
            n_gap_bars=n_gap_bars,
            n_gap_trades=n_gap_trades,
            n_gap_skipped=n_gap_skipped,
            funding_paid=np.asarray(funding_paid, dtype=float),
            n_funding_events=n_funding_events,
            latency_histogram=latency_hist,
            symbols=self.symbols,
            liquidated_legs=result_liquidated_legs,
        )
