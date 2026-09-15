"""成本模型 —— 手续费 / 点差 / 市场冲击（临时 + 永久）。

设计依据（见 PLAN.md §2）：
  临时冲击 ≈ k · σ · sqrt(Q / V)        （平方根律）
  永久冲击 ≈ γ · Q / V
  Q = 本笔名义额，V = 同周期市场成交名义额，σ = 同周期收益率标准差

比例项（手续费、点差、临时冲击）随名义额线性增长；
永久冲击虽有比例形式，但它改变的是后续价格水平，本项目保守地按一次性成本计。

**口径说明（与 `impact_estimate.py` 一致）**：σ 与 V 都用小时口径，
即 σ_h · √(Q/V_h)。这与日频约定等价——因为 σ_d = σ_h·√24、V_d = 24·V_h，
故 σ_d·√(Q/V_d) = σ_h·√(Q/V_h)。所以 `impact_k=1.0` 两种口径下是同一个数。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class CostModel:
    fee_taker: float = 0.001
    fee_maker: float = 0.001
    bnb_discount: float = 0.75
    use_bnb_discount: bool = True
    impact_k: float = 1.0
    impact_gamma_perm: float = 0.2
    spread_bp: dict = field(default_factory=dict)     # 单边，基点
    default_spread_bp: float = 1.0
    enabled: bool = True                              # False → 全部成本归零，用于 gross 对照
    # 冲击率的物理上限。**这不是调参旋钮，是公式的适用域护栏。**
    #
    # 平方根律 + 线性永久冲击项只在 part = Q/V ≪ 1 的区间成立。part 变大时
    # 线性项让成本 ∝ Q²/V，于是会算出"成本超过本笔成交额本身"——即掉进一个
    # 费率 > 100% 的世界，那里没有任何经济现实：你不可能为一笔 100 元的委托
    # 支付 200 元的冲击。
    #
    # 实测（2026-09-15）在 3x 杠杆 + 逐腿强平的配置下，未加护栏会让 `part` 达到
    # 6.1、单笔名义额达到 1.86 亿（本金 1 万），累计成本 35 亿——把 D 阶段全部
    # 结论污染成"成本是净值的 1.6 倍"。加护栏后极端值被截断并**计数上报**，
    # 于是"某个策略跑到了不可行的规模"这件事会被看见，而不是伪装成巨额成本。
    #
    # 取 1.0 的依据：冲击成本 = 名义额 × 费率，费率上限 100% 是"成本不超过本金"
    # 这一硬约束的边界（更严的取值需要另立校准，不在本 bug 修复范围内）。
    max_impact_rate: float = 1.0
    # 统计护栏被触发的次数。>0 说明有成交落在公式适用域之外，该结果不可用于
    # 横向比较（规模已不可行），只应用来看"这个策略在这个资本规模下不能跑"。
    n_impact_capped: int = field(default=0, init=False)

    # ---------------- 手续费 ----------------
    @property
    def fee_rate(self) -> float:
        r = self.fee_taker
        if self.use_bnb_discount:
            r *= self.bnb_discount
        return r

    def fee(self, notional: float, taker: bool = True) -> float:
        if not self.enabled:
            return 0.0
        rate = self.fee_rate if taker else self.fee_maker * (
            self.bnb_discount if self.use_bnb_discount else 1.0)
        return abs(notional) * rate

    # ---------------- 点差 ----------------
    def spread_rate(self, symbol: str) -> float:
        return self.spread_bp.get(symbol, self.default_spread_bp) / 1e4

    def spread(self, symbol: str, notional: float) -> float:
        if not self.enabled:
            return 0.0
        return abs(notional) * self.spread_rate(symbol)

    # ---------------- 冲击 ----------------
    def raw_impact_rate(self, notional: float, sigma: float,
                        period_notional: float) -> float:
        """未加护栏的原始冲击率。保留它只为**诊断**（看公式外推到多远），
        任何记账路径都必须走 `impact_rate`。"""
        if not self.enabled or period_notional <= 0:
            return 0.0
        part = abs(notional) / period_notional
        return self.impact_k * sigma * math.sqrt(part) + self.impact_gamma_perm * part

    def impact_rate(self, notional: float, sigma: float, period_notional: float) -> float:
        """返回单边冲击率（占名义额比例），并被 `max_impact_rate` 截断。"""
        r = self.raw_impact_rate(notional, sigma, period_notional)
        if r > self.max_impact_rate:
            self.n_impact_capped += 1
            return self.max_impact_rate
        return r

    def impact(self, notional: float, sigma: float, period_notional: float) -> float:
        return abs(notional) * self.impact_rate(notional, sigma, period_notional)

    # ---------------- 总成本 ----------------
    def trade_cost(self, symbol: str, notional: float, sigma: float,
                   period_notional: float, taker: bool = True) -> float:
        return (self.fee(notional, taker)
                + self.spread(symbol, notional)
                + self.impact(notional, sigma, period_notional))

    def round_trip_bp(self, symbol: str, order_usd: float, sigma: float,
                      period_notional: float) -> dict:
        """往返成本拆解（基点），供报告使用。"""
        f = 2 * self.fee_rate * 1e4
        s = 2 * self.spread_rate(symbol) * 1e4
        i = 2 * self.impact_rate(order_usd, sigma, period_notional) * 1e4
        return {"fee_bp": f, "spread_bp": s, "impact_bp": i, "total_bp": f + s + i}

    @classmethod
    def from_config(cls, cfg: dict, enabled: bool = True) -> "CostModel":
        c = cfg["costs"]
        return cls(
            fee_taker=c["fee_taker"], fee_maker=c["fee_maker"],
            bnb_discount=c["bnb_discount"], use_bnb_discount=c["use_bnb_discount"],
            impact_k=c["impact_k"], impact_gamma_perm=c["impact_gamma_perm"],
            spread_bp=c.get("spread_bp") or {}, enabled=enabled,
        )
