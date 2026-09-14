"""成本模型 —— 手续费 / 点差 / 市场冲击（临时 + 永久）。

设计依据（见 PLAN.md §2）：
  临时冲击 ≈ k · σ · sqrt(Q / V)        （平方根律）
  永久冲击 ≈ γ · Q / V
  Q = 本笔名义额，V = 同周期市场成交名义额，σ = 同周期收益率标准差

比例项（手续费、点差、临时冲击）随名义额线性增长；
永久冲击虽有比例形式，但它改变的是后续价格水平，本项目保守地按一次性成本计。
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
    def impact_rate(self, notional: float, sigma: float, period_notional: float) -> float:
        """返回单边冲击率（占名义额比例）。"""
        if not self.enabled or period_notional <= 0:
            return 0.0
        part = abs(notional) / period_notional
        return self.impact_k * sigma * math.sqrt(part) + self.impact_gamma_perm * part

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
