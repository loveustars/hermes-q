"""逐腿保证金账户 —— C 阶段 §19 主循环改造的核心组件。

设计要点：
  - 单桶 cash（整体账户），但 per-symbol 维护 margin_cash（逻辑子账户）用于强平判定
  - 这与 carry.py 的 spot+perp 双账户不同：SimExchange 是单账户主 sim，
    carry.py 是独立的 carry 仿真器，两者用途不一样
  - 强平判定：用本根 high（不是 close！）做压力测试 —— 真实交易所也是按最不利价算
  - 强平后：该腿 units 清零、margin_cash 清零，但 sim 继续跑

与 carry.py 的关系：
  - carry.py 的 margin_cash 是"永续腿保证金账户"，与现货腿的 reserve 隔离
  - 这里 margin_cash[s] 是"symbol s 的逻辑保证金账户"，整体 cash 是 sum(margin_cash) + 浮动 PnL
  - 数值上 carry.py 的"权益"≈ 这里 margin_cash[s] + units[s] * price（一致）
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class MarginConfig:
    """逐腿保证金参数。

    initial_margin_ratio: 开仓时需存入的保证金占名义额的比例（> 1 表示需要更多保证金）
    maintenance_margin_ratio: 维持保证金率（< initial_margin_ratio，强平线）
    topup_trigger_ratio: 保证金权益跌破初始保证金的该比例时，触发补保
    liquidation_penalty_bp: 强平罚金（基点），从保证金账户扣除（0 表示不罚）
    """
    initial_margin_ratio: float = 0.5       # 50% 初始保证金 = 2x 杠杆
    maintenance_margin_ratio: float = 0.25  # 25% 维持保证金 = 4x 杠杆强平
    topup_trigger_ratio: float = 0.5        # 跌破 50% 初始保证金时补保
    liquidation_penalty_bp: float = 0.0     # 默认不罚（C 阶段先无罚金）

    def __post_init__(self):
        if not 0.0 < self.maintenance_margin_ratio <= self.initial_margin_ratio:
            raise ValueError(
                f"maintenance_margin_ratio ({self.maintenance_margin_ratio}) "
                f"必须 ≤ initial_margin_ratio ({self.initial_margin_ratio})")
        if not 0.0 < self.topup_trigger_ratio <= 1.0:
            raise ValueError(f"topup_trigger_ratio 必须在 (0, 1] 内")
        if self.liquidation_penalty_bp < 0:
            raise ValueError("liquidation_penalty_bp 不能为负")


@dataclass
class MarginLeg:
    """单条腿的保证金账户（逻辑子账户）。

    margin_cash: 保证金账户现金
    initial_margin: 开仓时存入的初始保证金（用于 topup 触发判断）
    """
    margin_cash: float = 0.0
    initial_margin: float = 0.0


class MarginBook:
    """per-symbol 保证金账本（per-leg MarginLeg 的容器）。"""

    def __init__(self, cfg: MarginConfig, symbols: list[str]):
        self.cfg = cfg
        self.symbols = list(symbols)
        self.legs: dict[str, MarginLeg] = {s: MarginLeg() for s in symbols}
        self.liquidated: set[str] = set()        # 已强平的标的（一次性记录）

    # ------------------------------------------------------------------
    def initial_required(self, units: float, price: float) -> float:
        """开仓需要的初始保证金 = initial_margin_ratio × |units| × price。"""
        return self.cfg.initial_margin_ratio * abs(units) * price

    def maintenance_required(self, units: float, price: float) -> float:
        """维持保证金 = maintenance_margin_ratio × |units| × price。"""
        return self.cfg.maintenance_margin_ratio * abs(units) * price

    def topup_required(self, initial_margin: float) -> float:
        """补保触发线 = topup_trigger_ratio × initial_margin。"""
        return self.cfg.topup_trigger_ratio * initial_margin

    def equity(self, symbol: str, units: float, price: float) -> float:
        """该腿的保证金权益 = margin_cash + units × price（做空时 units<0）。"""
        leg = self.legs[symbol]
        return leg.margin_cash + units * price

    def is_liquidatable(self, symbol: str, units: float, high_price: float) -> bool:
        """用 high 做压力测试：保证金权益 ≤ 维持保证金 → 强平。

        注意：强平判定必须用 high（不是 close），否则低估风险。
        已强平的腿不再被强平（防止重复触发）。
        """
        if symbol in self.liquidated or units == 0.0:
            return False
        eq = self.equity(symbol, units, high_price)
        return eq <= self.maintenance_required(units, high_price)

    def needs_topup(self, symbol: str, units: float, price: float) -> bool:
        """保证金权益跌破 topup_required → 需补保。"""
        if symbol in self.liquidated or units == 0.0:
            return False
        leg = self.legs[symbol]
        if leg.initial_margin <= 0:
            return False
        return self.equity(symbol, units, price) < self.topup_required(leg.initial_margin)

    # ------------------------------------------------------------------
    def open_leg(self, symbol: str, units: float, price: float, cash_pool) -> float:
        """建仓：从 cash_pool 扣初始保证金，写进 margin_cash。

        简化：建仓时直接写 margin_cash = required，initial_margin = required。
        真实交易所更复杂（按 mark price 算 + 强平费 + funding 累积），C 阶段先简化为开仓时锁定。

        cash_pool 是 [cash_value] 形式的可变引用。
        返回扣的金额（正数 = 扣了，0 = 没变）。
        """
        if symbol in self.liquidated:
            raise ValueError(f"{symbol} 已强平，不能再开仓")
        if abs(units) < 1e-9:
            return 0.0
        required = self.initial_required(units, price)
        self.legs[symbol].margin_cash = required
        self.legs[symbol].initial_margin = required
        cash_pool[0] -= required
        return required

    def close_leg(self, symbol: str, units: float, price: float, cash_pool) -> float:
        """平仓：把当前权益（margin_cash + units×price）退到 cash_pool。

        这与 carry.py 强平前的"取出剩余"逻辑一致：平仓时 cash_pool 收到
        当前权益 = 已存入的 margin + 浮动 PnL。

        cash_pool 是 [cash_value] 形式的可变引用。
        返回释放的金额（正数 = 加回 cash_pool）。
        """
        if symbol not in self.legs:
            return 0.0
        if abs(units) < 1e-9:
            return 0.0
        eq = self.equity(symbol, units, price)
        released = eq
        cash_pool[0] += released
        self.legs[symbol].margin_cash = 0.0
        self.legs[symbol].initial_margin = 0.0
        return released

    def topup_to(self, symbol: str, units: float, price: float, cash_pool) -> float:
        """补保：从 cash_pool 扣到 margin_cash，恢复 equity 到 initial_margin 水平。

        cash_pool 是 [cash_value] 形式的可变引用。
        返回补的金额。
        """
        if symbol in self.liquidated or units == 0.0:
            return 0.0
        leg = self.legs[symbol]
        if leg.initial_margin <= 0:
            return 0.0
        eq = self.equity(symbol, units, price)
        target = leg.initial_margin
        if eq >= target:
            return 0.0
        need = target - eq
        take = min(need, max(cash_pool[0], 0.0))
        if take > 0:
            leg.margin_cash += take
            cash_pool[0] -= take
        return take

    def liquidate(self, symbol: str, units: float, high_price: float,
                  cash_pool) -> float:
        """强平：该腿清零，权益（margin_cash + units×high）退到 cash_pool（减去罚金）。

        强平价 = high（最不利价）—— 真实交易所也是按最不利价算。
        cash_pool 是 [cash_value] 形式的可变引用。

        强平后：
          - units 由调用方清零
          - margin_cash 清零
          - initial_margin 清零
          - 标记 liquidated

        返回释放到 cash_pool 的净额（可能为 0，如果 margin_cash + units×high < 罚金）。
        """
        leg = self.legs[symbol]
        eq = leg.margin_cash + units * high_price
        penalty = self.cfg.liquidation_penalty_bp / 1e4 * abs(units) * high_price
        release = max(0.0, eq - penalty)
        cash_pool[0] += release
        leg.margin_cash = 0.0
        leg.initial_margin = 0.0
        self.liquidated.add(symbol)
        return release
