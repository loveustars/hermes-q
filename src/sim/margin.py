"""逐腿保证金账户 —— C 阶段 §19 主循环改造的核心组件。

**2026-09-15 修复说明（重要，阅读者必看）**

本模块初版有三个缺陷，其中第一个会把任何启用杠杆+做空的回测变成印钞机，
是 D 阶段"成本是净值 1.6 倍 / 7,545,571 bp"结论的真正元凶（**不是成本模型**）。
三处修复如下，`tests/test_margin_book.py` 里对应的旧断言已随修复重写。

**缺陷 1：强平时凭空造钱（致命）**
  初版 `open_leg` 只记账不扣现金（理由："SimExchange 已全额扣 cash 买币"），
  但 `liquidate` 却把 margin_cash **真的退给现金池**：
      cash_pool[0] += max(0, margin_cash - penalty)
  那笔保证金从未被单独借记过（建仓时按名义额全额扣了现金），
  于是**每强平一次就白得 `initial_margin_ratio × |units| × 开仓价`**。
  做空时更严重：清掉 u<0 的负债同时还收到正现金，
  收益 = |u| × (initial_margin_ratio × p_open + p_now)，**恒为正**。
  策略于是"学会"故意让空头被强平。实测：权益从 1 万冲到 2.1 亿、434 次强平、
  累计成本 359 亿，而**同样的配置在关闭保证金后完全正常**（最大权益 29.7 万）。
  修复：强平 = **在强平价上强制平仓**，与主循环"全额扣现金买资产"的口径一致，
  即 `cash += units × 强平价 − 罚金`。平仓前后权益连续，不造钱也不烧钱。

**缺陷 2：权益口径错，导致空头一开仓就被强平**
  初版 `equity = margin_cash + units × price`（仓位的**绝对市值**）。
  对空头（units<0）这在开仓瞬间就是 `0.5N − N = −0.5N`，恒为负。
  代入参数得：空头在 `p ≥ 0.4 × p_open` 即判强平 —— 几乎任何价格都满足。
  修复：改用**相对建仓价的浮动盈亏** `(price − entry_price) × units`，
  多空自动对称。修正后 2x 杠杆下多头约 −33% 强平、空头约 +20% 强平，
  两个方向都是量级合理的经济含义。

**缺陷 3：压力测试用错不利价**
  初版对所有方向都用 `high` 判定。但多头的**不利价是 `low`**，
  用 high 会让多头看起来更安全（所以多头几乎从不会被强平）。
  修复：新增 `adverse_price()`，多头取 low、空头取 high。

**关于 `topup_to`**：初版只增加 margin_cash 而不动 cash_pool，同样是凭空造钱。
主循环并未调用它（只有单测在用），但既然它会影响强平判定（经 `equity()`），
一并改成**真正从现金池扣款**、并以可用现金为上限。

设计要点（沿用）：
  - 单桶 cash（整体账户），但 per-symbol 维护 margin_cash（逻辑子账户）用于强平判定
  - 与 carry.py 的 spot+perp 双账户不同：SimExchange 是单账户主 sim
  - 强平判定用**不利价**（多头 low / 空头 high）做压力测试
  - 强平后：该腿 units 清零（由调用方）、账本清零、标记 liquidated、允许日后重开
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class MarginConfig:
    """逐腿保证金参数。

    initial_margin_ratio: 开仓时需存入的保证金占名义额的比例（> 1 表示需要更多保证金）
    maintenance_margin_ratio: 维持保证金率（< initial_margin_ratio，强平线）
    topup_trigger_ratio: 保证金权益跌破初始保证金的该比例时，触发补保
    liquidation_penalty_bp: 强平罚金（基点），从平仓所得中扣除（0 表示不罚）
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
    entry_price: 建仓价 —— **浮动盈亏的基准**（见缺陷 2）。为 0 表示无持仓。
    """
    margin_cash: float = 0.0
    initial_margin: float = 0.0
    entry_price: float = 0.0


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

    def pnl(self, symbol: str, units: float, price: float) -> float:
        """该腿相对**建仓价**的浮动盈亏。多空自动对称（units 带符号）。

        多头 units>0：价涨盈利；空头 units<0：价涨亏损。
        """
        leg = self.legs[symbol]
        if leg.entry_price <= 0.0:
            return 0.0
        return (price - leg.entry_price) * units

    def equity(self, symbol: str, units: float, price: float) -> float:
        """该腿的保证金权益 = margin_cash + 相对建仓价的浮动盈亏。

        **不要**写成 `margin_cash + units × price`（初版的错误，见模块 docstring 缺陷 2）：
        那个式子把仓位的绝对市值当资产，对空头在开仓瞬间就给出负权益。
        """
        leg = self.legs[symbol]
        return leg.margin_cash + self.pnl(symbol, units, price)

    @staticmethod
    def adverse_price(units: float, low: float, high: float) -> float:
        """该腿的**不利价**：多头最怕跌（取 low），空头最怕涨（取 high）。"""
        return low if units > 0 else high

    def is_liquidatable(self, symbol: str, units: float,
                        adverse_price: float) -> bool:
        """保证金权益 ≤ 维持保证金 → 强平。

        参数是**不利价**（多头传 low、空头传 high），不是固定的 high ——
        见模块 docstring 缺陷 3。已强平的腿不再被强平（防止重复触发）。
        """
        if symbol in self.liquidated or units == 0.0:
            return False
        eq = self.equity(symbol, units, adverse_price)
        return eq <= self.maintenance_required(units, adverse_price)

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
        """建仓：记下初始保证金与**建仓价**（浮动盈亏的基准）。

        不扣 cash_pool —— 主循环建仓时已按名义额全额扣现金，
        这里再扣会双重扣钱。强平/平仓的现金动作见 `liquidate`。

        被强平的 symbol 在此解除标记，允许重新开仓
        （模拟"爆仓后账户归零、agent 重新建仓"）。
        """
        if abs(units) < 1e-9:
            return 0.0
        if symbol in self.liquidated:
            self.liquidated.discard(symbol)
        required = self.initial_required(units, price)
        self.legs[symbol].margin_cash = required
        self.legs[symbol].initial_margin = required
        self.legs[symbol].entry_price = price
        return required

    def close_leg(self, symbol: str, units: float, price: float, cash_pool) -> float:
        """平仓：清零账本（含建仓价）。

        不操作 cash_pool —— 主循环的 `cash -= d_n × px` 已经处理了平仓所得/支出。
        """
        if symbol not in self.legs:
            return 0.0
        if abs(units) < 1e-9:
            return 0.0
        self.legs[symbol].margin_cash = 0.0
        self.legs[symbol].initial_margin = 0.0
        self.legs[symbol].entry_price = 0.0
        return 0.0

    def topup_to(self, symbol: str, units: float, price: float, cash_pool) -> float:
        """补保：把 margin_cash 补到 initial_margin 水平，**并从现金池真实扣款**。

        以可用现金为上限（补不出来就只能不补，交由强平处理）。
        返回实际补入的金额。主循环目前不调用它，但既然它会经 `equity()`
        影响强平判定，就不能允许它凭空增加保证金。
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
        need = min(target - eq, max(0.0, cash_pool[0]))
        if need <= 0.0:
            return 0.0
        leg.margin_cash += need
        cash_pool[0] -= need
        return need

    def liquidate(self, symbol: str, units: float, exit_price: float,
                  cash_pool) -> float:
        """强平：**在强平价上强制平仓**，与主循环的记账口径一致。

        `cash_pool[0] += units × exit_price − 罚金`：把仓位按强平价变现
        （units 带符号，多空自动处理）。平仓前后权益连续 ——
        **不退还 margin_cash**，因为那笔钱从未被单独借记过（见模块 docstring 缺陷 1）。

        参数 `exit_price` 必须是**不利价**（多头 low / 空头 high）。

        强平后：units 由调用方清零、账本清零、标记 liquidated。
        返回释放到 cash_pool 的净额。
        """
        leg = self.legs[symbol]
        penalty = self.cfg.liquidation_penalty_bp / 1e4 * abs(units) * exit_price
        realized = units * exit_price - penalty
        cash_pool[0] += realized
        leg.margin_cash = 0.0
        leg.initial_margin = 0.0
        leg.entry_price = 0.0
        self.liquidated.add(symbol)
        return realized
