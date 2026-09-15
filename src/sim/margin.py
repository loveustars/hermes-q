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

**2026-09-15 第二次修复：参数语义（`initial_margin_ratio` 必须由杠杆推出）**

前三处修的是**记账**，这一处修的是**参数**：公式修对了，但喂给它的 `k` 是错的。

`open_leg` 里 `margin_cash = k × |units| × p₀`（`margin.py:98-100`），
而 D/C 阶段把所有脚本**硬编码成 `k = 0.5`** —— 这等于无条件假设每个仓位都是 2 倍杠杆。
推导强平阈值（`|u|` 两边约掉）：

    k|u|p₀ + |u|(p − p₀) ≤ m|u|p   ⇒   p/p₀ ≤ (1−k)/(1−m)

代入 (k=0.5, m=0.1) 得 **p/p₀ ≤ 0.5556，即跌 44.4% 就强平，且与杠杆完全无关**
（1x / 2x / 3x 都是 44.4%）。实测后果（可复现）：一条 1x、只交易一次、
买入 BTC 后不动的规则（`only_BTCUSDT`，band=0.20）在 **bar 9104 =
2018-11-23 01:00**（该根 `low = 4,239.67`，入场价 = bar 301 的 open = 7,676.8，
`low/p₀ = 0.5523 ≤ 0.5556`）被判定强平；因为 agent 是**开环**的（不看 fill、
不看强平），它此后每 bar 重发同一个目标、被 band 滤掉、**永不重入**，
曲线从索引 8803 起恒定 7.9 年：终值 5,469 vs 关掉保证金的 101,251（−94.6%），
而 BTC 同期从入场价涨到 77,770。

**为什么 `k` 必须由杠杆决定**：`SimExchange` 的现金记账是"全额扣现金买入"
（`exchange.py:251-252` 的 `cash -= d_n × px`）。1x 时现金归零、**没有任何隐性借款**，
所以*实际*杠杆由现金余额决定，不是由 `k` 决定。把 `k = 1/L`（L = 该账户的
max gross，即 `SimConfig.max_gross` / `HedgeEnsemble.max_exposure`）代回得教科书式

    p/p₀ ≤ (L−1) / (L(1−m))        L=1 ⇒ 0（**永不强平**）；L=2 ⇒ 0.5556；L=3 ⇒ 0.7407

L=1 的"永不强平"就是正确答案：满额持仓无法被强平（账上没有借款）。
**旁证**：同仓库 `src/sim/carry.py:31-32`（M10 阶段、已通过完整评估协议）
把"满额自筹"写成 `initial_margin_ratio=1.0` + `maintenance=0.005` ——
项目自己已经把这一档参数用对过一次，是 D 阶段另起炉灶时传成了 0.5。
（注意两套账的公式不同：`carry.py` 的 `margin_cash` 含卖出永续收到的现金
（`M + N`，`carry.py:139`），代入得**空头**阈值 `(1+k)/(1+m) = +98.5%`
（≈永不强平）；本模块多头的阈值是 0、空头是 `(1+k)/(1+m) = 1.818`。
两边的**意图**一致（满额自筹 ⇒ 不该有提前停损），数值不必也不该相等。）

因此**新增** `MarginConfig.for_leverage(L, ...)`：调用方的杠杆是唯一输入，
`k` 由它推出，杜绝每个脚本各写一个常数。**注意向后兼容**：`dataclass` 的默认值
与"显式传 `initial_margin_ratio=...`"的语义**都没有变**（`tests/test_margin_book.py`
的 22 条与 `tests/test_sim_conservation.py` 全部继续通过）——
本模块不猜调用方的杠杆，只提供一条不会写错的路径。

设计要点（沿用）：
  - 单桶 cash（整体账户），但 per-symbol 维护 margin_cash（逻辑子账户）用于强平判定
  - 与 carry.py 的 spot+perp 双账户不同：SimExchange 是单账户主 sim
  - 强平判定用**不利价**（多头 low / 空头 high）做压力测试
  - 强平后：该腿 units 清零（由调用方）、账本清零、标记 liquidated、允许日后重开
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class MarginConfig:
    """逐腿保证金参数。

    initial_margin_ratio: 开仓时需存入的保证金占名义额的比例（> 1 表示需要更多保证金）
    maintenance_margin_ratio: 维持保证金率（< initial_margin_ratio，强平线）
    topup_trigger_ratio: 保证金权益跌破初始保证金的该比例时，触发补保
    liquidation_penalty_bp: 强平罚金（基点），从平仓所得中扣除（0 表示不罚）

    **不要手写 `initial_margin_ratio`**：它必须与调用方的实际杠杆一致，
    否则会凭空给每条腿加一个与杠杆无关的停损（见模块 docstring 的第二次修复）。
    用 `MarginConfig.for_leverage(L)`。

    默认值 0.5 / 0.25 是历史值（= 2x 杠杆），保留只为向后兼容既有单测；
    新代码请一律用类方法构造。
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

    # ------------------------------------------------------------------
    @classmethod
    def for_leverage(cls, leverage: float, *,
                     maintenance_margin_ratio: float = 0.1,
                     topup_trigger_ratio: float = 0.5,
                     liquidation_penalty_bp: float = 0.0) -> "MarginConfig":
        """由**实际杠杆**推出初始保证金比例：`k = 1 / leverage`。

        参数
        ----
        leverage: 该账户的最大总敞口（gross），即 `SimConfig.max_gross` /
            `HedgeEnsemble.max_exposure`。1.0 = 满额、无借款。

        为什么是 `1/leverage`
        -------------------
        `SimExchange` 建仓时**全额扣现金**（`cash -= notional`），所以 1x
        持仓的账户里没有借款，不该有任何强平通道。把 `k = 1/L` 代入阈值式

            p/p₀ ≤ (1−k)/(1−m)

        得到教科书形式 `p/p₀ ≤ (L−1)/(L(1−m))`：

            L=1 ⇒ 0      （**永不强平** —— 满额持仓无法被强平）
            L=2 ⇒ 0.5556 （跌 44.4%）
            L=3 ⇒ 0.7407 （跌 25.9%）

        旁证：`src/sim/carry.py:31-32` 用的就是 `initial_margin_ratio=1.0`
        （M10 阶段、已通过完整评估协议）—— 项目已把"满额自筹"这一档用对过，
        C/D 阶段只是另起炉灶时传成了 0.5。（两套账公式不同：carry 的
        `margin_cash` 含卖出永续收到的现金，阈值是 `(1+k)/(1+m)`；
        **意图**一致，数值不必相等。）

        注意 `L` 与 `m` 的关系：`k = 1/L < m` ⇔ `L > 1/m`，此时仓位在**开仓
        那一刻**就满足强平条件（p/p₀ = 1 ≤ (1−k)/(1−m) 成立）。例如 m=0.1
        时 L > 10 会被拒绝 —— 这不是"更严格的风控"，而是一个无解的参数组合。
        """
        if not math.isfinite(leverage) or leverage <= 0.0:
            raise ValueError(f"leverage 必须为正的有限数，收到 {leverage!r}")
        k = 1.0 / float(leverage)
        if k < maintenance_margin_ratio:
            raise ValueError(
                f"leverage={leverage:g} ⇒ initial_margin_ratio={k:.4f} < "
                f"maintenance_margin_ratio={maintenance_margin_ratio:g}；"
                f"该组合下仓位在开仓瞬间即满足强平条件。"
                f"请用 leverage ≤ {1.0 / maintenance_margin_ratio:g}")
        return cls(initial_margin_ratio=k,
                   maintenance_margin_ratio=maintenance_margin_ratio,
                   topup_trigger_ratio=topup_trigger_ratio,
                   liquidation_penalty_bp=liquidation_penalty_bp)

    # ------------------------------------------------------------------
    def long_liquidation_ratio(self) -> float:
        """多头强平阈值 `p/p₀ = (1−k)/(1−m)`；返回 **0.0 表示永不强平**。

        解析式由 `equity ≤ maintenance` 两边约掉 `|u|` 得到（见模块 docstring）：
        与持仓量、与名义规模无关。1x（k=1）时分子为 0 ⇒ 价格必须 ≤ 0 才触发，
        即不可能触发。
        """
        num = 1.0 - self.initial_margin_ratio
        if num <= 0.0:
            return 0.0
        return num / (1.0 - self.maintenance_margin_ratio)

    def short_liquidation_ratio(self) -> float:
        """空头强平阈值 `p/p₀ = (1+k)/(1+m)`（k=1/3, m=0.1 ⇒ 1.2121）。"""
        return (1.0 + self.initial_margin_ratio) / (1.0 + self.maintenance_margin_ratio)

    def liquidation_description(self) -> str:
        """一行说明这套参数的强平距离 —— 便于把参数打进 run 的 config。"""
        lo = self.long_liquidation_ratio()
        return (f"k={self.initial_margin_ratio:.4f} m={self.maintenance_margin_ratio:g} ⇒ "
                f"多头 {'永不强平' if lo <= 0 else f'p/p0≤{lo:.4f}（跌 {(1-lo)*100:.1f}%）'}"
                f" / 空头 p/p0≥{self.short_liquidation_ratio():.4f}")


@dataclass
class MarginLeg:
    """单条腿的保证金账户（逻辑子账户）。

    margin_cash: 保证金账户现金
    initial_margin: 开仓时存入的初始保证金（用于 topup 触发判断）
    entry_price: 建仓价 —— **浮动盈亏的基准**（见缺陷 2）。为 0 表示无持仓。
    units_at_anchor: **锚定时的持仓量**（2026-09-15 加）。仅用于检测
        "仓位被调整但锚点没跟着刷新"（见 `n_stale_anchor_bars`），
        不参与任何金额计算 ⇒ 加上它不改变任何既有行为。
    """
    margin_cash: float = 0.0
    initial_margin: float = 0.0
    entry_price: float = 0.0
    units_at_anchor: float = 0.0


class MarginBook:
    """per-symbol 保证金账本（per-leg MarginLeg 的容器）。"""

    def __init__(self, cfg: MarginConfig, symbols: list[str]):
        self.cfg = cfg
        self.symbols = list(symbols)
        self.legs: dict[str, MarginLeg] = {s: MarginLeg() for s in symbols}
        self.liquidated: set[str] = set()        # 已强平的标的（一次性记录）
        # 诊断计数器（2026-09-15 加，不改变任何金额）：
        # 主循环只在"空→有仓 / 有仓→空 / 多空翻转"时同步保证金账本
        # （`exchange.py:258-269`）。**仓位被调大或调小却不穿过零**时，
        # `margin_cash`/`entry_price` 仍是旧仓位的锚 ⇒ 权益公式
        # `margin_cash + (p − p₀)·|u_现|` 不再是 `|u_现|·p`，
        # 于是连 k=1（本该永不强平）都会凭空触发强平。实测案例：`inv_vol`
        # 这条**只做多**的规则在 k=1 下被强平，终值 23,400 vs 关保证金的 29,516。
        # 修它要改 `exchange.py` 的同步路径（本次不在范围内），这里先把
        # "有多少根 bar 处在锚点过期状态"变成可见的数字。
        self.n_stale_anchor_bars = 0
        self.stale_anchor_symbols: set[str] = set()
        # 强平时该腿正处于"锚点过期"状态的记录（纯诊断，见 `liquidate`）。
        # 用途：把"空头腿的正常强平"与"锚点过期造成的假强平"分开数。
        self.liquidated_stale: list[str] = []

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

        顺带（纯诊断，不影响返回值）统计"锚点过期"的 bar 数：传入的 `units`
        与锚定时记录的 `units_at_anchor` 不一致 ⇒ 仓位被调整过，但
        `margin_cash`/`entry_price` 仍锚在旧仓位上（`exchange.py` 只在
        穿越零点时同步）。
        """
        leg = self.legs[symbol]
        if abs(units) > 1e-9 and abs(units - leg.units_at_anchor) > 1e-9:
            self.n_stale_anchor_bars += 1
            self.stale_anchor_symbols.add(symbol)
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
        self.legs[symbol].units_at_anchor = units      # 诊断用（见 n_stale_anchor_bars）
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
        self.legs[symbol].units_at_anchor = 0.0
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

        顺带记录**强平发生时该腿是否处于"锚点过期"状态**（纯诊断）：若是，则
        这次强平的触发线比"锚点刷新"时更近，可能与真实经济含义无关
        （见 `n_stale_anchor_bars` 与 `liquidated_stale`）。
        """
        leg = self.legs[symbol]
        if abs(units) > 1e-9 and abs(units - leg.units_at_anchor) > 1e-9:
            self.liquidated_stale.append(symbol)
        penalty = self.cfg.liquidation_penalty_bp / 1e4 * abs(units) * exit_price
        realized = units * exit_price - penalty
        cash_pool[0] += realized
        leg.margin_cash = 0.0
        leg.initial_margin = 0.0
        leg.entry_price = 0.0
        leg.units_at_anchor = 0.0
        self.liquidated.add(symbol)
        return realized