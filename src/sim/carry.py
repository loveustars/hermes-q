"""双腿 carry 仿真器 —— 现货多头 + 永续空头，显式建模保证金与强平。

为什么必须单独写而不是复用 SimExchange：
  SimExchange 是单账户、按目标权重下单、无保证金概念。carry 的核心风险恰恰在于
  **永续腿有独立的保证金账户**：现货腿赚的钱不能自动补到永续腿（普通账户），
  所以价格最终回归、整体持仓不亏，永续腿仍可能先被强平。
  M9 已量化过：30 日内 BTC 上冲 135.5%、ETH 150.0%，足以清掉 100% 保证金以下的所有空头腿。

三个资金桶（分开记账，这是本模块的全部要点）：
  spot    现货腿：买入 N 名义额的现货，不可加杠杆
  margin  永续腿保证金账户：存入 M，持有空头名义额 N
  reserve 备用金：用于补保，补完即无 → 触发强平

账户恒等式（开仓时）：
  spot   = N
  margin = M + N（卖出永续得到的现金）− N（空头市值） = M
  total  = N + M + reserve = C
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class CarryConfig:
    initial_capital: float = 10_000.0
    notional_ratio: float = 0.5          # 每条腿的名义额 = 该比例 × 初始资本
    initial_margin_ratio: float = 1.0    # 初始保证金 = m × 名义额
    maintenance_margin_ratio: float = 0.005   # 维持保证金率
    topup_trigger_ratio: float = 0.5     # 保证金权益跌破初始保证金的该比例时补保
    rebalance_every: int = 168           # 每多少根 bar 把两腿拉回 delta 中性
    fee_rate: float = 0.001              # 单边手续费
    spread_rate: float = 0.0001          # 单边点差+冲击的近似
    latency_bars: int = 1
    warmup: int = 300

    def __post_init__(self):
        if not 0.0 < self.notional_ratio <= 1.0:
            raise ValueError("notional_ratio 必须在 (0, 1] 内")
        if self.initial_margin_ratio <= 0:
            raise ValueError("initial_margin_ratio 必须为正")
        if self.rebalance_every < 1:
            raise ValueError("rebalance_every 至少为 1")


@dataclass
class CarryResult:
    index: pd.DatetimeIndex
    equity: np.ndarray
    spot_value: np.ndarray
    margin_equity: np.ndarray
    reserve: np.ndarray
    funding_income: np.ndarray
    price_pnl: np.ndarray
    fees_paid: np.ndarray
    n_rebalances: int = 0
    n_topups: int = 0
    topup_amount: float = 0.0
    liquidated_at: int | None = None
    initial_capital: float = 10_000.0
    config: dict = field(default_factory=dict)

    @property
    def liquidated(self) -> bool:
        return self.liquidated_at is not None

    def final(self) -> float:
        return float(self.equity[-1])

    def total_return(self) -> float:
        return self.final() / self.initial_capital - 1.0

    def total_funding(self) -> float:
        return float(self.funding_income.sum())

    def total_fees(self) -> float:
        return float(self.fees_paid.sum())

    def total_price_pnl(self) -> float:
        """两腿价格变动的净损益（delta 中性下主要是基差漂移）。"""
        return float(self.price_pnl.sum())

    def years(self) -> float:
        return len(self.equity) / (24 * 365)

    def annualized(self) -> float:
        y = self.years()
        if y <= 0 or self.final() <= 0:
            return -1.0
        g = float(np.log(self.final() / self.initial_capital) / y)
        return float(np.expm1(min(g, 20.0)))


class CarrySimulator:
    """现货多 + 永续空的 delta 中性 carry 仿真。"""

    def __init__(self, spot: pd.DataFrame, perp: pd.DataFrame, funding,
                 cfg: CarryConfig, symbol: str = ""):
        idx = spot.index.intersection(perp.index)
        self.spot = spot.loc[idx]
        self.perp = perp.loc[idx]
        self.idx = idx
        self.funding = funding
        self.cfg = cfg
        self.symbol = symbol
        self.T = len(idx)

    # ------------------------------------------------------------------
    def run(self) -> CarryResult:
        cfg = self.cfg
        N = cfg.initial_capital * cfg.notional_ratio
        M = N * cfg.initial_margin_ratio
        reserve0 = cfg.initial_capital - N - M
        if reserve0 < 0:
            raise ValueError(
                f"资本不足：需要 spots {N:,.0f} + margin {M:,.0f} = "
                f"{N + M:,.0f}，但初始资本只有 {cfg.initial_capital:,.0f}。"
                "请降低 notional_ratio 或 initial_margin_ratio。")

        start = cfg.warmup
        if start >= self.T - 2:
            raise ValueError("warmup 过长")

        spot_px = self.spot["close"].to_numpy(dtype=float)
        perp_px = self.perp["close"].to_numpy(dtype=float)
        # 空头的**最不利价是最高价**（价格上冲才会亏损），强平判定必须用它，
        # 只按收盘价判定会低估强平风险。
        perp_high = self.perp["high"].to_numpy(dtype=float)

        cost_rate = cfg.fee_rate + cfg.spread_rate

        # 开仓
        N_cur = N
        u_spot = N_cur / spot_px[start]
        u_perp = -N_cur / perp_px[start]
        margin_cash = M + N_cur                        # 卖出永续收到的现金
        reserve = reserve0
        open_fee = cost_rate * N_cur * 2               # 两条腿各一次
        reserve -= open_fee
        cum_fees = open_fee

        eq_out, sp_out, mg_out, rs_out = [], [], [], []
        fund_out, price_out, fee_out = [], [], []
        n_reb = n_top = 0
        topup_total = 0.0
        liquidated_at = None
        first_bar = True

        for t in range(start, self.T):
            # 开仓手续费记在第一根 bar 上，否则总费用会漏掉这一笔
            fee_bar = open_fee if first_bar else 0.0
            first_bar = False
            # ---- 结算资金费 ----
            # 现金流符号：仓位现金流 = −(持仓名义额) × 费率。
            # 空头 u_perp<0，故 rate>0 时 f_now>0（**收取**）。
            # 写反成 u_perp*px*rate 会把"收钱"算成"付钱"，方向整个颠倒。
            ts_ms = int(self.idx[t].timestamp() * 1000)
            rate = self.funding.rate_or_none(self.symbol, ts_ms)
            f_now = 0.0
            if rate is not None and u_perp != 0.0:
                f_now = -u_perp * perp_px[t] * rate
                margin_cash += f_now

            # ---- 强平判定：用本根最高价对空头腿做压力测试 ----
            margin_eq_stress = margin_cash + u_perp * perp_high[t]
            notional_now = abs(u_perp) * perp_high[t]
            maint = cfg.maintenance_margin_ratio * notional_now
            if u_perp != 0.0 and margin_eq_stress <= maint:
                # 强平：交易所接管空头腿，保证金账户清零。
                # 注意：此时现货腿仍在，组合变成**裸多头**暴露 ——
                # 这里选择终止仿真并如实记录，因为策略设计已不成立。
                margin_cash = 0.0
                u_perp = 0.0
                liquidated_at = len(eq_out)
                s_val = u_spot * spot_px[t]
                eq_out.append(s_val + reserve)
                sp_out.append(s_val)
                mg_out.append(0.0)
                rs_out.append(reserve)
                fund_out.append(f_now)
                price_out.append(0.0)
                fee_out.append(fee_bar)
                break

            # ---- 补保 ----
            margin_eq = margin_cash + u_perp * perp_px[t]
            target_margin = cfg.initial_margin_ratio * abs(u_perp) * perp_px[t]
            if margin_eq < cfg.topup_trigger_ratio * target_margin:
                need = target_margin - margin_eq
                take = min(need, reserve)
                if take > 0:
                    margin_cash += take
                    reserve -= take
                    topup_total += take
                    n_top += 1

            # ---- 定期再平衡：把两腿名义额拉回相等 ----
            if t > start and (t - start) % cfg.rebalance_every == 0:
                total_now = (u_spot * spot_px[t] + margin_cash
                             + u_perp * perp_px[t] + reserve)
                want_spot_units = (total_now * cfg.notional_ratio) / spot_px[t]
                d_spot = want_spot_units - u_spot
                f1 = cost_rate * abs(d_spot) * spot_px[t]
                # 现货腿没有独立的现金账户，买卖的本金必须走 reserve，
                # 否则"卖出一部分现货"的钱会凭空消失，账目对不上。
                reserve -= d_spot * spot_px[t]
                u_spot = want_spot_units

                want_perp_units = -(abs(u_spot) * spot_px[t]) / perp_px[t]
                d_perp = want_perp_units - u_perp
                f2 = cost_rate * abs(d_perp) * perp_px[t]
                margin_cash -= d_perp * perp_px[t]      # 加空得到现金
                u_perp = want_perp_units

                reserve -= (f1 + f2)
                fee_bar += f1 + f2
                cum_fees += f1 + f2
                n_reb += 1

            # ---- 记账 ----
            s_val = u_spot * spot_px[t]
            margin_eq = margin_cash + u_perp * perp_px[t]
            equity = s_val + margin_eq + reserve
            prev_eq = cfg.initial_capital if not eq_out else eq_out[-1]
            eq_out.append(equity)
            sp_out.append(s_val)
            mg_out.append(margin_eq)
            rs_out.append(reserve)
            fund_out.append(f_now)
            # 非资金费损益 = 权益变动 − 资金费。对 delta 中性组合，
            # 这一项主要是基差漂移（两腿价格未完全同步的部分）加费用。
            price_out.append(equity - prev_eq - f_now)
            fee_out.append(fee_bar)

        eq = np.asarray(eq_out, dtype=float)
        return CarryResult(
            index=self.idx[start:start + len(eq)],
            equity=eq,
            spot_value=np.asarray(sp_out, dtype=float),
            margin_equity=np.asarray(mg_out, dtype=float),
            reserve=np.asarray(rs_out, dtype=float),
            funding_income=np.asarray(fund_out, dtype=float),
            price_pnl=np.asarray(price_out, dtype=float),
            fees_paid=np.asarray(fee_out, dtype=float),
            n_rebalances=n_reb, n_topups=n_top,
            topup_amount=topup_total, liquidated_at=liquidated_at,
            initial_capital=cfg.initial_capital,
            config={"notional_ratio": cfg.notional_ratio,
                    "initial_margin_ratio": cfg.initial_margin_ratio,
                    "rebalance_every": cfg.rebalance_every,
                    "maintenance_margin_ratio": cfg.maintenance_margin_ratio},
        )
