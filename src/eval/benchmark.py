"""基准构造 —— 必须与策略在同一套机器、同一套收益口径下产生。

踩过的坑：早期版本把基准定义成「各资产对数收益的均值」，而策略收益是简单收益。
两者相差半个方差项（加密市场小时 σ≈1%，半个方差 ≈5e-5/根），
在 7.7 万根上累积成几千个百分点的虚假 alpha —— 连"基准对自己"都能算出
beta=1.07、alpha=+71%/年 且高度显著。

正确做法：基准就是一个**用同一撮合器、同一初始本金、零成本**跑出来的
等权买入持有组合。这样口径、起点、时间轴全部一致。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..agents.baselines import BuyHold


@dataclass
class Benchmark:
    returns: np.ndarray
    index: pd.DatetimeIndex
    equity: np.ndarray
    name: str = "equal_weight_buyhold"

    def cumulative_return(self) -> float:
        return float(self.equity[-1] / self.equity[0] - 1.0)


def equal_weight_buyhold(frames: dict[str, pd.DataFrame], cfg: dict,
                         initial_cash: float = 10_000.0,
                         warmup: int = 300) -> Benchmark:
    """等权买入持有基准，零成本，走同一个 SimExchange。"""
    from ..sim.costs import CostModel
    from ..sim.exchange import SimConfig, SimExchange

    zero_a = CostModel(enabled=False)
    zero_b = CostModel(enabled=False)
    simcfg = SimConfig(initial_cash=initial_cash, warmup=warmup,
                       latency_bars=cfg["costs"]["latency_bars"])
    res = SimExchange(frames, zero_a, zero_b, simcfg).run(BuyHold())
    eq = res.net_equity
    r = np.diff(eq) / eq[:-1]
    r = np.where(np.isfinite(r), r, 0.0)
    return Benchmark(returns=r, index=res.index[1:len(r) + 1], equity=eq)


def single_asset_buyhold(frames: dict[str, pd.DataFrame], symbol: str, cfg: dict,
                         initial_cash: float = 10_000.0,
                         warmup: int = 300) -> Benchmark:
    """单标的买入持有基准 —— 用于回答"跑不跑得赢直接拿着不动"。"""
    from ..agents.baselines import SingleAssetBuyHold
    from ..sim.costs import CostModel
    from ..sim.exchange import SimConfig, SimExchange

    zero = CostModel(enabled=False)
    simcfg = SimConfig(initial_cash=initial_cash, warmup=warmup,
                       latency_bars=cfg["costs"]["latency_bars"])
    res = SimExchange(frames, zero, zero, simcfg).run(SingleAssetBuyHold(symbol))
    eq = res.net_equity
    r = np.diff(eq) / eq[:-1]
    r = np.where(np.isfinite(r), r, 0.0)
    return Benchmark(returns=r, index=res.index[1:len(r) + 1], equity=eq,
                     name=f"buyhold_{symbol}")


def strategy_returns(res) -> tuple[np.ndarray, pd.DatetimeIndex]:
    """从 SimResult 取策略收益序列及其时间戳（供回归对齐用）。"""
    eq = res.net_equity
    r = np.diff(eq) / eq[:-1]
    r = np.where(np.isfinite(r), r, 0.0)
    return r, res.index[1:len(r) + 1]
