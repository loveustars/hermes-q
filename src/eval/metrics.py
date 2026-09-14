"""评估指标 —— 全部按 bar 频率年化，不做任何隐式假设。"""
from __future__ import annotations

import numpy as np
import pandas as pd

BARS_PER_YEAR = {"1h": 24 * 365, "1d": 365}


def _clean(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return x[np.isfinite(x)]


def total_return(equity: np.ndarray) -> float:
    e = _clean(equity)
    return float(e[-1] / e[0] - 1.0) if len(e) > 1 else 0.0


def ann_return(equity: np.ndarray, bars_per_year: int) -> float:
    e = _clean(equity)
    if len(e) < 2 or e[0] <= 0:
        return 0.0
    years = (len(e) - 1) / bars_per_year
    if years <= 0:
        return 0.0
    return float((e[-1] / e[0]) ** (1.0 / years) - 1.0)


def ann_vol(returns: np.ndarray, bars_per_year: int) -> float:
    r = _clean(returns)
    return float(r.std(ddof=1) * np.sqrt(bars_per_year)) if len(r) > 1 else 0.0


def sharpe(returns: np.ndarray, bars_per_year: int, rf: float = 0.0) -> float:
    r = _clean(returns)
    if len(r) < 2:
        return 0.0
    sd = r.std(ddof=1)
    if sd == 0:
        return 0.0
    return float((r.mean() - rf / bars_per_year) / sd * np.sqrt(bars_per_year))


def sortino(returns: np.ndarray, bars_per_year: int) -> float:
    r = _clean(returns)
    if len(r) < 2:
        return 0.0
    downside = r[r < 0]
    dd = downside.std(ddof=1) if len(downside) > 1 else 0.0
    if dd == 0:
        return 0.0
    return float(r.mean() / dd * np.sqrt(bars_per_year))


def max_drawdown(equity: np.ndarray) -> float:
    e = _clean(equity)
    if len(e) < 2:
        return 0.0
    peak = np.maximum.accumulate(e)
    dd = (e - peak) / peak
    return float(dd.min())


def calmar(equity: np.ndarray, bars_per_year: int) -> float:
    mdd = abs(max_drawdown(equity))
    return float(ann_return(equity, bars_per_year) / mdd) if mdd > 0 else 0.0


def summary(equity: np.ndarray, turnover_notional: np.ndarray | None,
            cost_paid: np.ndarray | None, bars_per_year: int,
            initial: float | None = None) -> dict:
    """注意 initial：收益必须以**固定本金**为基数。

    用 equity[0] 做基数会出错误结论——净账本在第一根 bar 就扣了成本，
    基数变小，百分比收益反而比毛账本更高，成本效应被洗掉。
    """
    e = np.asarray(equity, dtype=float)
    base = float(initial) if initial else float(e[0])
    ret = np.diff(e) / e[:-1] if len(e) > 1 else np.array([])
    out = {
        "bars": int(len(e)),
        "initial": round(base, 2),
        "final_equity": round(float(e[-1]), 2),
        "total_return": round(float(e[-1] / base - 1.0), 6) if base > 0 else 0.0,
        "ann_return": round(ann_return(np.concatenate([[base], e]), bars_per_year), 6),
        "ann_vol": round(ann_vol(ret, bars_per_year), 6),
        "sharpe": round(sharpe(ret, bars_per_year), 4),
        "sortino": round(sortino(ret, bars_per_year), 4),
        "max_drawdown": round(max_drawdown(np.concatenate([[base], e])), 6),
    }
    out["calmar"] = round(
        out["ann_return"] / abs(out["max_drawdown"]) if out["max_drawdown"] else 0.0, 4)
    if turnover_notional is not None and len(turnover_notional):
        years = len(e) / bars_per_year
        # 往返 = 一买一卖，故除以 2
        out["turnover_roundtrips_per_year"] = round(
            float(turnover_notional.sum()) / base / 2 / years, 2)
        out["turnover_notional"] = round(float(turnover_notional.sum()), 2)
    if cost_paid is not None and len(cost_paid):
        out["total_cost"] = round(float(cost_paid.sum()), 2)
        out["cost_drag_bp_of_initial"] = round(float(cost_paid.sum() / base * 1e4), 2)
        out["cost_as_pct_of_initial"] = round(float(cost_paid.sum() / base * 100), 2)
    return out


def compare(gross: dict, net: dict) -> dict:
    """毛/净对照 —— 成本究竟吃掉了什么。用绝对金额和固定本金口径。"""
    return {
        "initial": gross.get("initial"),
        "gross_final": gross["final_equity"],
        "net_final": net["final_equity"],
        "abs_loss": round(gross["final_equity"] - net["final_equity"], 2),
        "gross_total_return": gross["total_return"],
        "net_total_return": net["total_return"],
        "return_pp_loss": round((gross["total_return"] - net["total_return"]) * 100, 2),
        # 成本吃掉了毛收益的百分之几 —— 比绝对百分点更能说明问题
        "share_of_gross_profit_eaten_pct": (
            round((gross["final_equity"] - net["final_equity"])
                  / max(gross["final_equity"] - gross["initial"], 1e-9) * 100, 2)),
        "gross_sharpe": gross["sharpe"],
        "net_sharpe": net["sharpe"],
        "sharpe_loss": round(gross["sharpe"] - net["sharpe"], 4),
        "gross_max_dd": gross["max_drawdown"],
        "net_max_dd": net["max_drawdown"],
        "cost_as_pct_of_initial": net.get("cost_as_pct_of_initial"),
        "turnover_roundtrips_per_year": net.get("turnover_roundtrips_per_year"),
    }
