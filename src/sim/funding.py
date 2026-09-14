"""资金费模型 —— 永续持仓的持有成本。

机制：永续合约每 8 小时结算一次资金费（币安为 00:00 / 08:00 / 16:00 UTC）。
  正费率：多头付给空头；负费率：空头付给多头。
  单次成本 = 持仓名义额 × 费率
所以多头在正费率环境下是持续失血的，年化量级：
  实测 BTC 最近 5 次结算为 0.0048%~0.0080%/8h → 年化约 5%~8%。

实现要点：
  1. 币安的 fundingTime 不落在整点上（例如 ...0002 毫秒），所以按小时向下取整后再匹配 bar。
  2. 成本对多头和空头用同一个公式：cash -= units × price × rate。
     units > 0 时为付出，units < 0 时为收到，符号自动正确。
  3. 没有资金费数据的时段按 0 处理，并记录覆盖率 —— 覆盖率低时结论不可信。
"""
from __future__ import annotations

import csv
import os
from dataclasses import dataclass, field

import numpy as np

HOUR_MS = 3_600_000
FUNDING_INTERVAL_MS = 8 * HOUR_MS


def floor_hour(ms: int) -> int:
    return (ms // HOUR_MS) * HOUR_MS


@dataclass
class FundingTable:
    """按标的存放「小时时间戳 → 资金费率」。

    rates_by_hour[symbol] 的 key 是结算时刻向下取整到小时后的毫秒时间戳。
    """

    rates_by_hour: dict[str, dict[int, float]] = field(default_factory=dict)
    source: str = "binance_futures:/fapi/v1/fundingRate"

    def symbols(self) -> list[str]:
        return list(self.rates_by_hour)

    def rate_at(self, symbol: str, ts_ms: int) -> float:
        """该小时是否有结算；有则返回费率，没有返回 None 语义的 0.0。

        用 rate_or_none 可以区分「没结算」与「结算费率为 0」。
        """
        r = self.rate_or_none(symbol, ts_ms)
        return 0.0 if r is None else r

    def rate_or_none(self, symbol: str, ts_ms: int) -> float | None:
        return self.rates_by_hour.get(symbol, {}).get(floor_hour(ts_ms))

    def is_settlement(self, ts_ms: int) -> bool:
        return any(floor_hour(ts_ms) in d for d in self.rates_by_hour.values())

    def cost(self, symbol: str, units: float, price: float, ts_ms: int) -> float:
        """返回应支付的资金费（正数=付出，负数=收到）。"""
        rate = self.rate_or_none(symbol, ts_ms)
        if rate is None:
            return 0.0
        return units * price * rate

    # ---------------- 统计 ----------------
    def coverage(self, symbol: str, index, step_hours: int = 1) -> dict:
        """覆盖率：bar 序列里有多少个 8 小时边界落在有数据的区间内。"""
        if symbol not in self.rates_by_hour or len(index) == 0:
            return {"expected": 0, "found": 0, "coverage": 0.0}
        d = self.rates_by_hour[symbol]
        lo, hi = int(index[0].timestamp() * 1000), int(index[-1].timestamp() * 1000)
        expected = len(range(floor_hour(lo), floor_hour(hi) + 1, FUNDING_INTERVAL_MS))
        found = sum(1 for k in d if lo <= k <= hi)
        return {"expected": expected, "found": found,
                "coverage": round(found / expected, 4) if expected else 0.0}

    def summary(self, symbol: str) -> dict:
        d = self.rates_by_hour.get(symbol, {})
        if not d:
            return {"n": 0}
        r = np.array(list(d.values()), dtype=float)
        ann = float(r.mean()) * 3 * 365
        return {
            "n": len(r),
            "mean_per_settlement": round(float(r.mean()), 8),
            "annualized_mean": round(ann, 6),
            "median_annualized": round(float(np.median(r)) * 3 * 365, 6),
            "positive_share": round(float((r > 0).mean()), 4),
            "max": round(float(r.max()), 8),
            "min": round(float(r.min()), 8),
        }

    # ---------------- 存取 ----------------
    def add(self, symbol: str, funding_time_ms: int, rate: float) -> None:
        self.rates_by_hour.setdefault(symbol, {})[floor_hour(funding_time_ms)] = rate

    def save(self, path: str) -> str:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["symbol", "hour_ms", "funding_rate"])
            for s, d in sorted(self.rates_by_hour.items()):
                for k in sorted(d):
                    w.writerow([s, k, d[k]])
        return path

    @classmethod
    def load(cls, path: str) -> "FundingTable":
        t = cls()
        if not os.path.exists(path):
            return t
        with open(path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                t.rates_by_hour.setdefault(row["symbol"], {})[
                    int(row["hour_ms"])] = float(row["funding_rate"])
        return t


def funding_path(*parts: str) -> str:
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    p = os.path.join(root, "data", *parts)
    return p


def load_default() -> FundingTable:
    return FundingTable.load(funding_path("funding.csv"))
