"""纸面交易账本 —— 用**实时公开行情**推进 carry 组合，不碰历史数据。

## 与回测器的根本区别

回测是"给定一段历史数据，从头算到尾"；这里是"给定当前状态 + 一个实时价格，
往前推一步"。所以它是**有状态、可重入、且必须幂等**的：

- 同一时刻重复调用不能把资金费算两遍（否则跑一整年就凭空多出/少掉收益）
- 状态必须落盘，进程重启后能接着走（cron 每小时起一个新进程）
- 任何一次 tick 都只能看到该时刻 **已经存在** 的数据 ⇒ 未来函数在物理上不可能

## 三桶记账（与 `src/sim/carry.py` 同一套语义）

| 桶 | 含义 |
|---|---|
| 现货腿 | 持币，按现货价计价 |
| 永续保证金账户 | 存款 + 空头持仓（按**标记价**计价，与交易所一致） |
| 备用金 | 补保来源 |

恒等式：`权益 = 备用金 + 现货腿 + 保证金存款 + 空头持仓损益`

## 符号约定（这个坑踩过两次，写死在这里）

资金费**现金流** = `−perp_units × mark_price × rate`。
空头 `perp_units < 0`，费率 `rate > 0` ⇒ 现金流为正 ⇒ **收到**。
写成 `+perp_units×mark_price×rate` 会把"收钱"算成"付钱"，方向整个颠倒。
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

LIVE_DIR = os.path.join("runs", "live")


def _utc(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000.0, tz=timezone.utc).isoformat(
        timespec="seconds")


@dataclass
class LiveConfig:
    """与 M10 网格中存活最优、且通过 G3 的配置一致。"""

    initial_capital: float = 10_000.0
    notional_ratio: float = 0.4          # 每条腿名义额 / 权益
    initial_margin_ratio: float = 0.5    # 初始保证金 / 名义额
    rebalance_every_h: int = 720         # 30 天
    maintenance_margin_ratio: float = 0.005
    topup_trigger_ratio: float = 0.5
    cost_rate: float = 0.0011            # 手续费 0.001 + 单边点差 ~0.0001


@dataclass
class Book:
    """一个标的的纸面持仓。"""

    symbol: str
    cfg: LiveConfig = field(default_factory=LiveConfig)
    created_at_ms: int = 0
    opened_at_ms: int | None = None

    spot_units: float = 0.0
    perp_units: float = 0.0          # 负数 = 空头
    margin_cash: float = 0.0
    reserve: float = 0.0

    last_funding_ms: int = 0
    last_rebalance_ms: int = 0

    funding_total: float = 0.0       # 累计收到的资金费（正=净收到）
    fees_total: float = 0.0
    n_topups: int = 0
    topup_total: float = 0.0

    liquidated: bool = False
    liquidated_at_ms: int | None = None

    snapshots: list[dict] = field(default_factory=list)
    events: list[dict] = field(default_factory=list)

    # ---------------- 计价 ----------------
    def equity(self, spot_px: float, mark_px: float) -> float:
        return (self.reserve + self.spot_units * spot_px
                + self.margin_cash + self.perp_units * mark_px)

    def margin_equity(self, mark_px: float) -> float:
        """保证金账户的可用权益（补保与强平都看它）。"""
        return self.margin_cash + self.perp_units * mark_px

    def target_margin(self, equity: float) -> float:
        return self.cfg.notional_ratio * equity * self.cfg.initial_margin_ratio

    # ---------------- 生命周期 ----------------
    def open_(self, now_ms: int, spot_px: float, mark_px: float,
              reserve_cap: float | None = None) -> None:
        """按配置开仓：现货多 + 永续空，两腿名义额相等。"""
        if self.opened_at_ms is not None:
            raise RuntimeError("已开仓，不要重复开")
        cap = self.cfg.initial_capital if reserve_cap is None else reserve_cap
        notional = cap * self.cfg.notional_ratio
        margin = notional * self.cfg.initial_margin_ratio
        self.spot_units = notional / spot_px
        self.perp_units = -(notional / mark_px)
        self.margin_cash = margin + notional          # 存款 + 卖出永续所得
        self.reserve = cap - notional - margin
        fee = notional * self.cfg.cost_rate * 2.0     # 两条腿
        self.reserve -= fee
        self.fees_total += fee
        self.opened_at_ms = now_ms
        self.created_at_ms = now_ms
        self.last_funding_ms = (now_ms // 3_600_000) * 3_600_000
        self.last_rebalance_ms = now_ms
        self._event(now_ms, "open", {
            "spot_units": self.spot_units, "perp_units": self.perp_units,
            "notional": notional, "margin": margin, "reserve": self.reserve,
            "open_fee": fee, "spot_px": spot_px, "mark_px": mark_px})

    def apply_funding(self, rows: list[dict], now_ms: int) -> list[dict]:
        """应用**尚未处理过**的资金费结算。幂等：只看 funding_time > last_funding_ms。

        现金流 = −perp_units × mark_price × rate（符号见模块 docstring）。
        """
        applied = []
        for r in sorted(rows, key=lambda x: int(x["funding_time"])):
            ft = int(r["funding_time"])
            if ft <= self.last_funding_ms:
                continue
            rate = float(r["funding_rate"])
            px = float(r.get("mark_price") or 0.0)
            if px <= 0:
                continue
            cash = -self.perp_units * px * rate
            self.margin_cash += cash
            self.funding_total += cash
            self.last_funding_ms = ft
            rec = {"at": _utc(ft), "funding_time": ft, "rate": rate,
                   "mark_price": px, "cash": cash}
            applied.append(rec)
            self._event(ft, "funding", rec)
        return applied

    def check_margin(self, now_ms: int, mark_px: float) -> dict:
        """补保或强平。强平判定用**标记价**（交易所就是这么做的）。"""
        me = self.margin_equity(mark_px)
        target = self.target_margin(self.equity(mark_px, mark_px))
        maint = self.cfg.maintenance_margin_ratio * abs(self.perp_units) * mark_px
        if me <= maint:
            self.liquidated = True
            self.liquidated_at_ms = now_ms
            self._event(now_ms, "liquidated", {"margin_equity": me,
                                               "maintenance": maint})
            return {"action": "liquidated", "margin_equity": me}
        if me < self.cfg.topup_trigger_ratio * target:
            need = target - me
            if self.reserve >= need:
                self.reserve -= need
                self.margin_cash += need
                self.n_topups += 1
                self.topup_total += need
                self._event(now_ms, "topup", {"amount": need,
                                              "margin_equity_before": me})
                return {"action": "topup", "amount": need}
            # 备用金不够补到目标，能补多少补多少（不够就会被下一次判定强平）
            if self.reserve > 0:
                amt = self.reserve
                self.reserve = 0.0
                self.margin_cash += amt
                self.n_topups += 1
                self.topup_total += amt
                self._event(now_ms, "topup_partial", {"amount": amt})
                return {"action": "topup_partial", "amount": amt}
            return {"action": "topup_failed_no_reserve"}
        return {"action": "ok", "margin_equity": me}

    def maybe_rebalance(self, now_ms: int, spot_px: float, mark_px: float) -> dict | None:
        """到期再平衡：把两腿名义额拉回相等。

        注意再平衡同时是**风控**：它按缩水后的权益重设仓位，等于自动降杠杆
        （M10 冒烟测试验证过：不再平衡会被强平，定期再平衡则存活）。
        """
        due_h = (now_ms - self.last_rebalance_ms) / 3_600_000
        if due_h < self.cfg.rebalance_every_h:
            return None
        eq = self.equity(spot_px, mark_px)
        want_notional = eq * self.cfg.notional_ratio
        d_spot = want_notional / spot_px - self.spot_units
        d_perp = -(want_notional / mark_px) - self.perp_units
        fee = (abs(d_spot) * spot_px + abs(d_perp) * mark_px) * self.cfg.cost_rate
        self.reserve -= d_spot * spot_px        # 买现货要付现金（现货腿有现金账户）
        self.margin_cash -= d_perp * mark_px
        self.reserve -= fee
        self.fees_total += fee
        self.spot_units += d_spot
        self.perp_units += d_perp
        self.last_rebalance_ms = now_ms
        rec = {"at": _utc(now_ms), "d_spot": d_spot, "d_perp": d_perp,
               "fee": fee, "equity": eq}
        self._event(now_ms, "rebalance", rec)
        return rec

    # ---------------- 快照与持久化 ----------------
    def snapshot(self, now_ms: int, spot_px: float, mark_px: float) -> dict:
        eq = self.equity(spot_px, mark_px)
        basis_bp = (mark_px / spot_px - 1.0) * 1e4
        rec = {"at": _utc(now_ms), "at_ms": now_ms, "equity": eq,
               "spot_px": spot_px, "mark_px": mark_px, "basis_bp": basis_bp,
               "margin_equity": self.margin_equity(mark_px),
               "reserve": self.reserve, "funding_total": self.funding_total,
               "fees_total": self.fees_total}
        self.snapshots.append(rec)
        return rec

    def _event(self, ms: int, kind: str, detail: Any) -> None:
        self.events.append({"at": _utc(ms), "at_ms": ms, "kind": kind,
                            "detail": detail})

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "Book":
        d = dict(d)
        d["cfg"] = LiveConfig(**d["cfg"])
        return Book(**d)

    def _trim(self, keep_hourly: int = 48) -> None:
        """裁剪快照序列：保留最近 keep_hourly 条 + 每个 UTC 日的最后一条。

        为什么必须裁：每小时一条快照，一年 8,760 条。不裁的话状态文件无界增长，
        而这份文件是要进 git 当"前向测试记录"的。裁剪后一年约 400 条（≈80KB），
        同时**保住了有意义的曲线**（每日收盘 + 最近两天的小时级细节）。
        """
        if len(self.snapshots) <= keep_hourly:
            return
        recent = self.snapshots[-keep_hourly:]
        older = self.snapshots[:-keep_hourly]
        by_day: dict[str, dict] = {}
        for s in older:                      # 按 UTC 日取最后一条
            by_day[s["at"][:10]] = s
        self.snapshots = sorted(by_day.values(), key=lambda x: x["at"]) + recent

    def save(self, path: str | None = None) -> str:
        self._trim()
        p = path or os.path.join(LIVE_DIR, f"{self.symbol}.json")
        os.makedirs(os.path.dirname(p), exist_ok=True)
        tmp = p + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=1)
        os.replace(tmp, p)        # 原子替换：宁可丢一次 tick，不要半截状态
        return p

    @staticmethod
    def load(symbol: str, path: str | None = None) -> "Book | None":
        p = path or os.path.join(LIVE_DIR, f"{symbol}.json")
        if not os.path.exists(p):
            return None
        with open(p, encoding="utf-8") as f:
            return Book.from_dict(json.load(f))
