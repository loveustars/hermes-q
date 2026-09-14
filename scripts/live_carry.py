"""纸面交易 CLI —— 初始化 / 推进一次 / 查看状态。

用法：
    python3 scripts/live_carry.py --init     # 用当前实时价开仓，建基线
    python3 scripts/live_carry.py --tick     # 推进一次（cron 每小时调这个）
    python3 scripts/live_carry.py --status   # 只看状态，不推进

**只用实时数据**：本脚本不 import 历史数据层（`src.data.store`），
决策只依据 tick 时刻已存在的行情。未来函数在物理上不可能存在。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data import sources  # noqa: E402
from src.live.paper import LIVE_DIR, Book, LiveConfig  # noqa: E402

SYMBOLS = ["BTCUSDT", "ETHUSDT"]      # G3 判定为「有边际」的两个
LOG = os.path.join(LIVE_DIR, "ticks.jsonl")
HOUR_MS = 3_600_000


def _now_utc() -> str:
    """统一的 UTC 时间标签。

    早期版本用 time.strftime 却标成 UTC —— 那其实打印的是**本地时间**
    （本机 CST = UTC+8，差 8 小时）。资金费在 UTC 00:00/08:00/16:00 结算，
    标签错了就没法对账，所以这里一律走 datetime.now(timezone.utc)。
    """
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def fetch_live(symbol: str) -> dict:
    """取该时刻的实时行情。失败则抛错——宁可整次 tick 跳过，也不要写半截状态。"""
    spot = float(sources._get("binance", sources.BINANCE_BASE,   # noqa: SLF001
                              "/api/v3/ticker/price", {"symbol": symbol})["price"])
    perp = float(sources._get("binance_futures",                 # noqa: SLF001
                              sources.BINANCE_FUTURES_BASE,
                              "/fapi/v1/ticker/price", {"symbol": symbol})["price"])
    pi = sources.premium_index(symbol)
    now_ms = int(time.time() * 1000)
    return {"symbol": symbol, "spot_px": spot, "perp_px": perp,
            "mark_px": float(pi["markPrice"]),
            "last_funding_rate": float(pi.get("lastFundingRate") or 0.0),
            "now_ms": now_ms}


def funding_since(symbol: str, since_ms: int, now_ms: int) -> list[dict]:
    if now_ms <= since_ms:
        return []
    rows = sources.funding_rate_history(symbol, start_ms=since_ms + 1,
                                        end_ms=now_ms)
    return rows


def fmt_pct(x: float, nd: int = 3) -> str:
    return f"{x * 100:+.{nd}f}%"


def init_books() -> int:
    if os.path.exists(os.path.join(LIVE_DIR, "BTCUSDT.json")):
        print("已存在持仓状态；如需重来请先删除 runs/live/*.json（本脚本不覆盖）")
        return 1
    print(f"=== 初始化纸面持仓（实时价开仓）{_now_utc()} ===")
    for s in SYMBOLS:
        q = fetch_live(s)
        b = Book(symbol=s, cfg=LiveConfig())
        b.open_(q["now_ms"], q["spot_px"], q["mark_px"])
        snap = b.snapshot(q["now_ms"], q["spot_px"], q["mark_px"])
        p = b.save()
        print(f"  {s:<9} 现货 {q['spot_px']:>12,.2f}  标记 {q['mark_px']:>12,.2f}  "
              f"基差 {(q['mark_px']/q['spot_px']-1)*1e4:>+7.2f}bp  "
              f"费率 {q['last_funding_rate']*100:>+7.4f}%/8h")
        print(f"            现货腿 {b.spot_units:.8f}  永续腿 {b.perp_units:+.8f}  "
              f"保证金 {b.margin_cash:,.2f}  备用金 {b.reserve:,.2f}  "
              f"权益 {snap['equity']:,.2f}  → {p}")
    return 0


def tick(verbose: bool = True) -> int:
    now_str = _now_utc()
    lines, recs = [], []
    n_ok = n_books = 0
    for s in SYMBOLS:
        b = Book.load(s)
        if b is None:
            lines.append(f"  {s:<9} 无持仓状态（先跑 --init）")
            continue
        n_books += 1
        try:
            q = fetch_live(s)
        except Exception as e:                       # 网络问题：跳过，不动状态
            lines.append(f"  {s:<9} 行情获取失败，本次跳过：{type(e).__name__}: {e}")
            continue
        n_ok += 1

        if b.liquidated:
            snap = b.snapshot(q["now_ms"], q["spot_px"], q["mark_px"])
            b.save()
            lines.append(f"  {s:<9} **已强平**（{b.liquidated_at_ms}）"
                         f"  权益 {snap['equity']:,.2f}")
            continue

        fund = b.apply_funding(funding_since(s, b.last_funding_ms, q["now_ms"]),
                               q["now_ms"])
        mg = b.check_margin(q["now_ms"], q["mark_px"])
        reb = b.maybe_rebalance(q["now_ms"], q["spot_px"], q["mark_px"])
        snap = b.snapshot(q["now_ms"], q["spot_px"], q["mark_px"])

        b.save()
        recs.append({"symbol": s, "at": snap["at"], "equity": snap["equity"],
                     "funding_applied": fund, "margin_action": mg,
                     "rebalanced": reb, "basis_bp": snap["basis_bp"]})

        ret = snap["equity"] / b.cfg.initial_capital - 1.0
        tag = ""
        if fund:
            tag += f" 资金费{fmt_pct(sum(f['cash'] for f in fund) / b.cfg.initial_capital, 4)}×{len(fund)}"
        if mg["action"] != "ok":
            tag += f" 保证金:{mg['action']}"
        if reb:
            tag += " **再平衡**"
        lines.append(
            f"  {s:<9} 权益 {snap['equity']:>10,.2f} ({fmt_pct(ret)})"
            f"  基差 {snap['basis_bp']:>+7.2f}bp"
            f"  累计资金费 {b.funding_total:>+9,.2f}{tag}")

    failed = (n_books > 0 and n_ok == 0)
    rec = {"tick_at": now_str, "books": recs}
    if failed:
        # 一笔行情都没取到 ⇒ 记成**显式失败**，并在日志与退出码上反映出来。
        # 早期版本会安静地写一条 {"books": []}，看起来像"这次没事发生"，
        # 实则是数据源断了 —— 2026-09-14 23:00 那次就是这样被掩盖的。
        rec["error"] = "全部标的行情获取失败（数据源/代理不可达）"
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    if verbose:
        print(f"carry 纸面交易  {now_str}")
        print("\n".join(lines) if lines else "  （无标的）")
        if failed:
            print(f"  ** 本次 tick 失败：{n_books} 个持仓全部取不到行情 **")
            print("     检查代理是否可用（本机 DNS 屏蔽 binance.com，必须走代理）")
    return 1 if failed else 0


def status() -> int:
    print(f"=== carry 纸面交易状态  {_now_utc()} ===")
    total_init, total_eq = 0.0, 0.0
    for s in SYMBOLS:
        b = Book.load(s)
        if b is None:
            print(f"  {s:<9} 无状态")
            continue
        try:
            q = fetch_live(s)
            eq = b.equity(q["spot_px"], q["mark_px"])
            extra = (f"  实时 现货 {q['spot_px']:,.2f} 标记 {q['mark_px']:,.2f} "
                     f"基差 {(q['mark_px']/q['spot_px']-1)*1e4:+.2f}bp")
        except Exception as e:
            eq = b.snapshots[-1]["equity"] if b.snapshots else 0.0
            extra = f"  （行情不可达：{type(e).__name__}）"
        total_init += b.cfg.initial_capital
        total_eq += eq
        days = ((b.snapshots[-1]["at_ms"] - b.opened_at_ms) / 86_400_000
                if b.snapshots and b.opened_at_ms else 0.0)
        print(f"  {s:<9} 权益 {eq:>10,.2f}  收益 {fmt_pct(eq/b.cfg.initial_capital-1)}"
              f"  已跑 {days:>5.2f} 天  快照 {len(b.snapshots):>4} 条"
              f"  资金费 {b.funding_total:>+9,.2f}  费用 {b.fees_total:>8,.2f}"
              f"  补保 {b.n_topups} 次")
        print(f"            {extra}")
        if b.liquidated:
            print(f"            ** 已强平 **")
    if total_init:
        print(f"  {'合计':<9} 权益 {total_eq:>10,.2f}  "
              f"收益 {fmt_pct(total_eq/total_init-1)}  （初始 {total_init:,.0f}）")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--init", action="store_true")
    g.add_argument("--tick", action="store_true")
    g.add_argument("--status", action="store_true")
    a = ap.parse_args()
    if a.init:
        return init_books()
    if a.tick:
        return tick()
    return status()


if __name__ == "__main__":
    raise SystemExit(main())
