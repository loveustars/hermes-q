"""M1 数据层：三源接入、全量落盘、交叉校验、版本哈希。

验收标准：
  - BTCUSDT / ETHUSDT / BNBUSDT 全量 1h + 1d 落盘
  - manifest 带 sha256
  - 实测点差写入 configs/spreads.json
  - 三源近期价格交叉校验报告

可断点续传：已落盘的数据只补增量。
只用公开只读行情端点。
"""
from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd  # noqa: E402

from src import config as cfgmod  # noqa: E402
from src.data import sources, store  # noqa: E402
from src.registry.runs import Run  # noqa: E402

START_MS = int(datetime(2017, 8, 1, tzinfo=timezone.utc).timestamp() * 1000)
WORKERS = 6


def _raw_path(symbol: str, interval: str, source: str = "binance") -> str:
    return os.path.join(store.data_dir("raw"), f"{source}_{symbol}_{interval}.csv")


def existing_last_ms(symbol: str, interval: str) -> int | None:
    p = _raw_path(symbol, interval)
    if not os.path.exists(p):
        return None
    df = pd.read_csv(p, usecols=["open_time"])
    return int(df["open_time"].max()) if len(df) else None


def ingest_symbol(symbol: str, interval: str, workers: int = WORKERS) -> dict:
    t0 = time.time()
    step = sources.INTERVAL_MS[interval]
    last = existing_last_ms(symbol, interval)
    start = START_MS if last is None else last + step

    bars = sources.binance_klines_parallel(
        symbol, interval=interval, start_ms=start, workers=workers)
    new_rows = len(bars)

    path = _raw_path(symbol, interval)
    if last is not None and os.path.exists(path):
        old = pd.read_csv(path)
        if new_rows:
            fresh = pd.DataFrame([b.__dict__ for b in bars])
            merged = (pd.concat([old, fresh], ignore_index=True)
                        .drop_duplicates("open_time").sort_values("open_time"))
        else:
            merged = old
        merged.to_csv(path, index=False)
        store.update_manifest(path, symbol, interval, "binance", len(merged))
    else:
        if not new_rows:
            return {"symbol": symbol, "interval": interval, "rows": 0, "error": "空数据"}
        store.save_bars(bars, symbol, interval, "binance")

    df = store.load_bars(symbol, interval)
    h = store.health(df)
    h.update(symbol=symbol, interval=interval, new_rows=new_rows,
             resumed_from=last, fetch_seconds=round(time.time() - t0, 1),
             path=os.path.relpath(path, store.project_root()))
    return h


def cross_validate(symbol: str, limit_hours: int = 168, interval: str = "1h") -> dict:
    """用 OKX / Coinbase 的近期数据校验币安价格，三源互查。"""
    base = symbol.replace("USDT", "")
    primary = store.load_bars(symbol, interval).tail(limit_hours)
    out = {"symbol": symbol, "bars_compared": int(len(primary)),
           "okx": None, "coinbase": None}
    s_pri = pd.Series(primary["close"].values, index=primary["open_time"].values)

    try:
        okx = sources.okx_klines(f"{base}-USDT", bar="1H", limit=100)
        s2 = pd.Series({b.open_time: b.close for b in okx})
        j = s_pri.to_frame("pri").join(s2.to_frame("alt"), how="inner").dropna()
        if len(j):
            dev = (j["pri"] - j["alt"]).abs() / j["alt"] * 1e4
            out["okx"] = {"n": int(len(j)), "median_bp": round(float(dev.median()), 3),
                          "p99_bp": round(float(dev.quantile(0.99)), 3),
                          "max_bp": round(float(dev.max()), 3)}
    except Exception as e:
        out["okx"] = {"error": str(e)}

    try:
        cb = sources.coinbase_klines(f"{base}-USD", granularity=3600)
        s3 = pd.Series({b.open_time: b.close for b in cb})
        j = s_pri.to_frame("pri").join(s3.to_frame("alt"), how="inner").dropna()
        if len(j):
            dev = (j["pri"] - j["alt"]).abs() / j["alt"] * 1e4
            out["coinbase"] = {"n": int(len(j)), "median_bp": round(float(dev.median()), 3),
                               "p99_bp": round(float(dev.quantile(0.99)), 3),
                               "max_bp": round(float(dev.max()), 3)}
    except Exception as e:
        out["coinbase"] = {"error": str(e)}
    return out


def main() -> None:
    cfg = cfgmod.load("base")
    cfgmod.guard_frozen(cfg)
    core = cfg["universe"]["core"]
    reserve = cfg["universe"]["reserve"]

    hypothesis = {
        "question": "三个数据源在同一时间点的价格是否一致？数据层是否足以支撑后续回测？",
        "expected": "以币安为基准，OKX 偏差中位数 < 5bp；Coinbase 因 USD 计价存在基差但 < 30bp",
        "decision_rule": "若某源偏差中位数 > 50bp，则该源不用于校验并记录原因",
    }

    with Run("m1_ingest", cfg, hypothesis) as run:
        print(f"数据采集开始  run={run.id}")
        report = {"hourly": [], "daily": [], "spread_bp": {}, "cross_validation": []}

        for sym in core:
            print(f"  [{sym}] 1h 全量 ...", flush=True)
            r = ingest_symbol(sym, "1h")
            report["hourly"].append(r)
            print(f"      {r['rows']:,} 根 (新增 {r['new_rows']:,})  "
                  f"{r.get('start')} ~ {r.get('end')}  缺口={r.get('gaps')}  "
                  f"{r.get('fetch_seconds')}s", flush=True)

        for sym in core + reserve:
            print(f"  [{sym}] 1d 全量 ...", flush=True)
            r = ingest_symbol(sym, "1d")
            report["daily"].append(r)
            print(f"      {r['rows']:,} 根 (新增 {r['new_rows']:,})  "
                  f"{r.get('start')} ~ {r.get('end')}  缺口={r.get('gaps')}  "
                  f"{r.get('fetch_seconds')}s", flush=True)

        print("\n  实测点差（真实订单簿，单边成本）", flush=True)
        for sym in core + reserve:
            try:
                bp = sources.measure_spread_bp(sym)
                report["spread_bp"][sym] = round(bp, 4)
                print(f"    {sym:<10} {bp:.4f} bp", flush=True)
            except Exception as e:
                report["spread_bp"][sym] = None
                print(f"    {sym:<10} 失败: {e}", flush=True)

        print("\n  三源交叉校验（近 168 小时）", flush=True)
        for sym in core:
            cv = cross_validate(sym, 168, "1h")
            report["cross_validation"].append(cv)
            print(f"    {sym:<10} OKX={cv['okx']}", flush=True)
            print(f"    {'':<10} Coinbase={cv['coinbase']}", flush=True)

        sp_path = os.path.join(store.project_root(), "configs", "spreads.json")
        with open(sp_path, "w", encoding="utf-8") as f:
            json.dump({"measured_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                       "unit": "bp_single_side",
                       "source": "binance_orderbook_top5",
                       "spread_bp": report["spread_bp"]}, f, ensure_ascii=False, indent=2)

        run.log("ingest_report", report)
        bad = store.verify_manifest()
        metrics = {
            "hourly_rows": {r["symbol"]: r["rows"] for r in report["hourly"]},
            "daily_rows": {r["symbol"]: r["rows"] for r in report["daily"]},
            "total_hourly_rows": sum(r["rows"] for r in report["hourly"]),
            "manifest_integrity_ok": len(bad) == 0,
            "manifest_problems": bad,
        }
        run.record_metrics(metrics)
        run.note(f"点差写入 {os.path.relpath(sp_path, store.project_root())}")

        print("\n--- M1 验收 ---")
        print(f"  manifest 校验: {'通过' if not bad else bad}")
        print(f"  1h 总行数: {metrics['total_hourly_rows']:,}")
        print(f"  run 目录: {os.path.relpath(run.dir, store.project_root())}")


if __name__ == "__main__":
    main()
