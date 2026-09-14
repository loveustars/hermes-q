"""数据源适配器 —— 只读公开行情，不接任何账号、不存任何 API key。

铁律落实：本模块只允许出现 GET 行情端点。任何下单/账户端点都被
ALLOWED_PATHS 白名单挡在门外。
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any

UA = "Mozilla/5.0 (compatible; hermes-q/0.1; research; read-only)"
DAY_MS = 86_400_000
INTERVAL_MS = {"1h": 3_600_000, "1d": DAY_MS}

# 合规白名单：只读行情端点。任何不在此列的网络请求都会被拒绝。
ALLOWED_PATHS = {
    "binance": ("/api/v3/klines", "/api/v3/depth", "/api/v3/ticker/24hr"),
    # 永续（U 本位）公开行情。仍然只有只读市场数据，不含任何账户/下单端点。
    "binance_futures": ("/fapi/v1/fundingRate", "/fapi/v1/premiumIndex",
                        "/fapi/v1/klines", "/fapi/v1/ticker/24hr"),
    "okx": ("/api/v5/market/candles", "/api/v5/market/history-candles", "/api/v5/market/books"),
    "coinbase": ("/products/",),
}


class ComplianceError(RuntimeError):
    """试图访问非只读行情端点。"""


@dataclass(frozen=True)
class Bar:
    open_time: int
    open: float
    high: float
    low: float
    close: float
    volume: float
    quote_volume: float


def _get(source: str, base: str, path: str, params: dict[str, Any], timeout: int = 30) -> Any:
    if not any(path.startswith(p) for p in ALLOWED_PATHS[source]):
        raise ComplianceError(
            f"{source}{path} 不在只读白名单内，拒绝请求。"
            f"允许: {ALLOWED_PATHS[source]}"
        )
    url = base + path
    if params:
        url += "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "application/json"})
    last_err: Exception | None = None
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read().decode())
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:  # 退避重试
            last_err = e
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"GET {url} 失败: {last_err}")


# --------------------------------------------------------------------------
# 币安（主数据源）
# --------------------------------------------------------------------------
BINANCE_BASE = "https://api.binance.com"


def binance_klines(symbol: str, interval: str = "1h", start_ms: int | None = None,
                   end_ms: int | None = None, limit: int = 1000,
                   pause: float = 0.25) -> list[Bar]:
    """分页拉取全部历史 K 线（币安单次上限 1000 根）。"""
    step = INTERVAL_MS[interval]
    cursor = start_ms
    out: list[Bar] = []
    while True:
        params = {"symbol": symbol, "interval": interval, "limit": limit}
        if cursor is not None:
            params["startTime"] = cursor
        if end_ms is not None:
            params["endTime"] = end_ms
        batch = _get("binance", BINANCE_BASE, "/api/v3/klines", params)
        if not batch:
            break
        for k in batch:
            out.append(Bar(int(k[0]), float(k[1]), float(k[2]), float(k[3]),
                           float(k[4]), float(k[5]), float(k[7])))
        if len(batch) < limit:
            break
        cursor = int(batch[-1][0]) + step
        if end_ms is not None and cursor > end_ms:
            break
        time.sleep(pause)
    return out


def binance_klines_parallel(symbol: str, interval: str = "1h",
                            start_ms: int | None = None, end_ms: int | None = None,
                            workers: int = 6, pause: float = 0.05) -> list[Bar]:
    """分块并行拉取全部历史 K 线。

    币安单次上限 1000 根，串行拉 8 年小时线要 80 次请求、约 7 分钟。
    这里把时间轴切成 1000 根的块并行取，权重远低于 6000/分钟的限制。
    """
    from concurrent.futures import ThreadPoolExecutor

    step = INTERVAL_MS[interval]

    if end_ms is None:
        end_ms = int(time.time() * 1000)
    if start_ms is None:
        start_ms = end_ms - step * 1000

    starts = list(range(start_ms, end_ms, step * 1000))

    def fetch_one(s: int) -> list[Bar]:
        params = {"symbol": symbol, "interval": interval,
                  "startTime": s, "limit": 1000, "endTime": end_ms}
        batch = _get("binance", BINANCE_BASE, "/api/v3/klines", params)
        if pause:
            time.sleep(pause)
        return [Bar(int(k[0]), float(k[1]), float(k[2]), float(k[3]),
                    float(k[4]), float(k[5]), float(k[7])) for k in batch]

    out: list[Bar] = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for chunk in ex.map(fetch_one, starts):
            out.extend(chunk)

    seen: dict[int, Bar] = {}
    for b in out:
        seen[b.open_time] = b
    return [seen[k] for k in sorted(seen)]


def binance_depth(symbol: str, limit: int = 5) -> dict[str, Any]:
    """真实盘口，用于测量点差（只读）。"""
    return _get("binance", BINANCE_BASE, "/api/v3/depth", {"symbol": symbol, "limit": limit})


def measure_spread_bp(symbol: str, limit: int = 5) -> float:
    """实测半价差（基点）。基于真实订单簿，不猜。"""
    d = binance_depth(symbol, limit)
    bid = float(d["bids"][0][0])
    ask = float(d["asks"][0][0])
    mid = (bid + ask) / 2.0
    return (ask - bid) / mid / 2.0 * 1e4   # 单边成本 = 半价差


# --------------------------------------------------------------------------
# 币安永续（U 本位）—— 资金费与永续行情
# --------------------------------------------------------------------------
BINANCE_FUTURES_BASE = "https://fapi.binance.com"
FUNDING_MS = 8 * 3_600_000          # 资金费每 8 小时结算一次


def funding_rate_history(symbol: str, start_ms: int, end_ms: int | None = None,
                         pause: float = 0.15, workers: int = 4) -> list[dict]:
    """资金费历史（分页并行）。

    币安 fapi/v1/fundingRate 单次上限 1000 条，一天 3 条，
    9 年约 9,900 条 → 10 个分页。
    """
    from concurrent.futures import ThreadPoolExecutor

    if end_ms is None:
        end_ms = int(time.time() * 1000)
    span = 1000 * FUNDING_MS
    starts = list(range(start_ms, end_ms, span))

    def fetch_one(s: int) -> list[dict]:
        out = _get("binance_futures", BINANCE_FUTURES_BASE, "/fapi/v1/fundingRate",
                   {"symbol": symbol, "startTime": s, "endTime": min(s + span - 1, end_ms),
                    "limit": 1000})
        if pause:
            time.sleep(pause)
        return out if isinstance(out, list) else []

    rows: list[dict] = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for chunk in ex.map(fetch_one, starts):
            rows.extend(chunk)

    seen: dict[int, dict] = {}
    for r in rows:
        seen[int(r["fundingTime"])] = {
            "funding_time": int(r["fundingTime"]),
            "symbol": r["symbol"],
            "funding_rate": float(r["fundingRate"]),
            "mark_price": float(r.get("markPrice") or 0.0),
            "rate_type": r.get("rateType", ""),
        }
    return [seen[k] for k in sorted(seen)]


def perp_klines(symbol: str, interval: str = "1h", limit: int = 1000,
                start_ms: int | None = None, end_ms: int | None = None) -> list[Bar]:
    """永续 K 线（用于计算基差：永续价 vs 现货价）。"""
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    if start_ms is not None:
        params["startTime"] = start_ms
    if end_ms is not None:
        params["endTime"] = end_ms
    batch = _get("binance_futures", BINANCE_FUTURES_BASE, "/fapi/v1/klines", params)
    out = []
    for k in batch:
        out.append(Bar(int(k[0]), float(k[1]), float(k[2]), float(k[3]),
                       float(k[4]), float(k[5]), float(k[7])))
    return out


def perp_klines_parallel(symbol: str, interval: str = "1h",
                         start_ms: int | None = None, end_ms: int | None = None,
                         workers: int = 6, pause: float = 0.05) -> list[Bar]:
    """分块并行拉取永续全量历史（与 binance_klines_parallel 同一套做法）。"""
    from concurrent.futures import ThreadPoolExecutor

    step = INTERVAL_MS[interval]
    if end_ms is None:
        end_ms = int(time.time() * 1000)
    if start_ms is None:
        start_ms = end_ms - step * 1000
    starts = list(range(start_ms, end_ms, step * 1000))

    def fetch_one(s: int) -> list[Bar]:
        batch = _get("binance_futures", BINANCE_FUTURES_BASE, "/fapi/v1/klines",
                     {"symbol": symbol, "interval": interval,
                      "startTime": s, "endTime": min(s + step * 1000 - 1, end_ms),
                      "limit": 1000})
        if pause:
            time.sleep(pause)
        return [Bar(int(k[0]), float(k[1]), float(k[2]), float(k[3]),
                    float(k[4]), float(k[5]), float(k[7])) for k in batch]

    out: list[Bar] = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for chunk in ex.map(fetch_one, starts):
            out.extend(chunk)
    seen: dict[int, Bar] = {b.open_time: b for b in out}
    return [seen[k] for k in sorted(seen)]


def premium_index(symbol: str) -> dict:
    """当前标记价、指数价与最近资金费。"""
    return _get("binance_futures", BINANCE_FUTURES_BASE, "/fapi/v1/premiumIndex",
                {"symbol": symbol})


# --------------------------------------------------------------------------
# OKX / Coinbase（校验源，只取近期做交叉验证）
# --------------------------------------------------------------------------
OKX_BASE = "https://www.okx.com"
COINBASE_BASE = "https://api.exchange.coinbase.com"


def okx_klines(inst_id: str, bar: str = "1H", limit: int = 100) -> list[Bar]:
    data = _get("okx", OKX_BASE, "/api/v5/market/candles",
                {"instId": inst_id, "bar": bar, "limit": limit})
    if str(data.get("code")) != "0":
        raise RuntimeError(f"OKX 返回异常: {data.get('msg')}")
    out = []
    for k in data["data"]:            # OKX 返回按时间倒序
        out.append(Bar(int(k[0]), float(k[1]), float(k[2]), float(k[3]),
                       float(k[4]), float(k[5]), float(k[7])))
    return sorted(out, key=lambda b: b.open_time)


def coinbase_klines(product: str = "BTC-USD", granularity: int = 3600) -> list[Bar]:
    data = _get("coinbase", COINBASE_BASE, f"/products/{product}/candles",
                {"granularity": granularity})
    out = []
    for k in data:                    # [time, low, high, open, close, volume]
        out.append(Bar(int(k[0]) * 1000, float(k[3]), float(k[2]), float(k[1]),
                       float(k[4]), float(k[5]), float(k[5]) * float(k[4])))
    return sorted(out, key=lambda b: b.open_time)
