"""实时行情源封装 —— 带退避重试，且可测。

把取数与重试从脚本挪进 `src/`，理由是**脚本里的逻辑测不到**：
重试这种事一旦写错（比如把次数写反、退避不生效），
只会表现为"偶尔失败"，很难在生产里发现。

仍然只用只读公开端点（白名单在 `src/data/sources.py::ALLOWED_PATHS`）。
"""
from __future__ import annotations

import time
from typing import Any, Callable

from ..data import sources


def retry_call(fn: Callable[[], Any], attempts: int = 3,
               base_delay: float = 2.0, sleep: Callable[[float], None] = time.sleep):
    """带指数退避的重试：失败则等 base_delay、2×base_delay…… 后重试。

    最后一次失败后**照常抛错**——不吞异常。
    网络问题必须让上层看见（tick 会跳过该标的、不动状态），
    否则就会变成"安静地少算了一笔"，而那种错误没人会发现。

    `sleep` 可注入，测试里用假的避免真的等待。
    """
    attempts = max(1, int(attempts))
    last: Exception | None = None
    for k in range(attempts):
        try:
            return fn()
        except Exception as e:                     # noqa: BLE001
            last = e
            if k < attempts - 1:
                sleep(base_delay * (2 ** k))
    raise last if last is not None else RuntimeError("retry_call: 无异常但无返回值")


def _quote_once(symbol: str) -> dict:
    """单次取数：现货价 / 永续价 / 标记价 / 最近费率。"""
    spot = float(sources._get("binance", sources.BINANCE_BASE,      # noqa: SLF001
                              "/api/v3/ticker/price", {"symbol": symbol})["price"])
    perp = float(sources._get("binance_futures",                    # noqa: SLF001
                              sources.BINANCE_FUTURES_BASE,
                              "/fapi/v1/ticker/price", {"symbol": symbol})["price"])
    pi = sources.premium_index(symbol)
    return {"symbol": symbol, "spot_px": spot, "perp_px": perp,
            "mark_px": float(pi["markPrice"]),
            "last_funding_rate": float(pi.get("lastFundingRate") or 0.0),
            "now_ms": int(time.time() * 1000)}


def live_quote(symbol: str, attempts: int = 3, base_delay: float = 2.0,
               sleep: Callable[[float], None] = time.sleep) -> dict:
    """取实时行情，带重试。

    为什么需要重试：2026-09-14 17:00~23:00 UTC 连续 7 次整点 tick 全部失败，
    报错 `SSL: UNEXPECTED_EOF_WHILE_READING` —— **代理自身的上游在那几小时不可用**。
    秒级抖动本可以被重试吸收，单次尝试则会把整点白扔掉。

    但别指望重试能救小时级中断：那种情况就该如实失败并由调度器上报。
    """
    return retry_call(lambda: _quote_once(symbol), attempts=attempts,
                      base_delay=base_delay, sleep=sleep)
