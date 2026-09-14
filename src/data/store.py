"""本地数据仓库 —— CSV 落盘 + SHA256 版本清单。

不用 pyarrow/parquet：第一版只依赖 numpy/pandas，行为可控。
每份数据都带 sha256，任何一次实验都能追溯到确切的数据版本。
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
from datetime import datetime, timezone
from typing import Iterable

import numpy as np
import pandas as pd

COLS = ["open_time", "open", "high", "low", "close", "volume", "quote_volume"]


def project_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def data_dir(*parts: str) -> str:
    p = os.path.join(project_root(), "data", *parts)
    os.makedirs(p, exist_ok=True)
    return p


def sha256_file(path: str, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def save_bars(bars: Iterable, symbol: str, interval: str, source: str = "binance") -> str:
    """落盘为 CSV，返回路径。同时更新 manifest。"""
    rows = [b.__dict__ if hasattr(b, "__dict__") else b for b in bars]
    path = os.path.join(data_dir("raw"), f"{source}_{symbol}_{interval}.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        w.writeheader()
        for r in rows:
            w.writerow({c: r[c] for c in COLS})
    update_manifest(path, symbol, interval, source, len(rows))
    return path


def load_bars(symbol: str, interval: str = "1h", source: str = "binance") -> pd.DataFrame:
    path = os.path.join(data_dir("raw"), f"{source}_{symbol}_{interval}.csv")
    df = pd.read_csv(path)
    df["dt"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    return df.set_index("dt").sort_index()


def manifest_path() -> str:
    return os.path.join(data_dir(), "manifest.json")


def update_manifest(path: str, symbol: str, interval: str, source: str, n_rows: int) -> None:
    mp = manifest_path()
    man = {}
    if os.path.exists(mp):
        with open(mp, encoding="utf-8") as f:
            man = json.load(f)
    key = os.path.relpath(path, project_root())
    man[key] = {
        "symbol": symbol,
        "interval": interval,
        "source": source,
        "rows": n_rows,
        "sha256": sha256_file(path),
        "bytes": os.path.getsize(path),
        "recorded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    with open(mp, "w", encoding="utf-8") as f:
        json.dump(man, f, ensure_ascii=False, indent=2)


def read_manifest() -> dict:
    mp = manifest_path()
    if not os.path.exists(mp):
        return {}
    with open(mp, encoding="utf-8") as f:
        return json.load(f)


def verify_manifest() -> list[str]:
    """校验落盘数据是否与 manifest 一致。返回不一致项列表。"""
    bad = []
    for rel, meta in read_manifest().items():
        p = os.path.join(project_root(), rel)
        if not os.path.exists(p):
            bad.append(f"缺失 {rel}")
        elif sha256_file(p) != meta["sha256"]:
            bad.append(f"哈希不符 {rel}")
    return bad


def health(frame: pd.DataFrame) -> dict:
    """数据体检：缺口、重复、异常值。"""
    idx = frame.index
    gaps = idx.to_series().diff().dropna()
    step = gaps.mode().iloc[0] if len(gaps) else None
    holes = int((gaps > step).sum()) if step is not None else 0
    return {
        "rows": int(len(frame)),
        "start": str(idx[0]) if len(frame) else None,
        "end": str(idx[-1]) if len(frame) else None,
        "dup_index": int(idx.duplicated().sum()),
        "step": str(step) if step is not None else None,
        "gaps": holes,
        "nan": int(frame[["open", "high", "low", "close", "volume"]].isna().sum().sum()),
        "nonpositive_close": int((frame["close"] <= 0).sum()),
    }
