"""长窗口波动率与流动性核验 —— 用 5 年日线，判断哪些标的是"稳定"的。

短窗口（41 天小时线）的波动率估计不稳，这里拉 2021 年至今的日线：
  - 全样本年化波动率
  - 分年度年化波动率（看波动率本身稳不稳）
  - 近期日均成交额（流动性深度）
只用公开行情，不接任何账号。
"""
import json
import time
import urllib.request
import numpy as np
import pandas as pd

BASE = "https://api.binance.com/api/v3/klines"
SYMBOLS = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT",
           "ADAUSDT", "TRXUSDT", "LTCUSDT", "LINKUSDT"]
START = 1609459200000  # 2021-01-01 UTC
DAY = 86400000


def fetch_all(symbol):
    out, cursor = [], START
    while True:
        url = f"{BASE}?symbol={symbol}&interval=1d&startTime={cursor}&limit=1000"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=30) as r:
            batch = json.loads(r.read().decode())
        if not batch:
            break
        out.extend(batch)
        if len(batch) < 1000:
            break
        cursor = batch[-1][6] + DAY
        time.sleep(0.25)
    return out


recs = []
for sym in SYMBOLS:
    raw = fetch_all(sym)
    df = pd.DataFrame(raw).iloc[:, :6]
    df.columns = ["ot", "o", "h", "l", "c", "v"]
    for c in ["o", "h", "l", "c", "v"]:
        df[c] = df[c].astype(float)
    df["ot"] = pd.to_datetime(df["ot"], unit="ms", utc=True)
    df = df.drop_duplicates("ot").set_index("ot")
    df["notional"] = df["v"] * df["c"]
    ret = np.log(df["c"] / df["c"].shift()).dropna()

    ann = float(ret.std()) * np.sqrt(365)
    recent = df.loc[df.index >= "2025-01-01"]
    rec_ret = np.log(recent["c"] / recent["c"].shift()).dropna()

    rec = {
        "sym": sym.replace("USDT", ""),
        "days": len(df),
        "start": df.index[0].strftime("%Y-%m-%d"),
        "vol_all_%": ann * 100,
        "vol_2025_%": float(rec_ret.std()) * np.sqrt(365) * 100,
        "vol_ratio": (float(rec_ret.std()) * np.sqrt(365)) / ann,
        "notional_M": float(recent["notional"].median()) / 1e6,
    }
    for y in ["2021", "2022", "2023", "2024", "2025"]:
        s = df.loc[df.index.year == int(y)]
        if len(s) > 30:
            r = np.log(s["c"] / s["c"].shift()).dropna()
            rec[f"v{y}"] = float(r.std()) * np.sqrt(365) * 100
        else:
            rec[f"v{y}"] = np.nan
    recs.append(rec)
    time.sleep(0.25)

t = pd.DataFrame(recs).sort_values("vol_all_%")

print("\n5 年日线核验（2021-01-01 起，币安公开数据）\n")
hdr = f"{'币':<6}{'天数':>6}{'全样本年化波动':>14}" + "".join(f"{y:>9}" for y in ["2021", "2022", "2023", "2024", "2025"]) + f"{'日均成交额':>12}"
print(hdr)
print("-" * 88)
for _, r in t.iterrows():
    line = f"{r['sym']:<6}{r['days']:>6}{r['vol_all_%']:>13.1f}%"
    for y in ["2021", "2022", "2023", "2024", "2025"]:
        v = r[f"v{y}"]
        line += f"{'  --':>9}" if pd.isna(v) else f"{v:>8.1f}%"
    line += f"{r['notional_M']:>10,.0f}M"
    print(line)

print("\n（右起第 5~1 列是各年度年化波动率，最后一列是 2025 年至今日均成交额）")
t.to_csv("/home/nick/workspace/quant/data/universe_longwindow.csv", index=False)
print("已保存 -> quant/data/universe_longwindow.csv")
