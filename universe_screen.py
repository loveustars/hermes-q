"""标的池筛选 —— 用真实数据（币安公开只读行情）给候选币打分。

筛选维度：
  1. 年化波动率        σ_annual = σ_hourly * sqrt(24*365)
  2. 小时成交额中位数   流动性深度
  3. 1万U 参与率 Q/V
  4. 单边冲击 sqrt 律    σ_hourly * sqrt(Q/V)
  5. 往返总成本         2*fee + 2*impact   (fee=0.1%/边)
  6. 冲击占总成本比例    —— 这个数越高，成本模型越不可信

只用公开行情，不接任何账号。
"""
import json
import time
import urllib.request
import numpy as np
import pandas as pd

BASE = "https://api.binance.com/api/v3/klines"
SYMBOLS = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT", "ADAUSDT",
           "TRXUSDT", "LTCUSDT", "LINKUSDT", "DOTUSDT", "AVAXUSDT", "ATOMUSDT",
           "UNIUSDT", "DOGEUSDT"]
ORDER_USD = 1e4
FEE_ONE_SIDE = 0.001


def fetch(symbol, interval="1h", limit=1000):
    url = f"{BASE}?symbol={symbol}&interval={interval}&limit={limit}"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=25) as r:
        return json.loads(r.read().decode())


rows = []
for sym in SYMBOLS:
    try:
        raw = fetch(sym)
    except Exception as e:
        print(f"  ! {sym} 拉取失败: {e}")
        continue
    df = pd.DataFrame(raw).iloc[:, :6]
    df.columns = ["ot", "o", "h", "l", "c", "v"]
    for c in ["o", "h", "l", "c", "v"]:
        df[c] = df[c].astype(float)
    df["ot"] = pd.to_datetime(df["ot"], unit="ms", utc=True)
    df["notional"] = df["v"] * df["c"]

    ret = np.log(df["c"] / df["c"].shift()).dropna()
    sigma_h = float(ret.std())
    sigma_a = sigma_h * np.sqrt(24 * 365)
    vol_h = float(df["notional"].median())
    part = ORDER_USD / vol_h
    impact = sigma_h * np.sqrt(part)
    total = 2 * FEE_ONE_SIDE + 2 * impact

    rows.append({
        "symbol": sym.replace("USDT", ""),
        "price": float(df["c"].iloc[-1]),
        "sigma_h_%": sigma_h * 100,
        "sigma_ann_%": sigma_a * 100,
        "vol_h_M": vol_h / 1e6,
        "part_%": part * 100,
        "impact_bp": impact * 1e4,
        "rt_cost_%": total * 100,
        "impact_share_%": (2 * impact) / total * 100,
        "start": df["ot"].iloc[0].strftime("%Y-%m-%d"),
    })
    time.sleep(0.25)

t = pd.DataFrame(rows).sort_values("vol_h_M", ascending=False)

print(f"\n候选池打分（1万 U 单次下单，费率 0.1%/边）  数据窗口 {t['start'].min()} ~ 2026-09-14\n")
hdr = (f"{'币':<6}{'价格':>10}{'年化波动':>10}{'小时成交额':>12}{'参与率':>10}"
       f"{'单边冲击':>10}{'往返成本':>10}{'冲击占比':>10}")
print(hdr)
print("-" * len(hdr.encode('gbk', errors='ignore').decode('gbk')) )
for _, r in t.iterrows():
    print(f"{r['symbol']:<6}{r['price']:>10,.2f}{r['sigma_ann_%']:>9.1f}%"
          f"{r['vol_h_M']:>10,.2f}M{r['part_%']:>9.3f}%{r['impact_bp']:>9.2f}bp"
          f"{r['rt_cost_%']:>9.3f}%{r['impact_share_%']:>9.1f}%")

print()
print("=" * 76)
print("按「流动性深度」门槛筛选（推荐规则）")
for min_vol in [50e6, 20e6, 5e6]:
    keep = t[t["vol_h_M"] * 1e6 >= min_vol]
    print(f"  小时成交额 >= {min_vol/1e6:>5.0f}M USDT  ->  {len(keep):>2} 个: "
          f"{', '.join(keep['symbol'])}")

print()
print("按「波动率」门槛筛选")
for max_sig in [60, 80, 100, 120]:
    keep = t[t["sigma_ann_%"] <= max_sig]
    print(f"  年化波动 <= {max_sig:>3}%  ->  {len(keep):>2} 个: {', '.join(keep['symbol'])}")

print()
print("按「冲击占比 <= 25%」筛选（成本模型可信度门槛）")
keep = t[t["impact_share_%"] <= 25]
print(f"  -> {len(keep)} 个: {', '.join(keep['symbol'])}")

t.to_csv("/home/nick/workspace/quant/data/universe_screen.csv", index=False)
print("\n已保存 -> quant/data/universe_screen.csv")
