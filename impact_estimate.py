"""冲击成本量级估算 —— 用真实抓取的 BTCUSDT 小时线做锚点。

冲击模型（平方根律 / Almgren 简化版）：
    临时冲击 ≈ k · σ · sqrt(Q / V)
  Q = 本次下单名义额，V = 同周期市场成交名义额，σ = 同周期收益率标准差
往返总成本 = 2·f  (手续费)  +  2·impact  (进出各一次冲击)  +  点差
"""
import json
import numpy as np
import pandas as pd

raw = json.load(open('/home/nick/workspace/quant/data/btcusdt_1h_probe.json'))
df = pd.DataFrame(raw).iloc[:, :6]
df.columns = ['ot', 'o', 'h', 'l', 'c', 'v']
for c in ['o', 'h', 'l', 'c', 'v']:
    df[c] = df[c].astype(float)
df['ot'] = pd.to_datetime(df['ot'], unit='ms', utc=True)
df['notional'] = df['v'] * df['c']

sigma_h = float(np.log(df['c'] / df['c'].shift()).std())
vol_h = float(df['notional'].median())
last = float(df['c'].iloc[-1])

print(f"样本: {df['ot'].iloc[0]:%Y-%m-%d} ~ {df['ot'].iloc[-1]:%Y-%m-%d}  根数={len(df)}")
print(f"最新价           {last:,.1f} USDT")
print(f"小时波动率 sigma {sigma_h * 100:.3f}%")
print(f"小时成交额中位   {vol_h / 1e6:,.1f} 百万 USDT")
print()

FEE_RT = 0.002  # 往返手续费：普通用户 0.1% + 0.1%
ORDER = 1e4      # 单次下单 1 万 USDT

rows = [
    ("BTCUSDT (实测)", vol_h, sigma_h),
    ("中型山寨币 (估)", 8e5, 0.020),
    ("薄盘山寨币 (估)", 8e4, 0.035),
]

print(f"{'标的':<20}{'小时成交额':>12}{'参与率':>11}{'冲击(单边)':>12}{'往返总成本':>12}{'冲击占比':>10}")
for name, v, s in rows:
    part = ORDER / v
    imp = s * np.sqrt(part)
    total = FEE_RT + 2 * imp
    share = (2 * imp) / total
    print(f"{name:<20}{v / 1e6:>10,.1f}M{part * 100:>10.4f}%{imp * 100:>11.4f}%"
          f"{total * 100:>11.4f}%{share * 100:>9.1f}%")

print()
print("成本拖累：每笔往返 0.200%（普通档） / 0.150%（持 BNB 档）")
for n in [10, 50, 200, 1000]:
    print(f"  年换手 {n:>4} 次往返 -> 手续费拖累 {n * 0.002 * 100:>6.1f}%/年"
          f"   (BNB 档 {n * 0.0015 * 100:>6.1f}%/年)")

print()
print("结论：在 BTC 上 1 万 U 的冲击成本可以忽略；在小市值标的上冲击成为主要成本项。")
