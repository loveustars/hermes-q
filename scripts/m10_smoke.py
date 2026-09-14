"""M10-1 carry 仿真冒烟测试 —— 确认账目平衡与三桶记账正确。

在跑网格之前必须先确认这几件事，否则网格扫出来的全是垃圾：
  1. 开仓后 权益 = 初始资本 − 手续费（恒等式）
  2. 零资金费、零价格变动 ⇒ 权益只被手续费消耗
  3. 正资金费 ⇒ 空头腿收入，权益上升
  4. 备用金为 0 + 价格持续上冲 ⇒ 强平
  5. 同样的上冲但备用金充足 ⇒ 通过补保存活
  6. 瞬时跳空 ⇒ 补保来不及，直接强平（跳空风险，必须如实记录）
  7. 资本不足 ⇒ 报错而不是静默算错
  8. 两腿同向变动时 delta 中性 ⇒ 净值平稳
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.sim.carry import CarryConfig, CarrySimulator  # noqa: E402
from src.sim.funding import FundingTable  # noqa: E402

COST_RATE = 0.0011     # fee 0.001 + spread 0.0001


def frames(n=400, spot_drift=0.0, perp_drift=0.0, spike_at=None, spike_pct=0.0,
           spot0=100.0, perp0=100.0):
    idx = pd.date_range("2022-01-01", tz="UTC", periods=n, freq="h")
    s = spot0 * np.exp(spot_drift * np.arange(n))
    p = perp0 * np.exp(perp_drift * np.arange(n))
    if spike_at is not None:
        p = p.copy()
        p[spike_at:] *= (1.0 + spike_pct)
    def mk(px):
        return pd.DataFrame({"open": px, "high": px * 1.0001, "low": px * 0.9999,
                             "close": px, "volume": 1e6, "quote_volume": px * 1e6},
                            index=idx)
    return mk(s), mk(p)


def ms(ts):
    return int(ts.timestamp() * 1000)


def main() -> None:
    ok = True

    # ---- 1/2. 恒等式：零变动下权益只被手续费消耗 ----
    spot, perp = frames(400)
    ft_empty = FundingTable()
    cfg = CarryConfig(initial_capital=10_000, notional_ratio=0.4,
                      initial_margin_ratio=1.0, rebalance_every=168, warmup=50)
    res = CarrySimulator(spot, perp, ft_empty, cfg, "AAA").run()
    fees = res.total_fees()
    print(f"[1] 零变动：终值 {res.final():,.2f}  总手续费 {fees:.2f}  "
          f"恒等式偏差 {abs(res.final() - (10_000 - fees)):.2e}")
    if abs(res.final() - (10_000 - fees)) > 1e-6:
        print("    !! 恒等式不成立"); ok = False
    if res.liquidated:
        print("    !! 零变动不该被强平"); ok = False
    if res.n_rebalances < 1:
        print("    !! 再平衡没有发生"); ok = False

    # ---- 3. 正资金费 ⇒ 空头收入 ----
    ft = FundingTable()
    settles = [ms(spot.index[i]) for i in range(60, 400, 8)]
    for ts in settles:
        ft.add("AAA", ts, 0.001)
    cfg3 = CarryConfig(initial_capital=10_000, notional_ratio=0.4,
                       initial_margin_ratio=1.0, rebalance_every=10**9, warmup=50)
    res3 = CarrySimulator(spot, perp, ft, cfg3, "AAA").run()
    expect = 0.001 * 4_000 * len(settles)
    print(f"[2] 正资金费：收到 {res3.total_funding():,.2f}  期望 {expect:,.2f}  "
          f"终值 {res3.final():,.2f}  结算 {len(settles)} 次")
    if abs(res3.total_funding() - expect) > 1.0:
        print("    !! 资金费金额不符"); ok = False
    if res3.final() <= res3.initial_capital:
        print("    !! 正资金费未能提升权益"); ok = False

    # ---- 4. 不再平衡 + 持续上冲 + 备用金 0 ⇒ 强平 ----
    spot4, perp4 = frames(600, perp_drift=0.003)
    cfg4 = CarryConfig(initial_capital=10_000, notional_ratio=0.5,
                       initial_margin_ratio=1.0, rebalance_every=10**9, warmup=50)
    res4 = CarrySimulator(spot4, perp4, ft_empty, cfg4, "AAA").run()
    print(f"[3] 不再平衡、备用金 0、永续持续上冲："
          f"{'已强平 @' + str(res4.liquidated_at) if res4.liquidated else '未强平'}"
          f"  终值 {res4.final():,.2f}")
    if not res4.liquidated:
        print("    !! 不再平衡时应当被强平"); ok = False

    # ---- 5. 同上但上冲温和 + 备用金充足 ⇒ 补保留活 ----
    # 注意场景要匹配：永续涨 5.2 倍时空头亏 8,400，而可动用资金 8,000，
    # 强平是**正确**的。这里换成温和上冲，才是在检验补保机制本身。
    spot5, perp5 = frames(600, perp_drift=0.0008)     # 550 根后约 +55%
    cfg5 = CarryConfig(initial_capital=10_000, notional_ratio=0.2,
                       initial_margin_ratio=1.0, rebalance_every=10**9, warmup=50)
    res5 = CarrySimulator(spot5, perp5, ft_empty, cfg5, "AAA").run()
    print(f"[4] 温和上冲（+55%）+ 备用金 0.6："
          f"{'已强平' if res5.liquidated else '未强平'}  "
          f"补保 {res5.n_topups} 次 共 {res5.topup_amount:,.0f}  "
          f"终值 {res5.final():,.2f}")
    if res5.liquidated:
        print("    !! 备用金充足时不该被强平"); ok = False
    if res5.n_topups == 0:
        print("    !! 应触发过补保"); ok = False

    # ---- 5b. 有再平衡 + 备用金 0 ⇒ 靠自动降杠杆活下来（这是重要发现）----
    res5b = CarrySimulator(spot4, perp4, ft_empty,
                           CarryConfig(initial_capital=10_000, notional_ratio=0.5,
                                       initial_margin_ratio=1.0,
                                       rebalance_every=168, warmup=50), "AAA").run()
    print(f"[4b] 同样的上冲与备用金 0，但每 168 根再平衡："
          f"{'已强平' if res5b.liquidated else '未强平'}  终值 {res5b.final():,.2f}"
          f"  （再平衡 = 自动降杠杆，避免了强平）")
    print("     注：再平衡频率不只是成本参数，也是风控参数 —— 它会自动把仓位降到与缩水的权益匹配。")

    # ---- 6. 瞬时跳空 ⇒ 强平（补保来不及）----
    spot6, perp6 = frames(300, spike_at=150, spike_pct=1.5)
    cfg6 = CarryConfig(initial_capital=10_000, notional_ratio=0.2,
                       initial_margin_ratio=1.0, rebalance_every=10**9, warmup=50)
    res6 = CarrySimulator(spot6, perp6, ft_empty, cfg6, "AAA").run()
    print(f"[5] 瞬时 +150% 跳空："
          f"{'已强平 @' + str(res6.liquidated_at) if res6.liquidated else '未强平'}"
          f"（跳空风险，补保来不及）")
    if not res6.liquidated:
        print("    !! 瞬时跳空应触发强平"); ok = False

    # ---- 7. 资本不足报错 ----
    try:
        CarrySimulator(spot, perp, ft_empty,
                       CarryConfig(initial_capital=10_000, notional_ratio=0.8,
                                   initial_margin_ratio=1.0, warmup=50), "AAA").run()
    except ValueError as e:
        print(f"[6] 资本不足正确报错：{str(e)[:46]}...")
    else:
        print("    !! 资本不足未被拦截"); ok = False

    # ---- 8. 两腿同向变动 ⇒ delta 中性下净值平稳 ----
    spot8, perp8 = frames(1000, spot_drift=0.0005, perp_drift=0.0005)
    res8 = CarrySimulator(spot8, perp8, ft_empty,
                          CarryConfig(initial_capital=10_000, notional_ratio=0.4,
                                      initial_margin_ratio=1.0,
                                      rebalance_every=10**9, warmup=50), "AAA").run()
    drift = (res8.final() - 10_000) / 10_000
    print(f"[7] 两腿同向上涨 64%（delta 中性）：终值 {res8.final():,.2f}  "
          f"偏离 {drift*100:+.3f}%")
    if abs(drift) > 0.02:
        print("    !! delta 中性下不该有明显漂移"); ok = False

    print(f"\n{'全部通过' if ok else '存在失败项'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
