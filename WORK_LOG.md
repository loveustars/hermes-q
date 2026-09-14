# WORK_LOG —— 跨 session 交接

**项目**：Hermes-Q（资产无关的自学习交易框架，研究用途，只读公开行情）
**位置**：`/home/nick/workspace/quant`
**最后更新**：2026-09-14
**当前阶段**：M0 / M1 / M2 已完成，下一步 M3

---

## 一、目标一句话

在扣除真实摩擦成本（含市场冲击）后，在未参与训练的封存数据上，评估智能体是否存在
**可重复、统计显著**的交易边际。产出以框架和评估协议为主，智能体成败其次。

---

## 二、已完成

| 里程碑 | 状态 | 产物 |
|---|---|---|
| M0 骨架与铁律 | ✅ | `configs/base.json`、`src/registry/runs.py`、`src/config.py` |
| M1 数据层 | ✅ | 三源适配器、并行采集、manifest sha256、`configs/spreads.json` |
| M2 撮合与成本模型 | ✅ | `src/sim/costs.py`、`src/sim/exchange.py`、`scripts/m2_baseline.py` |

**核心结论（M2）**：换手率一票否决。年换手 131 次吃掉 30.6% 毛利润；
2160 次吃掉 95.2%；100,266 次（小时换仓）**账户不到两个月被手续费清零**。
买入持有（0.1 次/年）几乎不受影响。

---

## 三、已知问题 / 待办

1. **1h 数据有 27-28 处缺口**（BTC/ETH 各 28，BNB 27），来源是币安停机或维护。
   目录 `data/raw/` 中的缺口未补。影响：极小（占比 0.035%），但回测跨越缺口时
   `next-bar` 成交的时间间隔不是 1 小时，需在 M3 里显式处理或剔除。
2. **BNB 是边界标的**。冲击占比按近 12 个月口径 22.5%（门内），
   按全历史口径 35.9%（门外）。口径已统一为「近 12 个月小时成交额中位数」。
3. **本次样本里 Sharpe 没有区分度**：随机策略毛 Sharpe 0.94-1.04，
   因为 2017-2026 是大牛市。评估必须强制与 buy&hold 对照（M4 处理）。
4. **未实现的成本项**：资金费（永续）、部分成交队列模型、网络延迟的随机扰动。
   当前只有手续费 + 实测点差 + 平方根冲击 + 1 根 bar 延迟。
5. **只支持现货多头**（`allow_short=False`）。做空需接永续并计资金费，留到 M6 之后。

---

## 四、踩过的坑（务必先看）

1. **收益基数错误（结论级）**：净值曲线在第一根 bar 扣成本，用 `equity[0]` 做收益基数
   会让净收益百分比**高于**毛收益（BNB 毛 45,586% / 净 55,849%），成本效应被完全洗掉。
   修复：一律以固定初始本金为基数，并同时报绝对金额差。
2. **换手统计被截断（结论级）**：净账本归零后不再交易，换手停止累加，
   小时换仓策略换手一度显示 16 次/年（真实值 100,266）。
   修复：换手按毛账本统计，成本按净账本统计，并记录归零时刻。
3. **"持有"语义缺失**：agent 返回空 dict 曾被执行成"清仓"，导致买入持有曲线是平的。
   修复：契约改为「返回完整目标权重；返回 `None` 表示维持现状」。
4. **串行采集太慢**：8 年小时线串行要 7 分钟。已改为分块并行 + 断点续传，约 30 秒。
5. **1 万 U 的冲击成本在 BTC 上可忽略**（参与率 0.02%），
   所以冲击建模的价值在泛化能力，不在这三个标的。别把它当成当前标的的成本项。

---

## 五、怎么跑

```bash
cd /home/nick/workspace/quant

# 1. 采集全量数据（可重复执行，只补增量）
python3 scripts/m1_ingest.py

# 2. 基线对照 + 冲击表 + 权益曲线
python3 scripts/m2_baseline.py

# 3. 口径敏感性核验
python3 checks_window_sensitivity.py

# 4. 数据完整性校验
python3 -c "from src.data import store; print(store.verify_manifest())"
```

产物：`runs/<runid>/` 下含 `config.json`、`meta.json`（config hash + 数据版本哈希）、
`metrics.json`、`baseline_compare.json`、`curves/*.csv`。

---

## 六、关键文件位置

```
quant/
├── PLAN.md                      # 设计与里程碑（含 M0-M8 与淘汰门）
├── WORK_LOG.md                  # 本文件
├── configs/base.json            # 冻结核心池 BTC/ETH/BNB + 成本与风险参数
├── configs/spreads.json         # 实测点差（由 m1_ingest 生成）
├── src/data/sources.py          # 三源只读适配器 + 合规白名单
├── src/data/store.py            # CSV 落盘 + sha256 manifest
├── src/env/market_view.py       # 因果封印（越权抛 LookaheadError）
├── src/sim/costs.py             # 手续费 / 点差 / 平方根冲击
├── src/sim/exchange.py          # 双账本撮合仿真
├── src/eval/metrics.py          # 指标（固定本金基数）
├── src/agents/baselines.py      # L0 基线
└── runs/                        # 实验登记
```

---

## 七、下一步（M3 + M4）

- **M3**：`tests/test_causality.py` 未来函数注入测试（策略尝试读 t+1 必须抛
  `LookaheadError`）；补齐 L0 基线。
- **M4**：评估协议 —— walk-forward、purged k-fold + embargo、block bootstrap、
  Deflated Sharpe、PBO(CSCV)。
- **淘汰门 G1（硬）**：M4 完成后，对 100 个纯随机信号，假阳性率必须 ≤ 5%。
  超过 10% 就地停住修评估器，不许进入 M5。

---

## 八、合规

- 只读公开行情，**不接账号、不存 key、不碰下单接口**（代码级白名单强制）
- 不涉及虚拟货币兑换/交易/承销，不触及 2026-02-06 八部委 42 号文禁止的境内业务活动
