# 读这份归因表之前要知道的四件事

**run**：`20260915T064200_vol_attribution_c9c280`（`vol_attribution.py`，135/141 完成）
**自述**：`REPORT.md` 称其为**权威 run**（配套 run 是 `20260915T060059_vol_attribution_ab3b85`）
**窗口**：77,503 根（三池统一裁剪）
**复核**：2026-09-15 由委托人独立复核（`WORK_LOG.md` §17.11 / §17.12）

---

## 1. ⚠️ `n_funding_events = 0` 且 `n_liquidated > 0` 的行，资金费块无信息量

**强平通道会把若干候选的持仓清得极早**，此后全程空仓 ⇒ 资金费**按构造恒为零**。
`funding_pct_of_initial = 0.0` 是**真值**，但**不能用来论证"资金费无关紧要"**。

已确认的机制（与 `ab3b85` 同源）：`exp:flat` 无持仓；`k05_*` 的假强平**开仓即触发**；
满额做空在阈值 `(1+k)/(1+m) = +81.8%` 被强平（BTC 从 2017-11 的 7,676 到 12 月的 13,800 即 +80%）。

**读法**：先看 `n_funding_events` 与 `n_liquidated` 两列，再引用任何资金费行。

---

## 2. ⚠️ 有 3 行的**变体标签是误导的**（旧记录未被迁移）

| variant | candidate | 净终值 | `funding_pct_of_initial` | `engine_funding` | `n_funding_events` |
|---|---|---|---|---|---|
| `k1m005` | `exp:long_all` | 88,161 | 0.0 | **None** | None |
| `k1m005` | `exp:short_all` | 0 | 0.0 | **None** | None |
| `k1m005` | `exp:short_mom_1` | 2 | 0.0 | **None** | None |

`k1m005` **不在变体迁移映射里**，所以这 3 条以旧名存活。

- **数字是对的**：88,161 正是"资金费 **OFF**"下的值 —— 旧脚本没有"引擎资金费"这一维，
  它写出的记录全是**不收**资金费下跑的，故 `funding_pct_of_initial = 0.0` 为真值。
- **但标签会骗人**：读者可能把 `k1m005` 当成"k=1、m=0.005 且**收**资金费"的那一档。
- `engine_funding = None` 是唯一诚实的提示（"该行资金费口径未知"）。

**这 3 条已在 `WORK_LOG.md` §17.11 的 schema 守卫覆盖范围内**（它们只有 13 个字段、
无 `funding_paid`）⇒ 新版脚本会**丢弃并重跑**。本目录早于该修复。

---

## 3. 产物缺 `config.json` 与 `meta.json`（与 `REPORT.md` §8 的清单不符）

`REPORT.md` §8 把 `config.json` / `meta.json` 列在产物清单里，**实际两者都不存在**
⇒ 这份"权威 run"**没有走项目的 Run 登记**：**引擎设置只写在报告正文里、非机器可读**。
（假说与判据确实在 `metrics.json` 里：`H1_*` / `H2_*` / `status=complete` / `n_done=135`。）

**已核实这是那次运行的启动方式造成的异常，不是脚本缺陷**：用当前脚本以 `--run-dir` 全新跑一次
（`--pools btc`，33/33 完成）**会正常写出** `config.json` + `meta.json`。

另：`REPORT.md` 写 `n_done = 135/141`，而 `metrics.json` 的 `n_planned = 160`、另一份 run 是 241
—— **计划数没有单一权威值**。

---

## 4. 单次运行的数字不是结论

同一候选在 4–8 个"等价"配置下横跨一个数量级乃至 ∞（见 `metrics.json`：
`be|btc_bh` 等为 `∞`，有限的最大 3.22x）。**本表所有数字都是"某一朵噪声"下的值，
只有符号与量级可读。**

本次有 **6 个 run 因 deadline 未起跑**（全部是 `full|*|k05_head`）
⇒ 三标的池上"旧参数 vs 新参数"的对照缺一列。
