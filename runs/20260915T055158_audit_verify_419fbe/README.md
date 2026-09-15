# ⚠️ 本目录的 ② 组数字被**脚本自身 bug** 污染 —— 请用重跑的那份

**日期**：2026-09-15
**脚本**：`scripts/audit_verify.py`（当时的版本）

## 哪里错了

② 组（"引擎是否结算资金费"）原本对每个策略连跑两次（`funding OFF` 然后 `ON`），
但**复用了同一个 agent 实例**：

```python
for name, ag in cases:          # ← ag 是实例，不是工厂
    off = go(fr_3, ag, funding_arg=None)
    on  = go(fr_3, ag, funding_arg=FUNDING)
```

`HedgeEnsemble` **是有状态的**：它维护专家权重 `self.p`、`self.hold`、`log`
等跨 bar 累积的状态，且 `decide()` 有权重更新。因此第二次运行不是从初始权重出发，
而是从第一次运行结束时的权重出发 ⇒ **两条曲线不是同一策略**。

## 影响范围（哪些数字可信）

| 组 | 是否受影响 | 原因 |
|---|---|---|
| ① 清算事件与阈值 | **不受影响** | 纯数据与解析式，不涉及 agent |
| ② 引擎 funding（**学习体那两行**） | **受影响** | 复用了学习体实例 |
| ② 引擎 funding（被动等权 / 只持 BTC 两行） | 不受影响 | `FixedW` 无状态 |
| ③ funding 摆幅判据 | **不受影响** | 每次调用都新建实例 |
| ④ 窗口敏感性 | **不受影响** | 每次调用都新建实例 |

所以受污染的只有 **② 组里学习体的 `funding OFF` / `funding ON` 两个值**。

## 修法

改成**工厂**，每次运行新建实例：

```python
cases = [("学习体", lambda: HedgeEnsemble(...)), ...]
off = go(fr_3, mk_agent(), funding_arg=None)
on  = go(fr_3, mk_agent(), funding_arg=FUNDING)
```

重跑产物见同级的 `runs/<timestamp>_audit_verify_*/`（取时间戳较晚的那个）。
`WORK_LOG.md` §17.2 的数字应以重跑那份为准。

## 为什么留着不删

这条错误本身是证据：**"有状态对象被跨实验复用"是隐蔽类错误** ——
它不报错、不崩、数字看起来完全正常，只是悄悄换成了另一条策略路径。
留着它可以让下一个人照着这个模式检查自己的对照实验。
