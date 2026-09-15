# 产物出处说明（D 阶段 1x 网格）

**这是记账缺陷修复之后重跑的 1x 结果**（`LEVERAGE=1.0`，即 D2 的配置）。

| 产物 | 说明 |
|---|---|
| `all_configs.csv` / `summary.json` | 修复后重跑，18 组 |
| 修复**前**的同名结果 | 已归档到 `runs/d_param_search_INVALIDATED_margin_bug/`（附 README） |

## 复现

```bash
cd /home/nick/workspace/quant
python3 scripts/d_param_search.py            # 1x → 本目录
```
约 21.5 分钟（每组约 72 秒；脚本文档里曾写"3-5 分钟"，是错的）。

## 本目录的两个特殊标注

1. **`summary.json` 里有 `_repaired` 字段**：`best_by_sharpe` 原本因条件写反
   （`rows[0] if not rows else None`）恒为 `null`，已按**本次运行自己的
   `all_configs.csv`** 重新计算填回 —— 是修补，不是重新训练。
   生成代码的修复已进 `scripts/d_param_search.py`，下次运行不再需要修补。
2. **无破产组**：18 组的 `total_return` 全部 > −100%、`final_net` 全部 > 0，
   所以 §15.6 那个"爆仓记录语义"修复**不影响**本目录的任何数字
   （那个修复只在权益 ≤ 0 时触发）。

## 结论（供快速索引）

终值区间 53,679 ~ 200,595（本金 10,000）；sharpe 0.627 ~ 0.838；
**最大回撤 −75.5% ~ −92.0%**（全部 18 组）。

**不能**拿它去比 `equal_weight_buyhold` 的 1,584,313 —— 那个基准零成本且不再平衡，
而本策略每根 bar 都在按 band 再平衡并付成本；而且该基准 95.6% 的收益来自 BNB。
同口径对照见 `scripts/d_fair_comparison.py` 与 `WORK_LOG.md` §15.5。
