"""把机制建立**之前**的历史开封补录进账本（幂等）。

## 为什么需要这个

`unseal()` + 账本的机制是今天才加的。在此之前，G3 用裸接口
`holdout_slice()` 开封，**没有留下任何账本记录**。
如果不管，账本就会漏掉历史使用、从而低估真实开封次数 ——
下一个调用方会误以为"封存段还没被用过"，这正好是账本想防的事。

所以要把历史补进去。记录内容全部来自可核查的既有文件：

- `runs/_holdout_access.log`：逐次访问的时间戳
- `runs/20260914T132133_m11_walkforward_ef4839/holdout_guard.json`：
  成功那次的 3 次访问

实测：封存段共被访问 5 次 ——
- 成功那次（G3 正式评定）3 次，13:21:58 / 13:22:03 / 13:22:07
- 之前两次**失败的**运行各 1 次（分别因 HoldoutViolation 与 KeyError 中断），
  13:20:21 / 13:21:13 —— 这两次没有产出任何结论，但**访问是事实**，照样记。

## 幂等

`seal_id` 已有记录时直接退出，不会重复追加。
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.eval.holdout import DEFAULT_LEDGER, HoldoutGuard  # noqa: E402

# 历史事实（来源：runs/_holdout_access.log 与 G3 的 run 产物）
N = 57_765
FRACTION = 0.25
SUCCESS_RUN = "runs/20260914T132133_m11_walkforward_ef4839"
ACCESS_LOG = "runs/_holdout_access.log"


def main() -> None:
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(root)

    g = HoldoutGuard(n=N, fraction=FRACTION)
    seal_id = g.seal_id

    # 读访问日志，得到历史访问时间戳
    stamps = []
    if os.path.exists(ACCESS_LOG):
        with open(ACCESS_LOG, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        stamps.append(json.loads(line)["at"])
                    except (json.JSONDecodeError, KeyError):
                        pass
    print(f"seal_id = {seal_id}")
    print(f"历史访问 {len(stamps)} 次：{stamps}")

    existing = g._ledger_entries(seal_id)          # noqa: SLF001 (有意复用)
    if existing:
        print(f"账本已有 {len(existing)} 条记录，跳过（幂等）")
        return

    entry = {
        "at": max(stamps) if stamps else "unknown",
        "seal_id": seal_id,
        "purpose": "G3 封存段最终评定（**历史补录**：机制建立之前的开封，见下方 provenance）",
        "n": N,
        "fraction": FRACTION,
        "cut": g.cut,
        "git_head": "pre-ad39459(~252a76f)",
        "override": False,
        "override_reason": None,
        "prior_unseals": 0,
        "backfilled": True,
        "backfill_note": (
            "本条目由 scripts/seed_holdout_ledger.py 补录。当时用的是裸接口 "
            "holdout_slice()（无账本），访问次数来自 runs/_holdout_access.log。"
            "其中 3 次属成功运行（G3 正式评定），2 次属失败运行（无结论产出但访问是事实）。"
        ),
        "provenance": {
            "access_log": ACCESS_LOG,
            "success_run": SUCCESS_RUN,
            "access_timestamps": stamps,
            "n_accesses": len(stamps),
        },
    }
    g._write_ledger(entry)                          # noqa: SLF001
    print(f"已补录 1 条到 {DEFAULT_LEDGER}")
    print("→ 效果：下次任何人想再开这段封存段，都必须显式给 override_reason。")


if __name__ == "__main__":
    main()
