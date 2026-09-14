"""样本封存守卫 —— 铁律 4 的代码级落实。

封存样本只能用一次，而且必须是**显式**的。任何隐式触碰都会抛异常，
不靠"我记得别用"。

## 两档接口

| 接口 | 约束 | 用途 |
|---|---|---|
| `holdout_slice()` | 需 `allow=True`；**不查账本** | 裸接口，仅测试/调试 |
| `unseal(purpose)`   | 需显式 purpose；**查账本，一次性** | 正式评定，唯一许可路径 |

`unseal()` 会把开封写进**账本**（默认 `runs/_holdout_ledger.json`，JSONL 追加）。
第二个进程/脚本再想开封同一个封存段时，会抛 `HoldoutAlreadyUnsealed` ——
除非调用方给出非空的 `override_reason`，而那条理由也会被记进账本。

**为什么要有账本**：G3 之前，"只能开一次"只靠脚本自觉，实测开封发生了 3 次
（每个标的一次）而无人拦。自觉不是机制。详见 PLAN §15.6。
"""
from __future__ import annotations

import json
import os
import subprocess
from datetime import datetime, timezone

DEFAULT_LEDGER = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "runs", "_holdout_ledger.json")
"""开封账本路径 —— 同样锚定项目根。

与 `src/live/paper.py::LIVE_DIR` 同一个教训：相对路径隐含假设 CWD 是项目根，
换个目录跑就会把账本写到别处（或读不到历史记录），而**账本读不到 = 一次性约束失效**。
"""


class HoldoutViolation(RuntimeError):
    """试图访问封存样本。"""


class HoldoutAlreadyUnsealed(RuntimeError):
    """封存样本此前已被开封过，且本次未给出显式覆盖理由。"""


def _git_head() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, timeout=5)
        return out.stdout.strip() or "no-commit"
    except Exception:
        return "no-git"


class HoldoutGuard:
    """把时间轴切成 训练段 与 封存段，封存段默认不可读。

    用法（正式路径）：
        g = HoldoutGuard(n=len(index), fraction=0.25)
        train = frame.iloc[g.train_slice()]
        g.assert_clean(idx)                       # 越界即抛
        ...
        hs = g.unseal("G3 最终评定")               # 一次性，写账本
        r = CarrySimulator(spots.iloc[hs], ...).run()
    """

    def __init__(self, n: int, fraction: float = 0.25, allow: bool = False,
                 log_path: str | None = None,
                 ledger_path: str | None = DEFAULT_LEDGER,
                 seal_id: str | None = None):
        if not 0.0 <= fraction < 1.0:
            raise ValueError("fraction 必须在 [0, 1) 内")
        self.n = n
        self.fraction = fraction
        self.cut = int(n * (1.0 - fraction))
        self.allow = allow
        self.log_path = log_path
        self.ledger_path = ledger_path
        self.seal_id = seal_id or f"n{n}_cut{self.cut}_f{fraction:g}"
        self.accesses: list[dict] = []
        self._unsealed = False          # 本守卫实例是否已开封
        self._purpose: str | None = None

    # ------------------------------------------------------------------
    def train_slice(self) -> slice:
        return slice(0, self.cut)

    def holdout_slice(self) -> slice:
        """裸接口：需 allow=True，**不查账本**。正式评定请用 `unseal()`。"""
        self._record("holdout_slice")
        if not self.allow and not self._unsealed:
            raise HoldoutViolation(
                f"封存样本被访问（第 {self.cut}..{self.n} 条）。"
                "正式评定请用 unseal(purpose)，测试/调试请显式传 allow=True。"
            )
        return slice(self.cut, self.n)

    def unseal(self, purpose: str, override_reason: str | None = None) -> slice:
        """**一次性开封**封存段，并把开封写进账本。

        - 同一个守卫实例首次开封后，可重复调用（视为同一次评估事件的多处读取）
        - 不同实例/不同进程再开封同一个 `seal_id` ⇒ 抛 `HoldoutAlreadyUnsealed`
        - 确实必须二次开封时，传非空 `override_reason`，它会一起进账本
        """
        if not purpose or not str(purpose).strip():
            raise ValueError("开封必须写明 purpose，空字符串不接受")
        if not self._unsealed:
            prior = self._ledger_entries(self.seal_id)
            if prior and not (override_reason and override_reason.strip()):
                first = prior[0]
                raise HoldoutAlreadyUnsealed(
                    f"封存段 {self.seal_id} 已于 {first.get('at')} 被开封"
                    f"（purpose={first.get('purpose')!r}，git={first.get('git_head')}），"
                    f"账本共 {len(prior)} 条记录：{self.ledger_path}。"
                    "若确需二次开封，显式传 override_reason 说明理由（会记入账本）。"
                )
            entry = {"at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                     "seal_id": self.seal_id, "purpose": str(purpose),
                     "n": self.n, "fraction": self.fraction, "cut": self.cut,
                     "git_head": _git_head(),
                     "override": bool(prior),
                     "override_reason": override_reason or None,
                     "prior_unseals": len(prior)}
            self._write_ledger(entry)
            self._record("unseal", {"purpose": purpose, "override": bool(prior)})
            self._unsealed = True
            self._purpose = str(purpose)
        return slice(self.cut, self.n)

    # ------------------------------------------------------------------
    def _ledger_entries(self, seal_id: str) -> list[dict]:
        if not self.ledger_path or not os.path.exists(self.ledger_path):
            return []
        out = []
        with open(self.ledger_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    e = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if e.get("seal_id") == seal_id:
                    out.append(e)
        return out

    def _write_ledger(self, entry: dict) -> None:
        if not self.ledger_path:
            return
        d = os.path.dirname(self.ledger_path)
        if d:
            os.makedirs(d, exist_ok=True)
        with open(self.ledger_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def assert_clean(self, idx) -> None:
        """检查一组索引是否踩到封存段。"""
        if self.allow or self._unsealed:
            return
        arr = list(idx) if not isinstance(idx, int) else [idx]
        bad = [i for i in arr if i >= self.cut]
        if bad:
            self._record("violation", bad[:10])
            raise HoldoutViolation(
                f"索引 {bad[:10]} 落在封存段（起点 {self.cut}），共 {len(bad)} 处越界。"
            )

    # ------------------------------------------------------------------
    def _record(self, kind: str, detail=None) -> None:
        entry = {"at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                 "kind": kind, "detail": detail}
        self.accesses.append(entry)
        if self.log_path:
            os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def summary(self) -> dict:
        return {"n": self.n, "holdout_fraction": self.fraction,
                "sealed_from": self.cut, "sealed_count": self.n - self.cut,
                "seal_id": self.seal_id, "allow": self.allow,
                "unsealed": self._unsealed, "purpose": self._purpose,
                "ledger_path": self.ledger_path,
                "access_log": self.accesses}
