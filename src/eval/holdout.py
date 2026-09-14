"""样本封存守卫 —— 铁律 4 的代码级落实。

封存样本只能用一次，而且必须是**显式**的。任何隐式触碰都会抛异常，
不靠"我记得别用"。
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone


class HoldoutViolation(RuntimeError):
    """试图访问封存样本。"""


class HoldoutGuard:
    """把时间轴切成 训练段 与 封存段，封存段默认不可读。

    用法：
        g = HoldoutGuard(n=len(index), fraction=0.25)
        train = frame.iloc[g.train_slice()]          # 允许
        g.assert_clean(idx)                          # 越界即抛
    """

    def __init__(self, n: int, fraction: float = 0.25, allow: bool = False,
                 log_path: str | None = None):
        if not 0.0 <= fraction < 1.0:
            raise ValueError("fraction 必须在 [0, 1) 内")
        self.n = n
        self.fraction = fraction
        self.cut = int(n * (1.0 - fraction))
        self.allow = allow
        self.log_path = log_path
        self.accesses: list[dict] = []

    # ------------------------------------------------------------------
    def train_slice(self) -> slice:
        return slice(0, self.cut)

    def holdout_slice(self) -> slice:
        self._record("holdout_slice")
        if not self.allow:
            raise HoldoutViolation(
                f"封存样本被访问（第 {self.cut}..{self.n} 条）。"
                "只有最终评定那一次才允许，且必须显式传 allow=True。"
            )
        return slice(self.cut, self.n)

    def assert_clean(self, idx) -> None:
        """检查一组索引是否踩到封存段。"""
        if self.allow:
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
                "allow": self.allow, "access_log": self.accesses}
