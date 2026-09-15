"""实验登记 —— 每次实验一个 run 目录，带 config 哈希与数据版本哈希。

铁律落实：
  - 铁律 3（试验计数）：每次评估都登记，供 DSR/PBO 计算试验次数。
  - 铁律 5（事前登记）：run 创建后先写 hypothesis（假设 + 判定阈值），再跑。
  - 铁律 6（多种子）：seeds 从 config 读取，登记时原样记录。
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import time
import uuid
from datetime import datetime, timezone

from ..data import store


def _canonical(obj) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def config_hash(cfg: dict) -> str:
    return hashlib.sha256(_canonical(cfg).encode()).hexdigest()[:16]


def _git_head() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             cwd=store.project_root(), capture_output=True,
                             text=True, timeout=5)
        return out.stdout.strip() or "no-commit"
    except Exception:
        return "no-git"


class Run:
    """一次实验的记录。用作上下文管理器。"""

    def __init__(self, name: str, cfg: dict, hypothesis: dict | None = None):
        self.name = name
        self.cfg = cfg
        self.hypothesis = hypothesis or {}
        self.id = f"{datetime.now(timezone.utc):%Y%m%dT%H%M%S}_{name}_{uuid.uuid4().hex[:6]}"
        self.dir = os.path.join(store.project_root(), "runs", self.id)
        os.makedirs(self.dir, exist_ok=True)
        self.meta = {
            "run_id": self.id,
            "name": name,
            "config_hash": config_hash(cfg),
            "git_head": _git_head(),
            "python": platform.python_version(),
            "host": platform.node(),
            "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "data_manifest": {k: v["sha256"] for k, v in store.read_manifest().items()},
            "hypothesis": self.hypothesis,
        }

    def __enter__(self) -> "Run":
        self._write("config.json", self.cfg)
        self._write("meta.json", self.meta)
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.meta["ended_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        self.meta["status"] = "failed" if exc_type else "ok"
        if exc_type:
            self.meta["error"] = f"{exc_type.__name__}: {exc}"
        self._write("meta.json", self.meta)
        return False   # 不吞异常

    def _write(self, fname: str, obj) -> None:
        with open(os.path.join(self.dir, fname), "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)

    def log(self, label: str, obj) -> None:
        """把中间结果写进 run 目录。metrics 走 metrics.json，其余按文件名。"""
        self._write(f"{label}.json" if not label.endswith(".json") else label, obj)

    def record_metrics(self, metrics: dict) -> None:
        self._write("metrics.json", metrics)

    def note(self, text: str) -> None:
        with open(os.path.join(self.dir, "notes.md"), "a", encoding="utf-8") as f:
            f.write(f"- {datetime.now(timezone.utc):%Y-%m-%dT%H:%M:%SZ} {text}\n")

    def finish(self) -> str:
        """只填结论，不改假设（铁律 5）。"""
        return self.dir


def trial_count() -> int:
    """⚠️ **不要用它做 DSR 的多重试验校正。**

    它数的是 `runs/` 下**目录的个数**（只排除 `_` 前缀），**不是**已登记的试验数：
    空目录、被放弃的 scratch、未登记 run、乃至非试验目录（如 `runs/live/`）都被计入，
    而同一个网格的归档/重跑会被重复计数。实测 2026-09-15：本函数返回 **55**，
    而真正带 `meta.json` 的登记 run 只有 **44**（虚增 20%）。

    因此它**不可复现**：同一份代码、同一份数据，今天重跑（55）与当时重跑会得到不同的
    结果，而 DSR 门槛随目录累积**单调变严**（44 → 55 门槛 +3.85%；44 → 158 +20.8%）。
    也就是说，删/归档目录这种家务操作会静默改变一个统计校正的值。

    **正确做法（替换后的用法）**：把试验数**显式声明**，见
    `configs/base.json` 的 `eval.cumulative_trials`（含口径与构成说明），
    由调用方直接传入；`scripts/m11_walkforward.py` 用 `n_trials=len(GRID)`、
    `scripts/m5_online.py` 用 `cfg["eval"]["cumulative_trials"]`。

    保留此函数仅供**诊断/审计**（看看目录卫生状况），不参与任何判决。
    """
    runs_dir = os.path.join(store.project_root(), "runs")
    if not os.path.isdir(runs_dir):
        return 0
    return sum(1 for d in os.listdir(runs_dir)
               if os.path.isdir(os.path.join(runs_dir, d)) and not d.startswith("_"))
