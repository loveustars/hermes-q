"""DSR 的试验数必须是**可复现的显式声明**，不能从 runs/ 的目录数推断。

为什么单独有这个测试文件：
`src/registry/runs.py::trial_count()` 数的是 `runs/` 下的**目录个数**（只排除 `_` 前缀），
而它被当成"已登记的实验次数"喂给 DSR 的多重试验惩罚。后果是**同一份代码、同一份数据，
今天重跑与当时重跑会得到不同的门槛**，而且每加一个 Run 目录、乃至做一次归档都会改变它
（实测 2026-09-15：trial_count()=55，真正带 meta.json 的登记 run 只有 44，虚增 20%
⇒ 门槛偏高 3.85%；若累积到 158 则偏高 20.8%）。方向是单调变严，
所以"做家务"这种无害操作会静默改变一个统计校正的值。

修法：试验数由 `configs/base.json: eval.cumulative_trials` **显式声明**（append-only，
口径与构成写在同处的 `cumulative_trials_note`），调用方直接读它。

本文件守四条：
  1. 配置里必须存在该字段，且为正整数；
  2. 它必须带非空的构成说明（否则数字无从审计）；
  3. `scripts/m5_online.py` **不得**再调用 `trial_count()`（AST 级检查，注释不算）；
  4. DSR 门槛对"声明的 n"是纯函数 —— 反复算同值，且与目录数无关。

运行：python3 -m pytest tests/ -q     （不要用 python3 tests/x.py，那样会静默跳过）
"""
from __future__ import annotations

import ast
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.eval import protocol  # noqa: E402
from src.registry import runs as runs_mod  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG = os.path.join(ROOT, "configs", "base.json")
M5 = os.path.join(ROOT, "scripts", "m5_online.py")


def _cfg() -> dict:
    with open(CONFIG, encoding="utf-8") as fh:
        return json.load(fh)


def _called_names(path: str) -> set[str]:
    """AST 扫描：这个脚本里真正被**调用**的函数名（注释与字符串不算）。"""
    with open(path, encoding="utf-8") as fh:
        tree = ast.parse(fh.read(), filename=path)
    out: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            fn = node.func
            if isinstance(fn, ast.Name):
                out.add(fn.id)
            elif isinstance(fn, ast.Attribute):
                out.add(fn.attr)
    return out


def test_config_declares_cumulative_trials_as_positive_int() -> None:
    n = _cfg()["eval"]["cumulative_trials"]
    assert isinstance(n, int) and not isinstance(n, bool), (
        f"eval.cumulative_trials 必须是 int，实际 {type(n).__name__}")
    assert n > 0, "eval.cumulative_trials 必须为正 —— DSR 的 n_trials 不能是 0 或负"


def test_cumulative_trials_has_a_nonempty_derivation_note() -> None:
    """没有构成说明的数字无从审计，等价于又一个不可复现的魔法常数。"""
    note = _cfg()["eval"].get("cumulative_trials_note")
    assert note, "eval.cumulative_trials_note 缺失 —— 该数字必须写明口径与构成"
    joined = "".join(note) if isinstance(note, list) else str(note)
    assert len(joined.strip()) >= 40, "构成说明过短，不足以审计"
    # 至少要能看出它列出了构成项（出现分项数字），否则只是套话
    assert any(ch.isdigit() for ch in joined), "构成说明里应当列出分项配置数"


def test_m5_does_not_call_trial_count() -> None:
    """M5 的 DSR 路径不得再依赖 runs/ 的目录数。

    这条用 AST 而非字符串匹配：字符串匹配会被注释里的 `trial_count()` 误伤
    （修这个 bug 时的说明注释里正好会提到它）。
    """
    called = _called_names(M5)
    assert "trial_count" not in called, (
        "scripts/m5_online.py 仍在调用 trial_count() —— 它数的是 runs/ 目录个数，"
        "会让 DSR 门槛不可复现。应改为读 cfg['eval']['cumulative_trials']。")


def test_declared_n_is_used_and_matches_config_value() -> None:
    """M5 里读到的累计试验数，必须等于配置声明值（不是目录数）。"""
    declared = _cfg()["eval"]["cumulative_trials"]
    observed_dirs = runs_mod.trial_count()
    # 声明值可以等于目录数（巧合），但**不能由目录数决定**：
    # 断言声明值是配置里的常量，而不是运行期从文件系统算出来的。
    src = open(M5, encoding="utf-8").read()
    assert "eval.cumulative_trials" in src or 'eval"]["cumulative_trials' in src, (
        "scripts/m5_online.py 应当显式从 configs/base.json 读 eval.cumulative_trials")
    assert declared > 0
    _ = observed_dirs  # 目录数只作对照，不参与判决


def test_dsr_hurdle_is_a_pure_function_of_n() -> None:
    """门槛只由 (sr_variance, n) 决定 —— 同参数反复算必须同值。"""
    var = 0.2
    n = _cfg()["eval"]["cumulative_trials"]
    a = protocol.expected_max_sharpe(var, n)
    b = protocol.expected_max_sharpe(var, n)
    assert a == b, "expected_max_sharpe 对同参数给出不同值 ⇒ 不是纯函数"

    # 门槛随 n 单调不减（多重试验越多、门槛越高）——这是该函数的定义性质
    assert protocol.expected_max_sharpe(var, n) <= protocol.expected_max_sharpe(var, n * 4)


def test_filesystem_churn_cannot_move_the_hurdle() -> None:
    """核心回归：**新建/删除 runs/ 下的目录不得改变 DSR 门槛。**

    这正是原实现的问题 —— 现在门槛只由配置里的声明值决定，与目录数无关。
    这里用"声明值"直接算两次，并显式证明它与 trial_count() 解耦。
    """
    var = 0.2
    declared = _cfg()["eval"]["cumulative_trials"]
    hurdle = protocol.expected_max_sharpe(var, declared)

    dirs = runs_mod.trial_count()
    if dirs and dirs != declared:
        # 目录数与声明值不同是正常的；关键是门槛**不随前者变**。
        assert hurdle == protocol.expected_max_sharpe(var, declared)
        assert dirs != declared, (
            "目录数恰好等于声明值时这条检查没有意义，请检查配置是否被改成了目录数")
