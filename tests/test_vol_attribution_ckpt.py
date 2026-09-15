"""检查点(schema)守卫：缺字段的旧记录必须被**丢弃并重跑**，不得静默沿用。

出问题的机制（2026-09-15 实测）：
`vol_attribution.py` 用 `--ckpt` 断点续跑；旧版本写出的记录里没有 `funding_paid`。
读取端为了防御用了 `rec.get("funding_paid", 0.0)`，于是"**字段不存在**"被变成了
"**资金费恰好是 0.0**"，在归因表里与"这个策略真的没付过资金费"无法区分。
实测后果：用旧 ckpt 恢复跑出的表里，`btc_bh` 的 `funding_pct_of_initial` 显示 **0.0**，
而同池同策略的真实值约 **289%**；`G_fund` 退化成 `nan`。

⇒ 防御式默认值把"缺失"变成了"零"，这是**静默错误**，比崩溃更危险。
正确做法是让 schema 不匹配的记录**失效并重跑**（`drop_stale_ckpt_records`）。

另：该脚本底部有 `if __name__ == "__main__":` 守卫，所以可以安全导入 —— 本文件依赖这一点，
若守卫被移除，导入会直接跑 `main()`（会解析 sys.argv 并开跑实验），故一并设测试守住。

运行：python3 -m pytest tests/ -q     （不要用 python3 tests/x.py，那样会静默跳过）
"""
from __future__ import annotations

import ast
import importlib.util
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(ROOT, "scripts", "vol_attribution.py")


def _load_module():
    """按路径导入脚本（scripts/ 不在包路径里）。"""
    spec = importlib.util.spec_from_file_location("_va_under_test", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


VA = _load_module()


def _rec(**over) -> dict:
    """一条符合当前 schema 的记录。"""
    base = {f: 0.0 for f in VA.CKPT_REQUIRED}
    base.update({"final_net": 1.0, "final_gross": 1.0, "cost_paid": 0.0})
    base.update(over)
    return base


def test_script_has_main_guard_so_it_is_importable() -> None:
    """没有 __main__ 守卫的话，上面的 _load_module() 会直接把实验跑起来。"""
    src = open(SCRIPT, encoding="utf-8").read()
    assert 'if __name__ == "__main__":' in src, (
        "scripts/vol_attribution.py 缺少 __main__ 守卫 —— 导入即执行 main()，无法单测")


def test_required_fields_include_the_one_that_caused_silent_zero() -> None:
    """`funding_paid` 必须在必需字段里 —— 它的缺失正是静默 0.0 的来源。"""
    assert "funding_paid" in VA.CKPT_REQUIRED, (
        "CKPT_REQUIRED 未包含 funding_paid：缺该字段的记录经 .get(...,0.0) 会变成"
        "『资金费 0.0』，与真实为零无法区分")


def test_stale_records_are_dropped() -> None:
    """缺任一必需字段 ⇒ 该条被丢弃（并返回键名供报告）。"""
    ckpt = {
        "be|exp:long_all|A_head": _rec(),
        "be|btc_bh|A_head": _rec(),           # 旧记录：缺 funding_paid
        "be|ew_rebal|C_nofund": _rec(),
    }
    del ckpt["be|btc_bh|A_head"]["funding_paid"]

    stale = VA.drop_stale_ckpt_records(ckpt)

    assert stale == ["be|btc_bh|A_head"], f"应只丢弃那一条，实际 {stale}"
    assert "be|btc_bh|A_head" not in ckpt, "旧记录仍留在 ckpt 里 ⇒ 会被静默沿用"
    assert len(ckpt) == 2, "合规记录不应被误删"


def test_current_schema_records_survive() -> None:
    """字段齐全的记录必须原样保留（守卫不能过度丢弃、否则每次续跑都白跑）。"""
    ckpt = {f"be|cand{i}|A_head": _rec() for i in range(5)}
    stale = VA.drop_stale_ckpt_records(ckpt)
    assert stale == [], f"合规记录被误判为旧记录：{stale}"
    assert len(ckpt) == 5


def test_schema_marker_is_never_dropped() -> None:
    """`__schema__` 标记本身不是记录，不能被当成旧记录删掉。"""
    ckpt = {VA.CKPT_SCHEMA_KEY: VA.CKPT_SCHEMA, "be|exp:flat|A_head": _rec()}
    stale = VA.drop_stale_ckpt_records(ckpt)
    assert VA.CKPT_SCHEMA_KEY in ckpt, "schema 标记被误删"
    assert stale == [], f"不应丢弃任何东西，实际 {stale}"


def test_non_dict_values_do_not_crash() -> None:
    """检查点里混入非 dict 值（手工编辑/损坏）时不得崩，且不能误删合规记录。"""
    ckpt = {"be|exp:flat|A_head": _rec(), "junk": 42, "junk2": None}
    stale = VA.drop_stale_ckpt_records(ckpt)   # 不应抛异常
    assert "be|exp:flat|A_head" in ckpt, "合规记录被误删"
    assert isinstance(stale, list)


def test_guard_is_actually_wired_into_main() -> None:
    """守卫必须真的接在续跑路径上 —— 写了函数不用等于没修。"""
    tree = ast.parse(open(SCRIPT, encoding="utf-8").read(), filename=SCRIPT)
    called: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            fn = node.func
            if isinstance(fn, ast.Name):
                called.add(fn.id)
            elif isinstance(fn, ast.Attribute):
                called.add(fn.attr)
    assert "drop_stale_ckpt_records" in called, (
        "drop_stale_ckpt_records 定义了但没有被调用 ⇒ 旧 ckpt 仍会被静默沿用")


def test_shipped_old_ckpt_would_be_dropped_if_present() -> None:
    """回归实证：若环境里还留着那次真实的旧检查点，它必须被整批丢弃。

    这是把这个 bug 现场（`/tmp/vol_ckpt.json`，64 条、无 `funding_paid`）固化下来的检查。
    文件不存在时跳过（CI/换机场景），存在时必须 100% 被判为旧记录。
    """
    import json

    path = "/tmp/vol_ckpt.json"
    if not os.path.exists(path):
        return
    try:
        with open(path, encoding="utf-8") as fh:
            ckpt = json.load(fh)
    except Exception:
        return
    if not isinstance(ckpt, dict) or not ckpt:
        return
    recs = {k: v for k, v in ckpt.items() if isinstance(v, dict)}
    if not any("funding_paid" not in v for v in recs.values()):
        return  # 已经是新 schema，跳过
    stale = VA.drop_stale_ckpt_records(ckpt)
    assert set(stale) == set(recs), (
        f"旧检查点里 {len(recs)} 条记录本应全部失效，实际只丢弃 {len(stale)} 条")
