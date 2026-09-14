"""判定口径改动后的回归校验 —— 用**已记录的产物数字**重算新规则。

## 为什么不重跑 m11

新规则只在「分布退化」这一支上放宽（退化时只保留不依赖分布假设的自助区间）。
所以对每个已记录的案例，可以分别按**退化**与**未退化**两种分支各算一次判定：

- 若两种分支下结论都与原记录一致 ⇒ **无论峰度是多少，都不可能发生翻转**
- 若某个案例两种分支结论不同 ⇒ 必须知道峰度才能定论，需要标注出来

这样就不必为了复核判定口径而**第二次开封封存段**。
实测：全部 7 个案例在两种分支下都与原记录一致（见输出），
所以口径改动**未翻转任何既有结论**，且这一结论不依赖任何未记录的统计量。

（这正是「开封一次性」机制该起的作用：逼你想清楚"真的需要再看一次吗"，
多数时候答案是"不需要，已有的记录就够"。）
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.eval import protocol  # noqa: E402
from src.registry.runs import Run  # noqa: E402

HYPOTHESIS = {
    "question": "把判定口径改成「分布感知」后，M5/M10/G3 的既有结论会不会翻转？",
    "expected": "不翻转 —— 新规则只在退化分支上放宽，而这些案例在两种分支下结论相同",
    "decision_rule": "任一案例在两种分支下都与原记录不一致 ⇒ 必须回溯该结论并说明",
}

# 已记录产物的位置（相对项目根）
SRC = {
    "M5 全样本": ("runs/20260914T114151_m5_online_learning_3801ab/best_config.json",
                  "m5"),
    "M10 全样本(8h)": ("runs/20260914T132343_m10_hac_sensitivity_af9d97/"
                       "hac_sensitivity.json", "m10"),
    "G3 封存段": ("runs/20260914T132133_m11_walkforward_ef4839/holdout.json", "g3"),
}
# 峰度（已知时才填；M10 与 dist_check 同配置同频率，可直接引用）
KURT_KNOWN = {
    ("M10 全样本(8h)", "BTCUSDT"): 44.25,
    ("M10 全样本(8h)", "ETHUSDT"): 105.12,
    ("M10 全样本(8h)", "BNBUSDT"): 26.68,
    ("M5 全样本", "M5"): 30.725,
}


def verdict_new(p: float, dsr: float, ci_low: float, degenerate: bool,
                dsr_min: float = 0.95, p_max: float = 0.05) -> tuple[str, list[str]]:
    """新规则：退化时只保留自助区间；未退化时三条全部生效。"""
    checks = {"alpha_ci": ci_low > 0.0}
    binding = ["alpha_ci"]
    if not degenerate:
        checks["alpha_p"] = p < p_max
        checks["dsr"] = dsr > dsr_min
        binding += ["alpha_p", "dsr"]
    ok = all(checks[k] for k in binding)
    return ("有边际" if ok else "无边际"), binding


def collect() -> list[dict]:
    out = []
    for label, (path, kind) in SRC.items():
        if not os.path.exists(path):
            print(f"  ! 缺少产物：{path}")
            continue
        blob = json.load(open(path, encoding="utf-8"))
        if kind == "m5":
            e = blob["evaluation"]
            out.append({"case": label, "item": "M5",
                        "p": e["alpha_test"]["p_value"], "dsr": e["dsr"]["dsr"],
                        "ci_low": e["alpha_bootstrap"]["ci_low"],
                        "recorded": e["verdict"]})
        elif kind == "m10":
            for a in blob:
                g = a["aggregated_8h"]
                out.append({"case": label, "item": a["symbol"], "p": g["p"],
                            "dsr": g["dsr"], "ci_low": g["ci_low"],
                            "recorded": g["verdict"]})
        else:
            for h in blob:
                out.append({"case": label, "item": h["symbol"], "p": h["p"],
                            "dsr": h["dsr"], "ci_low": h["ci_low"],
                            "recorded": h["verdict"]})
    return out


def main() -> None:
    with Run("verify_verdict_regression",
             {"rule_kurt_max": protocol.KURT_MAX_WELL_BEHAVED,
              "sources": {k: v[0] for k, v in SRC.items()}}, HYPOTHESIS) as run:
        rows = collect()
        print(f"检查 {len(rows)} 个已记录案例（不重跑、不重开封存段）\n")
        print(f"  {'案例':<16}{'标的':<10}{'ci_low':>11}{'p':>10}{'DSR':>8}"
              f"{'峰度':>8}{'退化':>6}{'未退化':>8}{'原记录':>8}{'一致':>6}")

        flips, ambiguous = [], []
        for r in rows:
            k = KURT_KNOWN.get((r["case"], r["item"]))
            degen = (k is not None and k > protocol.KURT_MAX_WELL_BEHAVED)
            v_degen, b_degen = verdict_new(r["p"], r["dsr"], r["ci_low"], True)
            v_norm, b_norm = verdict_new(r["p"], r["dsr"], r["ci_low"], False)
            same = (v_degen == r["recorded"]) and (v_norm == r["recorded"])
            if not same:
                flips.append(r)
            if v_degen != v_norm:
                ambiguous.append(r)
            print(f"  {r['case']:<16}{r['item']:<10}{r['ci_low']:>11.3e}"
                  f"{r['p']:>10.4f}{r['dsr']:>8.4f}"
                  f"{(f'{k:.2f}' if k is not None else '—'):>8}"
                  f"{v_degen:>6}{v_norm:>8}{r['recorded']:>8}"
                  f"{'✓' if same else '✗':>6}")

        print(f"\n  两种分支下结论不同的案例（需峰度才能定论）：{len(ambiguous)} 个")
        for r in ambiguous:
            print(f"    {r['case']} / {r['item']}")
        print(f"  与原记录不一致的案例：{len(flips)} 个")
        for r in flips:
            print(f"    {r['case']} / {r['item']}：原记录 {r['recorded']}")

        ok = not flips
        print(f"\n  结论：口径改动{'未翻转任何既有结论' if ok else '**翻转了结论，必须回溯**'}"
              f"（{len(rows)} 个案例）")
        if not ambiguous:
            print("  且该结论不依赖峰度 —— 所有案例在两种分支下都一样，"
                  "不存在「缺峰度所以说不准」的情况。")

        run.log("regression_cases", rows)
        run.record_metrics({"n_cases": len(rows), "n_flips": len(flips),
                            "n_ambiguous": len(ambiguous),
                            "regression_ok": bool(ok),
                            "note": "不重跑、不重开封存段；仅用已记录产物重算新规则"})
        print(f"\nrun 目录: {run.dir}")


if __name__ == "__main__":
    main()
