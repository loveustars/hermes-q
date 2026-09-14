"""评估协议的自洽性测试。

最重要的一条：**拿基准对基准做回归，beta 必须等于 1、alpha 必须等于 0。**
这个测试若早先存在，就能立刻抓住"策略收益与基准按长度截断对齐"导致的错位 bug
——当时基准自比基准算出 beta=0.003、alpha=+162%/年。

运行：python3 tests/test_protocol.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.eval import protocol  # noqa: E402
from src.eval.holdout import HoldoutGuard, HoldoutViolation  # noqa: E402

BPY = 24 * 365


# ==========================================================================
# 1. 基准自回归 —— 最关键的健全性检查
# ==========================================================================
def test_benchmark_vs_itself_is_identity():
    rng = np.random.default_rng(0)
    idx = pd.date_range("2022-01-01", periods=3000, freq="h", tz="UTC")
    b = rng.normal(0.0002, 0.01, len(idx))
    at = protocol.alpha_vs_benchmark(b, b, BPY, r_index=idx, b_index=idx)
    assert abs(at.beta - 1.0) < 1e-6, f"基准自比基准 beta 应为 1，实际 {at.beta}"
    assert abs(at.alpha) < 1e-9, f"基准自比基准 alpha 应为 0，实际 {at.alpha}"
    assert at.r_squared > 0.999, f"R² 应接近 1，实际 {at.r_squared}"
    # alpha 恰为 0 时，单边 p 值就是 0.5 —— 恰好不显著
    assert abs(at.p_value - 0.5) < 1e-9, f"alpha=0 时单边 p 应为 0.5，实际 {at.p_value}"


def test_alignment_is_by_timestamp_not_length():
    """错位必须被对齐修正，而不是按长度截断。"""
    rng = np.random.default_rng(1)
    idx = pd.date_range("2022-01-01", periods=3000, freq="h", tz="UTC")
    b = rng.normal(0.0002, 0.01, len(idx))

    # 策略收益从第 301 根才开始（模拟 warmup）
    r = b[300:]
    r_idx = idx[300:]
    at = protocol.alpha_vs_benchmark(r, b, BPY, r_index=r_idx, b_index=idx)
    assert abs(at.beta - 1.0) < 1e-6, f"对齐后 beta 应为 1，实际 {at.beta}"
    assert abs(at.alpha) < 1e-9
    assert at.n == len(r), f"对齐后应剩 {len(r)} 个观测，实际 {at.n}"

    # 反例：不传 index 时按长度截断，必然错位 —— 这里确认差异确实存在
    at_bad = protocol.alpha_vs_benchmark(r, b, BPY)
    assert abs(at_bad.beta - 1.0) > 0.1, "错位情况下 beta 不该接近 1"


def test_identical_series_with_different_lengths_aligns_correctly():
    rng = np.random.default_rng(2)
    idx = pd.date_range("2020-06-01", periods=5000, freq="h", tz="UTC")
    b = rng.normal(0.0, 0.012, len(idx))
    r = b[1000:] * 1.0
    at = protocol.alpha_vs_benchmark(r, b, BPY, r_index=idx[1000:], b_index=idx)
    assert abs(at.beta - 1.0) < 1e-6
    assert abs(at.alpha) < 1e-9


# ==========================================================================
# 2. 正态分布工具
# ==========================================================================
def test_norm_functions_are_exact_inverses():
    for p in [1e-6, 0.001, 0.025, 0.5, 0.84, 0.975, 0.999, 1 - 1e-6]:
        assert abs(protocol.norm_cdf(protocol.norm_ppf(p)) - p) < 1e-9, f"p={p} 不闭合"
    assert abs(protocol.norm_ppf(0.975) - 1.959964) < 1e-5
    assert abs(protocol.norm_cdf(0.0) - 0.5) < 1e-12


# ==========================================================================
# 3. DSR：试验次数越多，惩罚越重
# ==========================================================================
def test_dsr_penalizes_more_trials():
    rng = np.random.default_rng(3)
    r = rng.normal(0.0004, 0.01, 5000)     # 有点正向漂移
    d1 = protocol.deflated_sharpe(r, n_trials=1)
    d10 = protocol.deflated_sharpe(r, n_trials=10)
    d1000 = protocol.deflated_sharpe(r, n_trials=1000)
    assert d1["dsr"] > d10["dsr"] > d1000["dsr"], "DSR 必须随试验次数单调下降"
    assert d1000["dsr"] < d1["dsr"]


def test_dsr_near_half_for_zero_skill():
    rng = np.random.default_rng(4)
    vals = []
    for i in range(50):
        r = rng.normal(0.0, 0.01, 3000)
        vals.append(protocol.deflated_sharpe(r, n_trials=1)["dsr"])
    assert 0.3 < np.mean(vals) < 0.7, f"零技能下 DSR 应中心在 0.5 附近，实际 {np.mean(vals):.3f}"


# ==========================================================================
# 4. Bootstrap
# ==========================================================================
def test_bootstrap_detects_positive_mean():
    rng = np.random.default_rng(5)
    r = rng.normal(0.001, 0.01, 4000)
    out = protocol.bootstrap_p_value(r, block=24)
    assert out["p_value"] < 0.05, "明显正均值应被检出"
    assert out["ci_low"] > 0


def test_bootstrap_no_false_alarm_on_zero_mean():
    rng = np.random.default_rng(6)
    r = rng.normal(0.0, 0.01, 4000)
    out = protocol.bootstrap_p_value(r, block=24)
    assert out["p_value"] > 0.05, "零均值不该被判为显著"


def test_bootstrap_p_value_is_fraction():
    rng = np.random.default_rng(7)
    r = rng.normal(0.0, 0.01, 1000)
    out = protocol.bootstrap_p_value(r, block=12, n_samples=500)
    assert 0.0 <= out["p_value"] <= 1.0


# ==========================================================================
# 5. PBO
# ==========================================================================
def test_pbo_near_half_for_pure_noise():
    """纯噪声下，样本内最优在样本外应无优势，PBO 接近 0.5。"""
    rng = np.random.default_rng(8)
    m = rng.normal(0.0, 0.01, size=(4000, 20))
    out = protocol.pbo_cscv(m, n_blocks=10)
    assert out["pbo"] is not None
    assert 0.25 < out["pbo"] < 0.75, f"噪声 PBO 应接近 0.5，实际 {out['pbo']}"


def test_pbo_low_when_one_strategy_is_genuinely_better():
    rng = np.random.default_rng(9)
    m = rng.normal(0.0, 0.01, size=(4000, 20))
    m[:, 0] += 0.0015                      # 第 0 列有真实优势
    out = protocol.pbo_cscv(m, n_blocks=10)
    assert out["pbo"] < 0.25, f"存在真优势时 PBO 应偏低，实际 {out['pbo']}"


def test_bootstrap_alpha_detects_real_alpha():
    """有明显 alpha 时，截距的置信区间下界必须为正。"""
    rng = np.random.default_rng(11)
    n = 20_000
    b = rng.normal(0.0001, 0.01, n)
    r = 0.0006 + 0.9 * b + rng.normal(0.0, 0.002, n)
    out = protocol.bootstrap_alpha(r, b, block=24, n_samples=600)
    at = protocol.alpha_vs_benchmark(r, b, BPY)
    assert out["ci_low"] > 0, f"真实 alpha 应被判为显著，实际 CI {out['ci_low']}~{out['ci_high']}"
    # CI 必须包含点估计
    assert out["ci_low"] <= at.alpha <= out["ci_high"], \
        f"CI 未包含点估计：alpha={at.alpha}, CI=[{out['ci_low']}, {out['ci_high']}]"
    assert out["p_value_alpha_positive"] < 0.05


def test_bootstrap_alpha_no_false_alarm_on_zero_alpha():
    rng = np.random.default_rng(12)
    n = 20_000
    b = rng.normal(0.0001, 0.01, n)
    r = 0.9 * b + rng.normal(0.0, 0.002, n)        # 无 alpha
    out = protocol.bootstrap_alpha(r, b, block=24, n_samples=600)
    assert out["ci_low"] < 0 < out["ci_high"], "零 alpha 时 CI 应跨零"


def test_evaluate_strategy_can_actually_pass():
    """回归守卫：协议必须**有可能**判出「有边际」。

    早期版本对 OLS 残差做 bootstrap 再检验均值 > 0 ——
    残差均值按构造恒为 0，第三条判据在数学上永远不可能通过，
    于是任何策略都只能得到「无边际」。这条测试确保那种失效不会回归。
    """
    rng = np.random.default_rng(13)
    n = 20_000
    idx = pd.date_range("2020-01-01", periods=n, freq="h", tz="UTC")
    b = rng.normal(0.0001, 0.01, n)
    r = 0.0008 + 0.2 * b + rng.normal(0.0, 0.0015, n)   # 强 alpha、低 beta
    out = protocol.evaluate_strategy(r, b, n_trials=1, bars_per_year=BPY,
                                     r_index=idx, b_index=idx)
    assert out["verdict"] == "有边际", (
        f"有明显 alpha 却判为无边际 —— 判定链条断了。"
        f" alpha_test={out['alpha_test']}, dsr={out['dsr']['dsr']},"
        f" boot_ci_low={out['alpha_bootstrap']['ci_low']}")
    assert out["alpha_bootstrap"]["ci_low"] > 0


def test_evaluate_strategy_rejects_pure_noise():
    rng = np.random.default_rng(14)
    n = 20_000
    idx = pd.date_range("2020-01-01", periods=n, freq="h", tz="UTC")
    b = rng.normal(0.0001, 0.01, n)
    r = 0.8 * b + rng.normal(0.0, 0.002, n)
    out = protocol.evaluate_strategy(r, b, n_trials=1, bars_per_year=BPY,
                                     r_index=idx, b_index=idx)
    assert out["verdict"] == "无边际", "纯噪声不该被判为有边际"


def test_freq_to_ns():
    assert protocol.freq_to_ns("8h") == 8 * 3_600_000_000_000
    assert protocol.freq_to_ns("30m") == 30 * 60_000_000_000
    assert protocol.freq_to_ns("1d") == 86_400_000_000_000
    assert protocol.freq_to_ns(12345) == 12345
    try:
        protocol.freq_to_ns("8w")
    except ValueError:
        pass
    else:
        raise AssertionError("无法解析的单位应抛 ValueError")


def test_aggregate_to_clock_compounds_correctly():
    """聚合必须是复利，且桶数正确。"""
    import pandas as pd
    idx = pd.date_range("2024-01-01", periods=8, freq="1h", tz="UTC")
    r = np.full(8, 0.01)
    out, ts = protocol.aggregate_to_clock(idx, r, "8h")
    assert len(out) == 1, f"8 根 1h 应合成 1 个 8h 桶，得 {len(out)}"
    assert abs(out[0] - (1.01 ** 8 - 1)) < 1e-12, "复利聚合不正确"
    assert ts[0] == idx[0].value // (8 * 3_600_000_000_000) * (
        8 * 3_600_000_000_000)


def test_aggregate_to_clock_aligns_across_different_starts():
    """**跨起点对齐**：两个系列起点不同，也必须落进同一批绝对时间桶。

    这是实测踩过的坑：按"每 k 根取一个"切分时，两边网格整体错开，
    对齐后几乎无重叠，回归退化成 t = 0.00。
    """
    import pandas as pd
    base = pd.date_range("2024-01-01", periods=24, freq="1h", tz="UTC")
    a_idx = base[:16]                     # 00:00 起
    b_idx = base[3:19]                    # 03:00 起（错开 3 小时）
    ra = np.full(len(a_idx), 0.001)
    rb = np.full(len(b_idx), 0.002)
    oa, ta = protocol.aggregate_to_clock(a_idx, ra, "8h")
    ob, tb = protocol.aggregate_to_clock(b_idx, rb, "8h")
    common = np.intersect1d(ta, tb)
    assert len(common) >= 1, (
        f"跨起点聚合后没有共同桶：ta={ta} tb={tb} —— 说明按了偏移量而不是时钟分桶")
    # 共同桶里，A 应有 8 根、B 应有 5 根（b 起于 03:00，第一桶只含 03..07）
    i = int(np.where(ta == common[0])[0][0])
    j = int(np.where(tb == common[0])[0][0])
    assert abs(oa[i] - (1.001 ** 8 - 1)) < 1e-9
    assert abs(ob[j] - (1.002 ** 5 - 1)) < 1e-9


def test_aggregate_to_clock_rejects_bad_input():
    import pandas as pd
    idx = pd.date_range("2024-01-01", periods=5, freq="1h", tz="UTC")
    try:
        protocol.aggregate_to_clock(idx, np.zeros(4), "8h")
    except ValueError:
        pass
    else:
        raise AssertionError("长度不一致应抛 ValueError")
    rev = idx[::-1]
    try:
        protocol.aggregate_to_clock(rev, np.zeros(5), "8h")
    except ValueError:
        pass
    else:
        raise AssertionError("索引非单调应抛 ValueError")


# ==========================================================================
# 6. 样本封存守卫
# ==========================================================================
def test_holdout_guard_blocks_access():
    g = HoldoutGuard(n=1000, fraction=0.25)
    assert g.cut == 750
    assert g.train_slice() == slice(0, 750)
    try:
        g.holdout_slice()
    except HoldoutViolation:
        pass
    else:
        raise AssertionError("封存样本被无声访问了")
    try:
        g.assert_clean([10, 20, 760])
    except HoldoutViolation:
        pass
    else:
        raise AssertionError("越界索引未被拦截")
    g.assert_clean([10, 20, 700])           # 合法，不该抛


def test_holdout_guard_allows_when_explicit():
    g = HoldoutGuard(n=1000, fraction=0.25, allow=True)
    assert g.holdout_slice() == slice(750, 1000)
    g.assert_clean([999])


# ==========================================================================
def main() -> int:
    tests = [(k, v) for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    passed, failed = 0, []
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS  {name}")
            passed += 1
        except Exception as e:
            print(f"  FAIL  {name}: {type(e).__name__}: {e}")
            failed.append(name)
    print(f"\n{passed}/{len(tests)} 通过")
    if failed:
        print("失败：", ", ".join(failed))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
