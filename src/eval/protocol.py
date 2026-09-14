"""评估协议 —— 抗过拟合与抗多重试验的统计工具。

包含：
  - Walk-forward / Purged K-Fold + Embargo 切分
  - Circular Block Bootstrap 置信区间
  - Probabilistic Sharpe Ratio (PSR)
  - Deflated Sharpe Ratio (DSR)  —— 惩罚多重试验
  - PBO(CSCV)                    —— 直接给出"这策略是挑出来的"概率
  - Beta 中性化检验              —— 剥离 beta 后检验 alpha

设计要点（M2 的教训）：**原始收益/Sharpe 在本样本里没有区分度**。
2017-2026 是大牛市，随机做多策略的毛 Sharpe 也能到 1.0。
所以任何"显著"判定都必须建立在**剥离市场暴露后的超额收益**上。

不依赖 scipy。
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

EULER_GAMMA = 0.5772156649015329


# ==========================================================================
# 正态分布工具（不依赖 scipy）
# ==========================================================================
def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def norm_ppf(p: float) -> float:
    """标准正态分位数 —— Acklam 有理逼近，精度约 1e-9。"""
    if p <= 0.0 or p >= 1.0:
        raise ValueError("p 必须在 (0, 1) 内")
    a = (-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00)
    b = (-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01)
    c = (-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00)
    d = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00)
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
               ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
                ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    q = p - 0.5
    r = q * q
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / \
           (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)


# ==========================================================================
# 数据切分
# ==========================================================================
def walk_forward_splits(n: int, train: int, test: int, step: int | None = None,
                        embargo: int = 0):
    """滚动/扩张窗口切分。返回 (train_idx, test_idx) 序列。"""
    step = step or test
    out = []
    start = train
    while start + test <= n:
        tr = np.arange(start - train, start)
        te = np.arange(start + embargo, min(start + embargo + test, n))
        if len(te):
            out.append((tr, te))
        start += step
    return out


def purged_kfold(n: int, k: int, embargo: int = 0):
    """Purged K-Fold + Embargo（López de Prado 方案）。

    测试块两侧各剔除 embargo 个观测，防止标签重叠导致的泄漏。
    """
    if k < 2:
        raise ValueError("k 至少为 2")
    bounds = np.linspace(0, n, k + 1).astype(int)
    for i in range(k):
        te = np.arange(bounds[i], bounds[i + 1])
        mask = np.ones(n, dtype=bool)
        lo = max(0, bounds[i] - embargo)
        hi = min(n, bounds[i + 1] + embargo)
        mask[lo:hi] = False
        yield np.where(mask)[0], te


# ==========================================================================
# 现金流频率对齐（协议要求，见 PLAN §14.7）
# ==========================================================================
_FREQ_UNIT_NS = {"h": 3_600_000_000_000, "m": 60_000_000_000,
                 "d": 86_400_000_000_000, "s": 1_000_000_000}


def freq_to_ns(freq) -> int:
    """把 '8h' / '30m' / '1d' 或整数纳秒转成纳秒。"""
    if isinstance(freq, (int, np.integer)):
        return int(freq)
    s = str(freq).strip().lower()
    if s.isdigit():
        return int(s)
    unit = s[-1]
    if unit not in _FREQ_UNIT_NS:
        raise ValueError(f"无法解析的时间频率：{freq!r}（支持 8h/30m/1d 或整数纳秒）")
    return int(float(s[:-1]) * _FREQ_UNIT_NS[unit])


def aggregate_to_clock(index, returns: np.ndarray, freq="8h"):
    """按**绝对时间桶**把收益聚合到现金流频率。

    为什么必须做：当策略的现金流按固定间隔到账（如资金费每 8 小时结算一次），
    逐 bar 检验等于把同一笔现金流重复计了多次，有效样本量被虚增，
    t 统计量与 Sharpe 都会被高估（本项实测 t 高估 1.2~1.6 倍）。
    **这条要求不只适用于 t 统计量，也适用于 Sharpe/DSR/偏度/峰度。**

    必须按**时钟分桶**、而不是"每 k 根取一个"：策略与基准的起点不同时，
    按偏移量切分会让两边的网格整体错开，对齐后几乎无重叠
    （实测曾导致回归退化成 t = 0.00）。

    只依赖 numpy：时间戳走鸭子类型（DatetimeIndex 有 asi8），不 import pandas。
    """
    r = np.asarray(returns, dtype=float)
    if len(r) == 0:
        return r, np.asarray([], dtype=np.int64)
    step = freq_to_ns(freq)
    if hasattr(index, "asi8"):
        t = np.asarray(index.asi8, dtype=np.int64)
    else:
        t = np.asarray(index, dtype=np.int64)
    if len(t) != len(r):
        raise ValueError(f"索引长度 {len(t)} 与收益长度 {len(r)} 不一致")
    if not np.all(np.diff(t) >= 0):
        raise ValueError("索引必须单调不减，否则分桶会错")
    bucket = t // step
    starts = np.unique(bucket, return_index=True)[1]
    ends = np.append(starts[1:], len(r))
    csum = np.concatenate([[0.0], np.cumsum(np.log1p(r))])   # 复利聚合
    out = np.expm1(csum[ends] - csum[starts])
    return out, (bucket[starts] * step)


# ==========================================================================
# Bootstrap
# ==========================================================================
def circular_block_bootstrap(x: np.ndarray, block: int, n_samples: int = 2000,
                             seed: int = 0) -> np.ndarray:
    """循环块自助法的均值分布 —— 保留时序自相关结构。

    实现要点：先把序列按块预聚合成块和，再对块抽样求和。
    直接构造 (n_samples, n_blocks, block) 的索引数组在 2.6 万观测下要 400MB+，
    这里把它降到 O(n_samples × n_blocks)。
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n == 0:
        return np.array([])
    block = max(1, min(block, n))
    n_blocks = int(np.ceil(n / block))
    if n_blocks * block > n:
        x = np.concatenate([x, x[:n_blocks * block - n]])
    bsum = x.reshape(n_blocks, block).sum(axis=1)
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, n_blocks, size=(n_samples, n_blocks))
    return bsum[starts].sum(axis=1) / (n_blocks * block)


def bootstrap_p_value(x: np.ndarray, block: int = 24, n_samples: int = 2000,
                      seed: int = 0, null: float = 0.0) -> dict:
    """单边检验：均值是否显著大于 null。"""
    dist = circular_block_bootstrap(x, block, n_samples, seed)
    if len(dist) == 0:
        return {"mean": 0.0, "p_value": 1.0, "ci_low": 0.0, "ci_high": 0.0}
    return {
        "mean": float(np.mean(x)),
        "p_value": float((dist <= null).mean()),
        "ci_low": float(np.quantile(dist, 0.025)),
        "ci_high": float(np.quantile(dist, 0.975)),
        "null": null,
    }


def bootstrap_alpha(r: np.ndarray, b: np.ndarray, block: int = 24,
                    n_samples: int = 1000, seed: int = 0,
                    bars_per_year: int = 24 * 365) -> dict:
    """对**回归截距 alpha 本身**做块自助，而不是对残差做。

    为什么必须这样：OLS 残差的均值按构造恒等于 0，
    对它做 bootstrap 再检验「均值是否 > 0」，在数学上永远不可能通过。
    早期版本正是这么写的，导致协议第三条判据（ci_low > 0）结构性失效——
    任何策略都不可能被判为「有边际」。

    做法：对 (r, b) 成对做循环块重采样，每个重采样重跑一遍回归取截距。
    关键优化：重采样后的均值/协方差/方差都可以由**块级预聚合量**直接算出，
    不必真的重建数组再回归，复杂度降到 O(n_samples × n_blocks)。
    """
    r = np.asarray(r, dtype=float)
    b = np.asarray(b, dtype=float)
    n = min(len(r), len(b))
    r, b = r[:n], b[:n]
    ok = np.isfinite(r) & np.isfinite(b)
    r, b = r[ok], b[ok]
    n = len(r)
    if n < 3 * block:
        return {"alpha_mean": 0.0, "ci_low": 0.0, "ci_high": 0.0,
                "p_value_alpha_positive": 1.0, "n_samples": 0,
                "annualized_ci_low": 0.0, "annualized_ci_high": 0.0}

    block = max(1, min(block, n))
    n_blocks = int(np.ceil(n / block))
    need = n_blocks * block - n
    rp = np.concatenate([r, r[:need]]) if need else r
    bp = np.concatenate([b, b[:need]]) if need else b
    R = rp.reshape(n_blocks, block)
    B = bp.reshape(n_blocks, block)
    s_r, s_b = R.sum(1), B.sum(1)
    s_rr, s_bb, s_rb = (R * R).sum(1), (B * B).sum(1), (R * B).sum(1)

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n_blocks, size=(n_samples, n_blocks))
    N = float(n_blocks * block)
    Sr = s_r[idx].sum(1) / N
    Sb = s_b[idx].sum(1) / N
    Srr = s_rr[idx].sum(1) / N
    Sbb = s_bb[idx].sum(1) / N
    Srb = s_rb[idx].sum(1) / N

    var_b = Sbb - Sb * Sb
    cov = Srb - Sr * Sb
    beta = np.where(var_b > 1e-24, cov / np.where(var_b > 1e-24, var_b, 1.0), 0.0)
    alpha = Sr - beta * Sb                     # 每个重采样的回归截距

    return {
        "alpha_mean": float(np.mean(alpha)),
        "ci_low": float(np.quantile(alpha, 0.025)),
        "ci_high": float(np.quantile(alpha, 0.975)),
        "annualized_ci_low": float(np.expm1(np.quantile(alpha, 0.025) * bars_per_year))
        if abs(np.quantile(alpha, 0.025) * bars_per_year) < 20 else -1.0,
        "annualized_ci_high": float(np.expm1(np.quantile(alpha, 0.975) * bars_per_year))
        if abs(np.quantile(alpha, 0.975) * bars_per_year) < 20 else 1e9,
        "p_value_alpha_positive": float((alpha <= 0).mean()),
        "n_samples": int(n_samples),
        "block": int(block),
    }


# ==========================================================================
# Sharpe 家族
# ==========================================================================
def _moments(r: np.ndarray) -> tuple[float, float, float, int]:
    r = np.asarray(r, dtype=float)
    r = r[np.isfinite(r)]
    n = len(r)
    if n < 3:
        return 0.0, 0.0, 3.0, n
    mu, sd = r.mean(), r.std(ddof=1)
    if sd == 0:
        return 0.0, 0.0, 3.0, n
    skew = float(((r - mu) ** 3).mean() / sd ** 3)
    kurt = float(((r - mu) ** 4).mean() / sd ** 4)
    return float(mu / sd), skew, kurt, n


def probabilistic_sharpe(sr: float, n: int, skew: float, kurt: float,
                         sr_benchmark: float = 0.0) -> float:
    """PSR：在给定基准 Sharpe 下，真实 Sharpe 大于基准的概率。"""
    if n < 2:
        return 0.0
    denom = 1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr ** 2
    if denom <= 0:
        return 0.0
    z = (sr - sr_benchmark) * math.sqrt(n - 1) / math.sqrt(denom)
    return norm_cdf(z)


def expected_max_sharpe(sr_variance: float, n_trials: int) -> float:
    """N 次独立试验下，零技能时最大 Sharpe 的期望值。

    E[max SR] ≈ sqrt(V) · [ (1-γ)·Z⁻¹(1-1/N) + γ·Z⁻¹(1-1/(N·e)) ]
    V = 各次试验 Sharpe 的方差，γ = 欧拉-马歇罗尼常数。
    """
    if n_trials < 2 or sr_variance <= 0:
        return 0.0
    z1 = norm_ppf(1.0 - 1.0 / n_trials)
    z2 = norm_ppf(1.0 - 1.0 / (n_trials * math.e))
    return math.sqrt(sr_variance) * ((1 - EULER_GAMMA) * z1 + EULER_GAMMA * z2)


def deflated_sharpe(returns: np.ndarray, n_trials: int,
                    sr_variance: float | None = None) -> dict:
    """DSR：已按试验次数惩罚的概率。DSR > 0.95 才算显著。"""
    sr, skew, kurt, n = _moments(returns)
    var = sr_variance if sr_variance is not None else 1.0 / max(n, 1)
    sr0 = expected_max_sharpe(var, n_trials)
    return {
        "sharpe": round(sr, 4),
        "n_obs": n,
        "skew": round(skew, 4),
        "kurtosis": round(kurt, 4),
        "n_trials": n_trials,
        "expected_max_sharpe_null": round(sr0, 4),
        "dsr": round(probabilistic_sharpe(sr, n, skew, kurt, sr0), 4),
    }


# ==========================================================================
# PBO —— 回测过拟合概率（CSCV）
# ==========================================================================
def pbo_cscv(returns_matrix: np.ndarray, n_blocks: int = 16) -> dict:
    """CSCV：组合对称交叉验证，估计"样本内挑出的最优策略在样本外落到中位数以下"的概率。

    returns_matrix: (T, N) —— T 个观测，N 个候选策略。

    实现要点：C(16,8)=12870 个组合，若每个组合都重新拼接数组再算 Sharpe，
    总量是几十亿次浮点，跑不动。这里预先把每个块的和、平方和算好，
    组合内只需对块级统计量求和，复杂度降到 O(C × n_blocks × N)。
    """
    from itertools import combinations

    m = np.asarray(returns_matrix, dtype=float)
    T, N = m.shape
    if N < 2:
        return {"pbo": None, "reason": "候选策略少于 2 个"}
    if n_blocks % 2 or n_blocks < 4:
        raise ValueError("n_blocks 必须是 >=4 的偶数")
    b = T // n_blocks
    if b < 2:
        return {"pbo": None, "reason": "每个块不足 2 个观测"}

    trimmed = m[:n_blocks * b]
    blocks = trimmed.reshape(n_blocks, b, N)
    bsum = blocks.sum(axis=1)                    # (n_blocks, N)
    bsq = (blocks ** 2).sum(axis=1)              # (n_blocks, N)
    half = n_blocks // 2
    all_idx = np.arange(n_blocks)

    def sharpe_from_blocks(idx) -> np.ndarray:
        k = len(idx) * b
        s = bsum[idx].sum(axis=0)
        q = bsq[idx].sum(axis=0)
        mean = s / k
        var = np.maximum(q / k - mean ** 2, 1e-24)     # 总体方差
        return mean / np.sqrt(var)

    logits = []
    for combo in combinations(range(n_blocks), half):
        is_idx = np.fromiter(combo, dtype=int, count=half)
        oos_idx = np.setdiff1d(all_idx, is_idx, assume_unique=False)
        is_sr = sharpe_from_blocks(is_idx)
        if not np.any(np.isfinite(is_sr)):
            continue
        oos_sr = sharpe_from_blocks(oos_idx)
        best = int(np.nanargmax(is_sr))
        ranks = np.argsort(np.argsort(oos_sr))         # 0..N-1
        w = (ranks[best] + 0.5) / N                    # 相对排名 ∈ (0,1)
        w = min(max(w, 1e-6), 1 - 1e-6)
        logits.append(math.log(w / (1 - w)))

    if not logits:
        return {"pbo": None, "reason": "无有效组合"}
    logits = np.array(logits)
    return {
        "pbo": round(float((logits < 0).mean()), 4),
        "n_combinations": len(logits),
        "n_trials": N,
        "median_logit": round(float(np.median(logits)), 4),
    }


# ==========================================================================
# Beta 中性化 —— 剥离市场暴露后检验 alpha
# ==========================================================================
@dataclass
class AlphaTest:
    alpha: float
    alpha_ann: float
    beta: float
    t_stat: float
    p_value: float
    r_squared: float
    n: int

    def as_dict(self) -> dict:
        return {"alpha_per_bar": round(self.alpha, 8),
                "alpha_annualized": round(self.alpha_ann, 4),
                "beta": round(self.beta, 4),
                "t_stat": round(self.t_stat, 4),
                "p_value": round(self.p_value, 6),
                "r_squared": round(self.r_squared, 4),
                "n_obs": self.n}


def _align(r: np.ndarray, b: np.ndarray,
           r_index=None, b_index=None) -> tuple[np.ndarray, np.ndarray]:
    """按时间戳对齐两条收益序列。

    这是个**必须做**的步骤：策略收益从 warmup 之后才开始，而基准从第 1 根就有。
    早期版本按 min(len) 截断对齐，相当于拿策略的第 301..N 根去比基准的第 1..N-300 根，
    结果 beta 算成 0.003（基准自比基准都得不到 1），alpha 变成毫无意义的 +162%/年。
    """
    if r_index is None or b_index is None:
        n = min(len(r), len(b))
        return r[:n], b[:n]
    import pandas as pd

    sr = pd.Series(np.asarray(r, dtype=float), index=r_index)
    sb = pd.Series(np.asarray(b, dtype=float), index=b_index)
    sr = sr[~sr.index.duplicated(keep="first")]
    sb = sb[~sb.index.duplicated(keep="first")]
    j = pd.concat([sr.rename("r"), sb.rename("b")], axis=1, join="inner").dropna()
    return j["r"].to_numpy(), j["b"].to_numpy()


def alpha_vs_benchmark(returns: np.ndarray, bench: np.ndarray,
                       bars_per_year: int = 24 * 365,
                       hac_lags: int | None = None,
                       r_index=None, b_index=None) -> AlphaTest:
    """对基准做回归，返回 alpha 的 t 统计量（Newey-West 稳健标准误）。

    这是本项目"显著"判定的唯一入口：不看原始收益，只看剥离 beta 后的 alpha。
    两条序列必须按时间戳对齐（传 r_index / b_index）。
    """
    r, b = _align(np.asarray(returns, dtype=float),
                  np.asarray(bench, dtype=float), r_index, b_index)
    ok = np.isfinite(r) & np.isfinite(b)
    r, b = r[ok], b[ok]
    n = len(r)
    if n < 10:
        return AlphaTest(0, 0, 0, 0, 1.0, 0.0, n)

    X = np.column_stack([np.ones(n), b])
    coef, *_ = np.linalg.lstsq(X, r, rcond=None)
    resid = r - X @ coef
    k = X.shape[1]

    # Newey-West HAC 标准误
    lags = hac_lags if hac_lags is not None else int(np.floor(4 * (n / 100) ** (2 / 9)))
    lags = max(lags, 1)
    XtX_inv = np.linalg.inv(X.T @ X)
    S = (X * resid[:, None]).T @ (X * resid[:, None])
    for l in range(1, lags + 1):
        w = 1.0 - l / (lags + 1)
        G = (X[l:] * resid[l:, None]).T @ (X[:-l] * resid[:-l, None])
        S += w * (G + G.T)
    cov = XtX_inv @ S @ XtX_inv

    alpha = float(coef[0])
    se = float(math.sqrt(max(cov[0, 0], 1e-18)))
    t = alpha / se if se > 0 else 0.0
    p = 1.0 - norm_cdf(t)                    # 单边：alpha > 0
    ss_tot = float(((r - r.mean()) ** 2).sum())
    r2 = 1.0 - float((resid ** 2).sum()) / ss_tot if ss_tot > 0 else 0.0
    return AlphaTest(alpha, float(np.expm1(alpha * bars_per_year)) if abs(alpha) < 20 else 0.0,
                     float(coef[1]), float(t), float(p), r2, n)


# ==========================================================================
# 综合报告
# ==========================================================================
def evaluate_strategy(returns: np.ndarray, bench: np.ndarray, n_trials: int,
                      bars_per_year: int = 24 * 365, block: int = 24,
                      sr_variance: float | None = None,
                      alpha_threshold: float = 0.95,
                      r_index=None, b_index=None) -> dict:
    """一条策略的完整判定。

    判定规则（全部必须满足才算"有边际"）：
      1. 剥离 beta 后的 alpha，单边 p < 0.05
      2. DSR > 0.95（已按试验次数惩罚）
      3. Block bootstrap 的 alpha 置信区间下界 > 0
    """
    r = np.asarray(returns, dtype=float)
    at = alpha_vs_benchmark(r, bench, bars_per_year,
                            r_index=r_index, b_index=b_index)
    dsr = deflated_sharpe(r, n_trials, sr_variance)

    # alpha 的置信区间必须对**截距本身**做自助，不能对残差做。
    # 残差均值按构造恒为 0，对它做 bootstrap 再检验 >0 是永远不可能通过的。
    rr, bb = _align(r, np.asarray(bench, dtype=float), r_index, b_index)
    boot = bootstrap_alpha(rr, bb, block=block, bars_per_year=bars_per_year)

    passed = (at.p_value < 0.05) and (dsr["dsr"] > alpha_threshold) \
        and (boot["ci_low"] > 0)
    return {
        "alpha_test": at.as_dict(),
        "dsr": dsr,
        "alpha_bootstrap": boot,
        "verdict": "有边际" if passed else "无边际",
        "passed": bool(passed),
    }
