"""M5 在线自学习体 —— 每个 bar、每笔交易的结果都进入更新。

设计（Hedge / 指数梯度专家集成）：
  - 一组固定规则的「专家」，各自提案一个目标权重向量
  - 学习者维护专家权重 p_k，按**扣除成本后的净收益**做乘法更新：
        p_k ← p_k · exp(η · payoff_k) / Z
  - 组合目标权重 = Σ p_k · w_k
  - 于是每一根 bar 都有 K 笔"交易结果"进入学习，且不依赖批次划分——
    这就是"始终从所有进行的交易中学习"

关键设计约束（M2 的教训）：
  1. payoff 必须是**净**收益（含手续费/点差/冲击）。若用毛收益，
     学习者必然收敛到高频换仓，因为在无摩擦假设下换仓是免费的。
  2. 必须有不交易带宽（band）。混合权重每根 bar 都在微调，
     没有带宽就会退化成"每根 bar 精确再平衡"。
  3. 学习率 η 要小。专家集成天然会追逐近期噪声，η 过大等于过拟合最近几根 bar。

因果性：decide(view) 只用 view 暴露的 0..t 数据；所有状态更新都在读到
t 的收盘价之后、提出 t 的新目标之前完成。
"""
from __future__ import annotations

import numpy as np

from ..env.market_view import MarketView

BARS_PER_YEAR = 24 * 365


# ==========================================================================
# 专家
# ==========================================================================
class Expert:
    name = "expert"

    def weights(self, view: MarketView, symbols: list[str]) -> np.ndarray:
        raise NotImplementedError

    @staticmethod
    def _equal(symbols: list[str], on: bool) -> np.ndarray:
        n = len(symbols)
        return np.full(n, 1.0 / n) if on else np.zeros(n)


class FlatExpert(Expert):
    name = "flat"

    def weights(self, view, symbols):
        return np.zeros(len(symbols))


class LongExpert(Expert):
    name = "long_all"

    def weights(self, view, symbols):
        return self._equal(symbols, True)


class MomentumExpert(Expert):
    """过去 k 根累计收益为正则等权持有，否则空仓。"""

    def __init__(self, lookback: int):
        self.lookback = lookback
        self.name = f"mom_{lookback}"

    def weights(self, view, symbols):
        if view.available_history() <= self.lookback:
            return np.zeros(len(symbols))
        tot = 0.0
        for s in symbols:
            tot += float(view.returns(s, self.lookback).sum())
        return self._equal(symbols, tot > 0.0)


class ReversalExpert(Expert):
    """过去 k 根累计收益为负则买入（反转）。"""

    def __init__(self, lookback: int):
        self.lookback = lookback
        self.name = f"rev_{lookback}"

    def weights(self, view, symbols):
        if view.available_history() <= self.lookback:
            return np.zeros(len(symbols))
        tot = 0.0
        for s in symbols:
            tot += float(view.returns(s, self.lookback).sum())
        return self._equal(symbols, tot < 0.0)


class ShortExpert(Expert):
    """等权做空 —— 纯空仓基线。"""

    name = "short_all"

    def weights(self, view, symbols):
        n = len(symbols)
        return np.full(n, -1.0 / n)


class ShortMomentumExpert(Expert):
    """动量反转做空：过去 k 根累计收益为正则等权做空（追涨杀跌型空头）。"""

    def __init__(self, lookback: int):
        self.lookback = lookback
        self.name = f"short_mom_{lookback}"

    def weights(self, view, symbols):
        if view.available_history() <= self.lookback:
            return np.zeros(len(symbols))
        tot = 0.0
        for s in symbols:
            tot += float(view.returns(s, self.lookback).sum())
        return self._equal(symbols, tot > 0.0) * -1.0


class InverseVolExpert(Expert):
    """按逆波动率配权 —— 永远满仓，但倾斜到低波动标的。"""

    name = "inv_vol"

    def __init__(self, window: int = 168):
        self.window = window

    def weights(self, view, symbols):
        if view.available_history() <= self.window:
            return np.zeros(len(symbols))
        iv = np.array([1.0 / max(view.sigma(s, self.window), 1e-6) for s in symbols])
        return iv / iv.sum()


class SingleAssetExpert(Expert):
    def __init__(self, symbol: str):
        self.symbol = symbol
        self.name = f"only_{symbol}"

    def weights(self, view, symbols):
        w = np.zeros(len(symbols))
        if self.symbol in symbols:
            w[symbols.index(self.symbol)] = 1.0
        return w


# ==========================================================================
# 学习者
# ==========================================================================
class HedgeEnsemble:
    """在线学习的专家集成。实现 Agent 接口（decide -> dict 或 None）。"""

    def __init__(self, symbols: list[str], experts: list[Expert] | None = None,
                 cost_rate: np.ndarray | None = None, eta: float | None = None,
                 band: float = 0.03, max_exposure: float = 1.0,
                 funding: "FundingTable | None" = None):
        self.symbols = list(symbols)
        self.experts = experts or default_experts(self.symbols)
        self.K = len(self.experts)
        self.cost_rate = (np.asarray(cost_rate, dtype=float)
                          if cost_rate is not None
                          else np.full(len(self.symbols), 0.0012))
        # 学习率：经典 Hedge 尺度，再按经验压小一档以抑制追逐噪声
        self.eta = eta if eta is not None else 0.5 * np.sqrt(8 * np.log(self.K) / 1000)
        self.band = band
        self.max_exposure = max_exposure
        self.funding = funding                     # 可选：用于在 payoffs 里加 funding carry
        self.name = f"hedge_K{self.K}_eta{self.eta:.4f}_band{band}"

        self.p = np.full(self.K, 1.0 / self.K)
        self.hold = [np.zeros(len(self.symbols)) for _ in range(self.K)]
        self.eq = np.ones(self.K)
        self.prev_close: np.ndarray | None = None
        self.last_target: np.ndarray | None = None
        self.last_emitted: np.ndarray | None = None

        # 学习轨迹（供分析用）
        self.log: dict[str, list] = {
            "step": [], "p": [], "expert_returns": [], "turnover": [],
            "mixed_weight": [], "entropy": [],
        }

    # ------------------------------------------------------------------
    def _update_experts(self, close: np.ndarray,
                        prev_ts_ms: int | None = None) -> None:
        """用上一根到本根的收益更新各专家，然后做 Hedge 乘法更新。

        若 self.funding 非空且 prev_ts_ms 给出，则对每个持仓的 expert 加上
        funding carry 的 payoff（short 持仓 + 正费率 = 收钱）。
        """
        if self.prev_close is None or self.last_target is None:
            return
        r = close / self.prev_close - 1.0
        r = np.where(np.isfinite(r), r, 0.0)

        # funding carry：上一根到本根之间发生的资金费（8h 一次，
        # 多数 bar 上为 None → 0）。cash 变化 = -units × close × rate，
        # 对 short 持仓（units<0）+ 正费率 → f>0（**收钱**）。
        fr_arr = np.zeros(len(self.symbols))
        if self.funding is not None and prev_ts_ms is not None:
            for i, s in enumerate(self.symbols):
                rate = self.funding.rate_or_none(s, prev_ts_ms)
                if rate is not None:
                    fr_arr[i] = -rate   # 站在 short 一侧时是 +rate（符号见 carry.py 的 f_now = -u*px*rate）

        payoffs = np.empty(self.K)
        for k in range(self.K):
            target = self.last_target[k]
            # 从「上根漂移后的持仓」调到 target 的成本
            cost = float(self.cost_rate @ np.abs(target - self.hold[k]))
            # 价格 PnL + funding carry（无 funding 时 fr_arr 全 0，自动跳过）
            payoffs[k] = float(target @ r) - cost + float(target @ fr_arr)
            # 持仓漂移到 target 并随本根收益变动
            h = target
            d = 1.0 + float(h @ r)
            if d > 1e-9:
                h = h * (1.0 + r) / d
            self.hold[k] = h
            self.eq[k] *= max(1.0 + payoffs[k], 1e-9)

        # Hedge 乘法更新（对收益做过标准化，防止尺度导致的爆炸）
        scale = max(float(np.std(payoffs)), 1e-6)
        z = np.clip(payoffs / scale, -5.0, 5.0)
        self.p *= np.exp(self.eta * z)
        self.p = np.clip(self.p, 1e-12, None)
        self.p /= self.p.sum()

    # ------------------------------------------------------------------
    def decide(self, view: MarketView):
        close = np.array([view.close(s) for s in self.symbols], dtype=float)
        # 取上一根 bar 时刻的 funding（用上一根 ts，结算一般发生在 8h 边界）
        prev_ts_ms = None
        if view.step > 0:
            try:
                prev_ts_ms = int(view.timestamp(self.symbols[0]).timestamp() * 1000)
            except Exception:
                prev_ts_ms = None
        # 先结算上一根提出、本根持仓的那批目标 —— 每一根 bar 都学一次
        self._update_experts(close, prev_ts_ms=prev_ts_ms)
        # 本根收盘价已观测，立即推进 prev_close，
        # 否则跳过的 bar 会让下一根把两段收益当成一段来学
        self.prev_close = close

        targets = np.array([ex.weights(view, self.symbols) for ex in self.experts])
        self.last_target = targets
        mixed = self.p @ targets
        # 对称裁剪：单标的 |w_s| ≤ max_exposure。
        # 此前 clip(0, None) 把所有负权重清零，使训练目标只能在 {0, 多} 之间选；
        # 放开后允许做空，但 B 阶段把 max_exposure 提到 3.0 时也只需改这一处。
        mixed = np.clip(mixed, -self.max_exposure, self.max_exposure)
        # gross exposure 兜底：Σ|w_s| > max_exposure 时等比缩放
        gross = float(np.abs(mixed).sum())
        if gross > self.max_exposure and gross > 0:
            mixed = mixed * (self.max_exposure / gross)

        # 带宽：混合权重每根 bar 都在微调，没有带宽会退化成逐 bar 再平衡
        if self.last_emitted is not None and \
                float(np.abs(mixed - self.last_emitted).sum()) < self.band:
            self._log(view.step, mixed, 0.0)
            return None

        turnover = (0.0 if self.last_emitted is None
                    else float(np.abs(mixed - self.last_emitted).sum()))
        self.last_emitted = mixed
        self._log(view.step, mixed, turnover)
        return {s: float(w) for s, w in zip(self.symbols, mixed)}

    # ------------------------------------------------------------------
    def _log(self, step: int, mixed: np.ndarray, turnover: float) -> None:
        self.log["step"].append(step)
        self.log["p"].append(self.p.copy())
        self.log["mixed_weight"].append(mixed.copy())
        self.log["turnover"].append(turnover)
        self.log["entropy"].append(float(-(self.p * np.log(self.p)).sum()))

    # ------------------------------------------------------------------
    def expert_weight_history(self) -> np.ndarray:
        return np.array(self.log["p"]) if self.log["p"] else np.zeros((0, self.K))

    def expert_names(self) -> list[str]:
        return [e.name for e in self.experts]


def default_experts(symbols: list[str]) -> list[Expert]:
    ex: list[Expert] = [FlatExpert(), LongExpert(), InverseVolExpert(168)]
    for k in (24, 72, 168, 720):
        ex.append(MomentumExpert(k))
    for k in (24, 168):
        ex.append(ReversalExpert(k))
    ex.append(ShortExpert())                                # 纯空仓基线
    # 短动量（1h/4h/12h）+ 中动量（720h=30 天）：覆盖更细的下行反弹和月级回调
    for k in (1, 4, 12, 24, 168, 720):
        ex.append(ShortMomentumExpert(k))
    for s in symbols[:2]:
        ex.append(SingleAssetExpert(s))
    return ex
